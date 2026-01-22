import modal

app = modal.App("hpc-a2-gemm-modal")

# CUDA + nvcc image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "make")
)

gpu = "any"  # Modal will choose an available NVIDIA GPU


GEMM_CUH = r"""
#ifndef GEMM_CUH
#define GEMM_CUH

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Setter to provide dimensions (m, n, k) since the assignment gemm() signature doesn't include them.
void gemm_set_dims(int m, int n, int k);

// Assignment-required API:
// Computes: C <- alpha * op(A) * op(B) + beta * C   (in-place update of C)
// Row-major storage.
// If transposeA==false: A stored as m x k
// If transposeA==true : A stored as k x m  (so op(A)=A^T is m x k)
// If transposeB==false: B stored as k x n
// If transposeB==true : B stored as n x k  (so op(B)=B^T is k x n)
void gemm(float alpha, float* A, bool transposeA,
          float* B, bool transposeB,
          float beta, float* C);

#ifdef __cplusplus
}
#endif

#endif
"""


GEMM_CU = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "gemm.cuh"

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",          \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
        std::fflush(stderr);                                    \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// Dimensions are stored globally (for the required gemm() signature).
static int g_m = 0, g_n = 0, g_k = 0;

void gemm_set_dims(int m, int n, int k) {
    g_m = m; g_n = n; g_k = k;
}

__global__ void gemm_kernel(
    int m, int n, int k,
    float alpha,
    const float* __restrict__ A, bool transposeA,
    const float* __restrict__ B, bool transposeB,
    float beta,
    float* __restrict__ C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, n)

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    // op(A) is m x k
    // op(B) is k x n
    for (int kk = 0; kk < k; ++kk) {
        float a = transposeA ? A[kk * m + row] : A[row * k + kk];
        float b = transposeB ? B[col * k + kk] : B[kk * n + col];
        sum += a * b;
    }

    int idxC = row * n + col;
    C[idxC] = alpha * sum + beta * C[idxC];
}

// Required signature (exact name + args as assignment)
// Uses global dims set via gemm_set_dims().
void gemm(float alpha, float* A, bool transposeA,
          float* B, bool transposeB,
          float beta, float* C) {
    if (g_m <= 0 || g_n <= 0 || g_k <= 0) {
        std::fprintf(stderr, "gemm_set_dims(m,n,k) must be called before gemm().\n");
        std::exit(1);
    }

    dim3 block(16, 16);
    dim3 grid((g_n + block.x - 1) / block.x,
              (g_m + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(
        g_m, g_n, g_k,
        alpha,
        A, transposeA,
        B, transposeB,
        beta,
        C
    );

    CUDA_CHECK(cudaGetLastError());
}
"""


MAIN_CU = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include "gemm.cuh"

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",          \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
        std::fflush(stderr);                                    \
        std::exit(1);                                           \
    }                                                           \
} while(0)

static float frand() {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

// CPU reference: C = alpha*op(A)op(B) + beta*C
static void gemm_cpu_ref(
    int m, int n, int k,
    float alpha,
    const float* A, bool tA,
    const float* B, bool tB,
    float beta,
    float* C
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                float a = tA ? A[kk * m + i] : A[i * k + kk];
                float b = tB ? B[j * k + kk] : B[kk * n + j];
                sum += a * b;
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

static void fill_random(std::vector<float>& x) {
    for (auto& v : x) v = frand();
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) md = std::max(md, std::fabs(a[i] - b[i]));
    return md;
}

int main() {
    srand(0);

    // Test sizes (can change)
    const int m = 128;
    const int n = 256;
    const int k = 64;

    const float alpha = 1.25f;
    const float beta  = -0.75f;

    // Set dims for the assignment-signature gemm()
    gemm_set_dims(m, n, k);

    struct Case { bool tA; bool tB; const char* name; };
    Case cases[] = {
        {false, false, "C = a*A *B + b*C"},
        {true,  false, "C = a*A^T*B + b*C"},
        {false, true,  "C = a*A *B^T + b*C"},
        {true,  true,  "C = a*A^T*B^T + b*C"},
    };

    for (auto cs : cases) {
        bool tA = cs.tA;
        bool tB = cs.tB;

        // Storage sizes depend on transpose flags (how data is stored in memory)
        size_t sizeA = (size_t)(tA ? (k * m) : (m * k));
        size_t sizeB = (size_t)(tB ? (n * k) : (k * n));
        size_t sizeC = (size_t)(m * n);

        std::vector<float> hA(sizeA), hB(sizeB), hC(sizeC);
        fill_random(hA);
        fill_random(hB);
        fill_random(hC);

        std::vector<float> hCref(hC);
        gemm_cpu_ref(m, n, k, alpha, hA.data(), tA, hB.data(), tB, beta, hCref.data());

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice));

        // Assignment signature call
        gemm(alpha, dA, tA, dB, tB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        float diff = max_abs_diff(hC, hCref);
        std::printf("[%s] max |GPU-CPU| = %.6g\n", cs.name, diff);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        if (diff > 1e-3f) {
            std::fprintf(stderr, "FAILED: diff too large.\n");
            return 1;
        }
    }

    std::printf("All cases passed.\n");
    return 0;
}
"""


def _write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@app.function(image=image, gpu=gpu, timeout=60 * 20)
def build_and_run():
    import subprocess

    _write_file("gemm.cuh", GEMM_CUH)
    _write_file("gemm.cu", GEMM_CU)
    _write_file("main.cu", MAIN_CU)

    # compile + run (no Makefile needed)
    subprocess.run(["nvcc", "--version"], check=True)
    subprocess.run(["nvcc", "-O2", "-std=c++17", "main.cu", "gemm.cu", "-o", "gemm_test"], check=True)
    subprocess.run(["./gemm_test"], check=True)


@app.local_entrypoint()
def main():
    build_and_run.remote()
