import modal

# -----------------------------
# Embedded source files (C + CUDA)
# -----------------------------

FLASH_H = r"""
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void attention_naive_cpu(const float* Q, const float* K, const float* V,
                         float* O, float* L, int N, int d);

void flash_forward_cpu(const float* Q, const float* K, const float* V,
                       float* O, float* L, int N, int d, int Br, int Bc);

void flash_forward_cuda(const float* dQ, const float* dK, const float* dV,
                        float* dO, float* dL, int N, int d, int Br, int Bc);

#ifdef __cplusplus
}
#endif
"""

FLASH_CPU_C = r"""
#include "flash.h"
#include <math.h>
#include <stdlib.h>

static inline float dot_scaled(const float* a, const float* b, int d, float scale) {
    float acc = 0.0f;
    for (int k = 0; k < d; k++) acc += a[k] * b[k];
    return acc * scale;
}

// Naive reference: O = softmax(QK^T/sqrt(d)) V, L = logsumexp per row
void attention_naive_cpu(const float* Q, const float* K, const float* V,
                         float* O, float* L, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);

    float* scores = (float*)malloc(sizeof(float) * (size_t)N);
    float* probs  = (float*)malloc(sizeof(float) * (size_t)N);

    for (int i = 0; i < N; i++) {
        const float* qi = Q + (size_t)i * d;

        float m = -INFINITY;
        for (int j = 0; j < N; j++) {
            scores[j] = dot_scaled(qi, K + (size_t)j * d, d, scale);
            if (scores[j] > m) m = scores[j];
        }

        float l = 0.0f;
        for (int j = 0; j < N; j++) {
            probs[j] = expf(scores[j] - m);
            l += probs[j];
        }

        float* oi = O + (size_t)i * d;
        for (int k = 0; k < d; k++) oi[k] = 0.0f;

        for (int j = 0; j < N; j++) {
            float w = probs[j] / l;
            const float* vj = V + (size_t)j * d;
            for (int k = 0; k < d; k++) oi[k] += w * vj[k];
        }

        L[i] = m + logf(l);
    }

    free(scores);
    free(probs);
}

// FlashAttention-2 Algorithm 1 forward pass (CPU, correctness-first)
void flash_forward_cpu(const float* Q, const float* K, const float* V,
                       float* O, float* L, int N, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    float* scores = (float*)malloc(sizeof(float) * (size_t)Bc);

    for (int ti = 0; ti < Tr; ti++) {
        int row_start = ti * Br;
        int row_end = row_start + Br;
        if (row_end > N) row_end = N;
        int curBr = row_end - row_start;

        float* m = (float*)malloc(sizeof(float) * (size_t)curBr);
        float* l = (float*)malloc(sizeof(float) * (size_t)curBr);
        float* Oacc = (float*)malloc(sizeof(float) * (size_t)curBr * d);

        for (int r = 0; r < curBr; r++) {
            m[r] = -INFINITY;
            l[r] = 0.0f;
            for (int k = 0; k < d; k++) Oacc[(size_t)r * d + k] = 0.0f;
        }

        for (int tj = 0; tj < Tc; tj++) {
            int col_start = tj * Bc;
            int col_end = col_start + Bc;
            if (col_end > N) col_end = N;
            int curBc = col_end - col_start;

            for (int r = 0; r < curBr; r++) {
                int i = row_start + r;
                const float* qi = Q + (size_t)i * d;

                float rowmax = -INFINITY;
                for (int jj = 0; jj < curBc; jj++) {
                    scores[jj] = dot_scaled(qi, K + (size_t)(col_start + jj) * d, d, scale);
                    if (scores[jj] > rowmax) rowmax = scores[jj];
                }

                float m_new = fmaxf(m[r], rowmax);

                float l_new = l[r] * expf(m[r] - m_new);
                for (int jj = 0; jj < curBc; jj++) {
                    l_new += expf(scores[jj] - m_new);
                }

                float scale_old = expf(m[r] - m_new);
                for (int k = 0; k < d; k++) {
                    Oacc[(size_t)r * d + k] *= scale_old;
                }

                for (int jj = 0; jj < curBc; jj++) {
                    float p = expf(scores[jj] - m_new);
                    const float* vj = V + (size_t)(col_start + jj) * d;
                    for (int k = 0; k < d; k++) {
                        Oacc[(size_t)r * d + k] += p * vj[k];
                    }
                }

                m[r] = m_new;
                l[r] = l_new;
            }
        }

        for (int r = 0; r < curBr; r++) {
            int i = row_start + r;
            float inv_l = 1.0f / l[r];
            for (int k = 0; k < d; k++) {
                O[(size_t)i * d + k] = Oacc[(size_t)r * d + k] * inv_l;
            }
            L[i] = m[r] + logf(l[r]);
        }

        free(m);
        free(l);
        free(Oacc);
    }

    free(scores);
}
"""

FLASH_CUDA_CU = r"""
#include "flash.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(x) do {                                   \
    cudaError_t err = (x);                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                              \
    }                                                         \
} while(0)

__device__ __forceinline__ float dot_scaled(const float* a, const float* b, int d, float scale) {
    float acc = 0.f;
    for (int k = 0; k < d; k++) acc += a[k] * b[k];
    return acc * scale;
}

// One block = one Qi tile. One thread = one row in Qi tile.
// Shared memory holds K and V tiles (Bc*d each).
__global__ void flash_forward_kernel(const float* Q, const float* K, const float* V,
                                     float* O, float* L, int N, int d, int Br, int Bc) {
    int tile_i = (int)blockIdx.x;
    int r = (int)threadIdx.x;
    int row_start = tile_i * Br;
    int i = row_start + r;

    if (r >= Br || i >= N) return;

    float scale = rsqrtf((float)d);

    extern __shared__ float smem[];
    float* Ksh = smem;                 // size Bc*d
    float* Vsh = Ksh + Bc * d;         // size Bc*d

    float m = -INFINITY;
    float l = 0.f;

    // Simple correctness-first kernel supports d <= 256
    float Orow[256];
    for (int k = 0; k < d; k++) Orow[k] = 0.f;

    int Tc = (N + Bc - 1) / Bc;
    const float* qi = Q + (size_t)i * d;

    for (int tj = 0; tj < Tc; tj++) {
        int col_start = tj * Bc;
        int curBc = Bc;
        if (col_start + curBc > N) curBc = N - col_start;

        // load K and V tiles cooperatively
        int t = (int)threadIdx.x;
        int total = curBc * d;

        for (int idx = t; idx < total; idx += (int)blockDim.x) {
            int jj = idx / d;
            int kk = idx % d;
            Ksh[idx] = K[(size_t)(col_start + jj) * d + kk];
            Vsh[idx] = V[(size_t)(col_start + jj) * d + kk];
        }
        __syncthreads();

        // rowmax for this row over tile
        float rowmax = -INFINITY;
        for (int jj = 0; jj < curBc; jj++) {
            float s = dot_scaled(qi, &Ksh[jj * d], d, scale);
            rowmax = fmaxf(rowmax, s);
        }
        float m_new = fmaxf(m, rowmax);

        // rescale old accumulators
        float scale_old = expf(m - m_new);
        l = l * scale_old;
        for (int k = 0; k < d; k++) Orow[k] *= scale_old;

        // accumulate this tile
        for (int jj = 0; jj < curBc; jj++) {
            float s = dot_scaled(qi, &Ksh[jj * d], d, scale);
            float p = expf(s - m_new);
            l += p;

            const float* vj = &Vsh[jj * d];
            for (int k = 0; k < d; k++) {
                Orow[k] += p * vj[k];
            }
        }

        m = m_new;
        __syncthreads();
    }

    // finalize
    float inv_l = 1.f / l;
    for (int k = 0; k < d; k++) O[(size_t)i * d + k] = Orow[k] * inv_l;
    L[i] = m + logf(l);
}

void flash_forward_cuda(const float* dQ, const float* dK, const float* dV,
                        float* dO, float* dL, int N, int d, int Br, int Bc) {
    if (d > 256) {
        fprintf(stderr, "Error: this simple kernel supports d <= 256. Got d=%d\n", d);
        exit(1);
    }
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr);
    dim3 block(Br);

    size_t shmem = (size_t)(Bc * d * 2) * sizeof(float); // K + V
    flash_forward_kernel<<<grid, block, shmem>>>(dQ, dK, dV, dO, dL, N, d, Br, Bc);
    CHECK_CUDA(cudaGetLastError());
}
"""

MAIN_C = r"""
#include "flash.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

static float frand_unit(unsigned int* state) {
    *state = (*state * 1664525u + 1013904223u);
    return (float)((*state >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
}

static void fill_random(float* x, size_t n, unsigned int seed) {
    unsigned int st = seed;
    for (size_t i = 0; i < n; i++) {
        float r = frand_unit(&st);
        x[i] = (r - 0.5f) * 2.0f;
    }
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    int N  = (argc > 1) ? atoi(argv[1]) : 256;
    int d  = (argc > 2) ? atoi(argv[2]) : 64;
    int Br = (argc > 3) ? atoi(argv[3]) : 64;
    int Bc = (argc > 4) ? atoi(argv[4]) : 64;

    printf("N=%d d=%d Br=%d Bc=%d\n", N, d, Br, Bc);

    size_t qkv_bytes = (size_t)N * d * sizeof(float);
    size_t out_bytes = (size_t)N * d * sizeof(float);
    size_t l_bytes   = (size_t)N * sizeof(float);

    float* Q = (float*)malloc(qkv_bytes);
    float* K = (float*)malloc(qkv_bytes);
    float* V = (float*)malloc(qkv_bytes);

    float* O_naive = (float*)malloc(out_bytes);
    float* L_naive = (float*)malloc(l_bytes);

    float* O_cpu = (float*)malloc(out_bytes);
    float* L_cpu = (float*)malloc(l_bytes);

    float* O_gpu = (float*)malloc(out_bytes);
    float* L_gpu = (float*)malloc(l_bytes);

    fill_random(Q, (size_t)N * d, 1u);
    fill_random(K, (size_t)N * d, 2u);
    fill_random(V, (size_t)N * d, 3u);

    attention_naive_cpu(Q, K, V, O_naive, L_naive, N, d);
    flash_forward_cpu(Q, K, V, O_cpu, L_cpu, N, d, Br, Bc);

    printf("[CPU] max|O_naive - O_flash| = %.6g\n", max_abs_diff(O_naive, O_cpu, (size_t)N*d));
    printf("[CPU] max|L_naive - L_flash| = %.6g\n", max_abs_diff(L_naive, L_cpu, (size_t)N));

    float *dQ, *dK, *dV, *dO, *dL;
    cudaMalloc((void**)&dQ, qkv_bytes);
    cudaMalloc((void**)&dK, qkv_bytes);
    cudaMalloc((void**)&dV, qkv_bytes);
    cudaMalloc((void**)&dO, out_bytes);
    cudaMalloc((void**)&dL, l_bytes);

    cudaMemcpy(dQ, Q, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, qkv_bytes, cudaMemcpyHostToDevice);

    flash_forward_cuda(dQ, dK, dV, dO, dL, N, d, Br, Bc);
    cudaDeviceSynchronize();

    cudaMemcpy(O_gpu, dO, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(L_gpu, dL, l_bytes, cudaMemcpyDeviceToHost);

    printf("[GPU] max|O_naive - O_flash| = %.6g\n", max_abs_diff(O_naive, O_gpu, (size_t)N*d));
    printf("[GPU] max|L_naive - L_flash| = %.6g\n", max_abs_diff(L_naive, L_gpu, (size_t)N));

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);

    free(Q); free(K); free(V);
    free(O_naive); free(L_naive);
    free(O_cpu); free(L_cpu);
    free(O_gpu); free(L_gpu);

    return 0;
}
"""

# -----------------------------
# Modal app: write files -> nvcc compile -> run
# -----------------------------

app = modal.App("a4-flashattention-singlefile")

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("build-essential")
)

@app.function(image=image, gpu="A10", timeout=60 * 20)
def run(N: int = 256, d: int = 64, Br: int = 64, Bc: int = 64):
    import os, subprocess, textwrap

    workdir = "/root/project"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    def write(path: str, content: str):
        with open(path, "w") as f:
            f.write(content)

    # Write all sources
    write("flash.h", FLASH_H)
    write("flash_cpu.c", FLASH_CPU_C)
    write("flash_cuda.cu", FLASH_CUDA_CU)
    write("main.c", MAIN_C)

    print("Wrote source files:")
    subprocess.run(["ls", "-la"], check=True)

    print("\nGPU check:")
    subprocess.run(["nvidia-smi"], check=True)

    print("\nCompiling (no Makefile)...")
    # nvcc handles both C and CUDA compilation + link
    subprocess.run(
        [
            "nvcc", "-O2",
            "main.c",
            "flash_cpu.c",
            "flash_cuda.cu",
            "-o", "flash_test",
            "-lm",
        ],
        check=True,
    )

    print("\nRunning:")
    subprocess.run(["./flash_test", str(N), str(d), str(Br), str(Bc)], check=True)

@app.local_entrypoint()
def main():
    # Change these if needed
    run.remote(256, 64, 64, 64)
