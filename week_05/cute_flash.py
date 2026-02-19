# modal_a5_cute_flash.py
#
# Run:
#   modal run modal_a5_cute_flash.py

import modal
import subprocess
from pathlib import Path

app = modal.App("week5-cute-flashattention-demo")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
)

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cute/tensor.hpp>
using namespace cute;

#define CUDA_CHECK(call) do {                                  \
  cudaError_t err = (call);                                    \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
    std::exit(1);                                              \
  }                                                            \
} while(0)

// IMPORTANT:
// CUTE_HOST_DEVICE often expands to include an inlining specifier
// (e.g., __forceinline__). So DO NOT add "inline" or "static" here.
CUTE_HOST_DEVICE
float h2f(half h) { return __half2float(h); }

CUTE_HOST_DEVICE
half f2h(float f) { return __float2half(f); }

// Row-major layout for [R,C] with leading dimension ld (ld==C for contiguous).
CUTE_HOST_DEVICE
auto layout_rm(int R, int C, int ld) {
  return make_layout(make_shape(R, C), make_stride(ld, Int<1>{}));
}

// Naive FlashAttention forward (single head, single batch):
// O[m,d] = sum_n softmax( scale * dot(Q[m,:], K[n,:]) ) * V[n,d]
__global__ void flash_fwd_cute(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              half* __restrict__ O,
                              int M, int N, int Kdim, int D,
                              int ldq, int ldk, int ldv, int ldo,
                              float scale) {
  auto gQ = make_tensor(make_gmem_ptr(Q), layout_rm(M, Kdim, ldq)); // [M,K]
  auto gK = make_tensor(make_gmem_ptr(K), layout_rm(N, Kdim, ldk)); // [N,K]
  auto gV = make_tensor(make_gmem_ptr(V), layout_rm(N, D,    ldv)); // [N,D]
  auto gO = make_tensor(make_gmem_ptr(O), layout_rm(M, D,    ldo)); // [M,D]

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * D;
  if (tid >= total) return;

  int m = tid / D;
  int d = tid % D;

  // 1) row max
  float row_max = -INFINITY;
  for (int n = 0; n < N; ++n) {
    float dot = 0.f;
    #pragma unroll 1
    for (int k = 0; k < Kdim; ++k) {
      dot += h2f(gQ(m,k)) * h2f(gK(n,k));
    }
    float s = dot * scale;
    row_max = fmaxf(row_max, s);
  }

  // 2) denom
  float denom = 0.f;
  for (int n = 0; n < N; ++n) {
    float dot = 0.f;
    #pragma unroll 1
    for (int k = 0; k < Kdim; ++k) {
      dot += h2f(gQ(m,k)) * h2f(gK(n,k));
    }
    denom += expf(dot * scale - row_max);
  }
  denom = fmaxf(denom, 1e-20f);

  // 3) weighted sum
  float out = 0.f;
  for (int n = 0; n < N; ++n) {
    float dot = 0.f;
    #pragma unroll 1
    for (int k = 0; k < Kdim; ++k) {
      dot += h2f(gQ(m,k)) * h2f(gK(n,k));
    }
    float p = expf(dot * scale - row_max) / denom;
    out += p * h2f(gV(n,d));
  }

  gO(m,d) = f2h(out);
}

static void cpu_reference(const half* Q, const half* K, const half* V, half* O,
                          int M, int N, int Kdim, int D, float scale) {
  auto idxQ = [&](int m,int k){ return m*Kdim + k; };
  auto idxK = [&](int n,int k){ return n*Kdim + k; };
  auto idxV = [&](int n,int d){ return n*D + d; };
  auto idxO = [&](int m,int d){ return m*D + d; };

  for (int m = 0; m < M; ++m) {
    float mx = -INFINITY;
    for (int n = 0; n < N; ++n) {
      float dot = 0.f;
      for (int k = 0; k < Kdim; ++k) dot += __half2float(Q[idxQ(m,k)]) * __half2float(K[idxK(n,k)]);
      mx = fmaxf(mx, dot * scale);
    }
    float denom = 0.f;
    for (int n = 0; n < N; ++n) {
      float dot = 0.f;
      for (int k = 0; k < Kdim; ++k) dot += __half2float(Q[idxQ(m,k)]) * __half2float(K[idxK(n,k)]);
      denom += expf(dot * scale - mx);
    }
    denom = fmaxf(denom, 1e-20f);

    for (int d = 0; d < D; ++d) {
      float out = 0.f;
      for (int n = 0; n < N; ++n) {
        float dot = 0.f;
        for (int k = 0; k < Kdim; ++k) dot += __half2float(Q[idxQ(m,k)]) * __half2float(K[idxK(n,k)]);
        float p = expf(dot * scale - mx) / denom;
        out += p * __half2float(V[idxV(n,d)]);
      }
      O[idxO(m,d)] = __float2half(out);
    }
  }
}

int main() {
  const int M = 32;
  const int N = 64;
  const int Kdim = 32;
  const int D = 32;
  const float scale = 1.0f / sqrtf((float)Kdim);

  size_t bytesQ = (size_t)M * Kdim * sizeof(half);
  size_t bytesK = (size_t)N * Kdim * sizeof(half);
  size_t bytesV = (size_t)N * D    * sizeof(half);
  size_t bytesO = (size_t)M * D    * sizeof(half);

  half *hQ = (half*)malloc(bytesQ);
  half *hK = (half*)malloc(bytesK);
  half *hV = (half*)malloc(bytesV);
  half *hO = (half*)malloc(bytesO);
  half *hOref = (half*)malloc(bytesO);

  for (int i = 0; i < M*Kdim; ++i) hQ[i] = __float2half((i % 17) * 0.01f);
  for (int i = 0; i < N*Kdim; ++i) hK[i] = __float2half((i % 23) * 0.01f);
  for (int i = 0; i < N*D;    ++i) hV[i] = __float2half((i % 19) * 0.01f);

  half *dQ, *dK, *dV, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, bytesQ));
  CUDA_CHECK(cudaMalloc(&dK, bytesK));
  CUDA_CHECK(cudaMalloc(&dV, bytesV));
  CUDA_CHECK(cudaMalloc(&dO, bytesO));

  CUDA_CHECK(cudaMemcpy(dQ, hQ, bytesQ, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK, bytesK, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV, bytesV, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (M*D + threads - 1) / threads;
  flash_fwd_cute<<<blocks, threads>>>(dQ, dK, dV, dO,
                                      M, N, Kdim, D,
                                      Kdim, Kdim, D, D,
                                      scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hO, dO, bytesO, cudaMemcpyDeviceToHost));
  cpu_reference(hQ, hK, hV, hOref, M, N, Kdim, D, scale);

  float max_abs = 0.f;
  for (int i = 0; i < M*D; ++i) {
    float diff = fabsf(__half2float(hO[i]) - __half2float(hOref[i]));
    max_abs = fmaxf(max_abs, diff);
  }
  printf("Max abs diff vs CPU reference: %.6f\n", max_abs);
  printf("O[0,0]=%.6f  O[0,1]=%.6f  O[1,0]=%.6f\n",
         __half2float(hO[0]), __half2float(hO[1]), __half2float(hO[D]));

  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO));
  free(hQ); free(hK); free(hV); free(hO); free(hOref);
  return 0;
}
"""

@app.function(image=image, gpu="A10G", timeout=600)
def build_and_run():
    work = Path("/root/work")
    work.mkdir(parents=True, exist_ok=True)

    cutlass_dir = work / "cutlass"
    if not cutlass_dir.exists():
        subprocess.check_call(
            ["bash", "-lc", f"cd {work} && git clone --depth 1 https://github.com/NVIDIA/cutlass.git"]
        )

    src = work / "flash_cute_demo.cu"
    src.write_text(CUDA_SRC)

    cmd = f"""
    set -euxo pipefail
    cd {work}
    nvcc -O3 --std=c++17 \
      -I{cutlass_dir}/include \
      -gencode arch=compute_80,code=compute_80 \
      flash_cute_demo.cu -o flash_cute_demo
    ./flash_cute_demo
    """
    subprocess.check_call(["bash", "-lc", cmd])


@app.local_entrypoint()
def main():
    build_and_run.remote()
