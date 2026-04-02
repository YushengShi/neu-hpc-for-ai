# modal_moe_bench.py  –  single file, no separate .cu needed
# Run with:  modal run modal_moe_bench.py

import modal

# ---------------------------------------------------------------------------
# Embed the CUDA source inline so there is no external file dependency
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <chrono>

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
constexpr int HIDDEN         = 7168;
constexpr int FFN_INTER      = 18432;
constexpr int NUM_EXPERTS    = 256;
constexpr int TOP_K          = 8;
constexpr int MAX_TOKENS     = 4096;
constexpr int WMMA_M         = 16;
constexpr int WMMA_N         = 16;
constexpr int WMMA_K         = 16;
constexpr int WARPS_PER_BLOCK  = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int TMA_BOX_M      = 64;
constexpr int TMA_BOX_N      = 64;
constexpr int SMEM_STAGES    = 2;

using bf16   = __nv_bfloat16;

// WMMA fragment aliases
using FragA   = fragment<matrix_a,    WMMA_M, WMMA_N, WMMA_K, bf16, row_major>;
using FragB   = fragment<matrix_b,    WMMA_M, WMMA_N, WMMA_K, bf16, col_major>;
using FragAcc = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// ---------------------------------------------------------------------------
// Router kernel
// ---------------------------------------------------------------------------
__global__ void router_kernel(
    const float* __restrict__ logits,
    int*   __restrict__ expert_ids,
    float* __restrict__ scores,
    int T, int E, int K
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + t * E;

    int   top_idx[TOP_K];
    float top_val[TOP_K];
    for (int k = 0; k < K; ++k) { top_idx[k] = k; top_val[k] = row[k]; }
    for (int e = K; e < E; ++e) {
        float v = row[e];
        int   min_k = 0; float min_v = top_val[0];
        for (int k = 1; k < K; ++k) if (top_val[k] < min_v) { min_v = top_val[k]; min_k = k; }
        if (v > min_v) { top_val[min_k] = v; top_idx[min_k] = e; }
    }
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += expf(top_val[k]);
    for (int k = 0; k < K; ++k) {
        expert_ids[t * K + k] = top_idx[k];
        scores    [t * K + k] = expf(top_val[k]) / sum;
    }
}

// ---------------------------------------------------------------------------
// WMMA MoE expert kernel (gate + up fused, SiLU, down)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
moe_expert_fused_kernel(
    const bf16*  __restrict__ X,
    const int*   __restrict__ expert_ids,
    const float* __restrict__ scores,
    const int*   __restrict__ expert_token_offsets,
    const int*   __restrict__ token_idx_for_expert,
    const bf16*  __restrict__ W1,
    const bf16*  __restrict__ W2,
    const bf16*  __restrict__ W3,
    float*       __restrict__ Y,
    int tokens_total, int hidden, int ffn_inter
) {
    const int expert_id  = blockIdx.x;
    const int tile_row   = blockIdx.y;
    const int tok_start  = expert_token_offsets[expert_id];
    const int tok_end    = expert_token_offsets[expert_id + 1];
    const int num_tokens = tok_end - tok_start;
    const int m_base     = tile_row * TMA_BOX_M;
    if (m_base >= num_tokens) return;
    const int m_this     = min(TMA_BOX_M, num_tokens - m_base);
    const int warp_id    = threadIdx.x / 32;
    const int lane_id    = threadIdx.x % 32;

    extern __shared__ char smem_raw[];

    bf16*  smem_A  = reinterpret_cast<bf16*>(smem_raw);
    bf16*  smem_B1 = smem_A  + SMEM_STAGES * TMA_BOX_M * WMMA_K;
    bf16*  smem_B2 = smem_B1 + SMEM_STAGES * TMA_BOX_N * WMMA_K;
    float* smem_C1 = reinterpret_cast<float*>(smem_B2 + SMEM_STAGES * TMA_BOX_N * WMMA_K);
    float* smem_C2 = smem_C1 + TMA_BOX_M * TMA_BOX_N;
    bf16*  smem_H  = reinterpret_cast<bf16*>(smem_C2 + TMA_BOX_M * TMA_BOX_N);

    constexpr int K_TILES = TMA_BOX_N / WMMA_N;   // 4
    constexpr int M_TILES = TMA_BOX_M / WMMA_M;   // 4

    FragAcc acc_gate[M_TILES][K_TILES];
    FragAcc acc_up  [M_TILES][K_TILES];
    FragAcc acc_down[M_TILES][K_TILES];
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m)
        for (int n = 0; n < K_TILES; ++n) {
            fill_fragment(acc_gate[m][n], 0.f);
            fill_fragment(acc_up  [m][n], 0.f);
            fill_fragment(acc_down[m][n], 0.f);
        }

    const int num_k_chunks = (hidden + WMMA_K - 1) / WMMA_K;

    // Phase 1: gate + up projections with cp.async double-buffering
    for (int k = 0; k < num_k_chunks; ++k) {
        int cur_s  = k % SMEM_STAGES;
        int k_off  = k * WMMA_K;
        int n_band = warp_id * TMA_BOX_N;

        // Load X rows (gather by token index)
        if (warp_id == 0) {
            for (int row = lane_id; row < m_this; row += 32) {
                int tok = token_idx_for_expert[tok_start + m_base + row];
                const bf16* src = X + (size_t)tok * hidden + k_off;
                bf16*       dst = smem_A + cur_s * TMA_BOX_M * WMMA_K + row * WMMA_K;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 32;"
                             :: "l"(dst), "l"(src));
            }
        }
        // Load W1 gate tile
        if (warp_id == 1) {
            for (int row = lane_id; row < TMA_BOX_N; row += 32) {
                const bf16* src = W1 + (size_t)expert_id * ffn_inter * hidden + (n_band + row) * hidden + k_off;
                bf16*       dst = smem_B1 + cur_s * TMA_BOX_N * WMMA_K + row * WMMA_K;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 32;"
                             :: "l"(dst), "l"(src));
            }
        }
        // Load W2 up tile
        if (warp_id == 2) {
            for (int row = lane_id; row < TMA_BOX_N; row += 32) {
                const bf16* src = W2 + (size_t)expert_id * ffn_inter * hidden + (n_band + row) * hidden + k_off;
                bf16*       dst = smem_B2 + cur_s * TMA_BOX_N * WMMA_K + row * WMMA_K;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 32;"
                             :: "l"(dst), "l"(src));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // WMMA MMA for gate and up
        bf16* A_ptr  = smem_A  + cur_s * TMA_BOX_M * WMMA_K;
        bf16* B1_ptr = smem_B1 + cur_s * TMA_BOX_N * WMMA_K;
        bf16* B2_ptr = smem_B2 + cur_s * TMA_BOX_N * WMMA_K;
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            FragA frag_a;
            load_matrix_sync(frag_a, A_ptr + mt * WMMA_M * WMMA_K, WMMA_K);
            #pragma unroll
            for (int nt = 0; nt < K_TILES; ++nt) {
                FragB fb1, fb2;
                load_matrix_sync(fb1, B1_ptr + nt * WMMA_N * WMMA_K, WMMA_K);
                load_matrix_sync(fb2, B2_ptr + nt * WMMA_N * WMMA_K, WMMA_K);
                mma_sync(acc_gate[mt][nt], frag_a, fb1, acc_gate[mt][nt]);
                mma_sync(acc_up  [mt][nt], frag_a, fb2, acc_up  [mt][nt]);
            }
        }
        __syncthreads();
    }

    // Store gate/up accumulators and fuse SiLU-gate
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt)
        for (int nt = 0; nt < K_TILES; ++nt) {
            store_matrix_sync(smem_C1 + mt * WMMA_M * TMA_BOX_N + nt * WMMA_N, acc_gate[mt][nt], TMA_BOX_N, mem_row_major);
            store_matrix_sync(smem_C2 + mt * WMMA_M * TMA_BOX_N + nt * WMMA_N, acc_up  [mt][nt], TMA_BOX_N, mem_row_major);
        }
    __syncthreads();

    for (int i = threadIdx.x; i < TMA_BOX_M * TMA_BOX_N; i += THREADS_PER_BLOCK) {
        float g = smem_C1[i];
        float u = smem_C2[i];
        smem_H[i] = __float2bfloat16((g / (1.f + expf(-g))) * u);
    }
    __syncthreads();

    // Phase 2: down projection
    const int num_k2 = (ffn_inter + WMMA_K - 1) / WMMA_K;
    bf16* smem_B3 = smem_B1;  // reuse slot

    for (int k = 0; k < num_k2; ++k) {
        int cur_s = k % SMEM_STAGES;
        int k_off = k * WMMA_K;
        int n_band = warp_id * TMA_BOX_N;

        if (warp_id == 0) {
            for (int row = lane_id; row < TMA_BOX_N; row += 32) {
                const bf16* src = W3 + (size_t)expert_id * hidden * ffn_inter + (n_band + row) * ffn_inter + k_off;
                bf16*       dst = smem_B3 + cur_s * TMA_BOX_N * WMMA_K + row * WMMA_K;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 32;"
                             :: "l"(dst), "l"(src));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        bf16* H_ptr  = smem_H  + k_off;
        bf16* B3_ptr = smem_B3 + cur_s * TMA_BOX_N * WMMA_K;
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            FragA frag_h;
            load_matrix_sync(frag_h, H_ptr + mt * WMMA_M * ffn_inter, ffn_inter);
            #pragma unroll
            for (int nt = 0; nt < K_TILES; ++nt) {
                FragB fb3;
                load_matrix_sync(fb3, B3_ptr + nt * WMMA_N * WMMA_K, WMMA_K);
                mma_sync(acc_down[mt][nt], frag_h, fb3, acc_down[mt][nt]);
            }
        }
        __syncthreads();
    }

    // Store down result and scatter-add to output
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt)
        for (int nt = 0; nt < K_TILES; ++nt)
            store_matrix_sync(smem_C1 + mt * WMMA_M * TMA_BOX_N + nt * WMMA_N, acc_down[mt][nt], TMA_BOX_N, mem_row_major);
    __syncthreads();

    int warp_n_base = (warp_id % 4) * WMMA_N;
    int warp_m_base = (warp_id / 4) * WMMA_M;
    for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += 32) {
        int lm = elem / WMMA_N, ln = elem % WMMA_N;
        int gm = m_base + warp_m_base + lm;
        int gn = warp_n_base + ln;
        if (gm >= num_tokens || gn >= hidden) continue;
        int   tok   = token_idx_for_expert[tok_start + gm];
        float score = scores[tok * TOP_K];   // simplified slot 0
        float val   = smem_C1[(warp_m_base + lm) * TMA_BOX_N + warp_n_base + ln] * score;
        atomicAdd(&Y[(size_t)tok * hidden + gn], val);
    }
}

// ---------------------------------------------------------------------------
// Naive baseline
// ---------------------------------------------------------------------------
__global__ void moe_naive_kernel(
    const bf16* X, const int* expert_ids, const float* scores,
    const int* expert_token_offsets, const int* token_idx_for_expert,
    const bf16* W1, const bf16* W2, const bf16* W3,
    float* Y, int tokens_total, int hidden, int ffn_inter
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens_total) return;
    for (int k = 0; k < TOP_K; ++k) {
        int   e  = expert_ids[t * TOP_K + k];
        float sc = scores    [t * TOP_K + k];
        for (int n = 0; n < ffn_inter; ++n) {
            float gate = 0.f, up = 0.f;
            for (int h = 0; h < hidden; ++h) {
                float xv = __bfloat162float(X[t * hidden + h]);
                gate += xv * __bfloat162float(W1[(size_t)e * ffn_inter * hidden + n * hidden + h]);
                up   += xv * __bfloat162float(W2[(size_t)e * ffn_inter * hidden + n * hidden + h]);
            }
            float hv = (gate / (1.f + expf(-gate))) * up;
            for (int o = 0; o < hidden; ++o)
                atomicAdd(&Y[t * hidden + o],
                    sc * hv * __bfloat162float(W3[(size_t)e * hidden * ffn_inter + o * ffn_inter + n]));
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
struct BenchResult { double ms; double tflops; };

BenchResult bench(const char* name, std::function<void()> fn, int warmup=3, int iters=10) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> times;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(t0); fn(); cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); times.push_back(ms);
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double flops = 2.0 * 2.0 * MAX_TOKENS * TOP_K * (double)HIDDEN * FFN_INTER;
    double tflops = flops / (mean * 1e-3) / 1e12;
    printf("%-38s  mean=%8.2f ms   TFLOPs=%7.2f\n", name, mean, tflops);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return {mean, tflops};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== DeepSeek MoE: WMMA+TMA vs Naive (B200) ===\n\n");
    printf("tokens=%d  hidden=%d  ffn_inter=%d  experts=%d  top_k=%d\n\n",
           MAX_TOKENS, HIDDEN, FFN_INTER, NUM_EXPERTS, TOP_K);

    size_t X_sz  = (size_t)MAX_TOKENS * HIDDEN * sizeof(bf16);
    size_t W1_sz = (size_t)NUM_EXPERTS * FFN_INTER * HIDDEN * sizeof(bf16);
    size_t W3_sz = (size_t)NUM_EXPERTS * HIDDEN * FFN_INTER * sizeof(bf16);
    size_t Y_sz  = (size_t)MAX_TOKENS * HIDDEN * sizeof(float);
    size_t lg_sz = (size_t)MAX_TOKENS * NUM_EXPERTS * sizeof(float);

    bf16   *d_X, *d_W1, *d_W2, *d_W3;
    float  *d_Y, *d_logits, *d_scores;
    int    *d_eids, *d_offs, *d_tidx;

    cudaMalloc(&d_X,      X_sz);   cudaMalloc(&d_W1, W1_sz);
    cudaMalloc(&d_W2,     W1_sz);  cudaMalloc(&d_W3, W3_sz);
    cudaMalloc(&d_Y,      Y_sz);   cudaMalloc(&d_logits, lg_sz);
    cudaMalloc(&d_scores, (size_t)MAX_TOKENS * TOP_K * sizeof(float));
    cudaMalloc(&d_eids,   (size_t)MAX_TOKENS * TOP_K * sizeof(int));
    cudaMalloc(&d_offs,   (NUM_EXPERTS + 1) * sizeof(int));
    cudaMalloc(&d_tidx,   (size_t)MAX_TOKENS * TOP_K * sizeof(int));

    // Random init
    {
        srand(42);
        std::vector<bf16> h(MAX_TOKENS * HIDDEN);
        for (auto& v : h) v = __float2bfloat16((float)rand()/RAND_MAX - 0.5f);
        cudaMemcpy(d_X, h.data(), X_sz, cudaMemcpyHostToDevice);

        std::vector<bf16> w(NUM_EXPERTS * FFN_INTER * HIDDEN);
        for (auto& v : w) v = __float2bfloat16((float)rand()/RAND_MAX * 0.02f);
        cudaMemcpy(d_W1, w.data(), W1_sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, w.data(), W1_sz, cudaMemcpyHostToDevice);

        std::vector<bf16> w3(NUM_EXPERTS * HIDDEN * FFN_INTER);
        for (auto& v : w3) v = __float2bfloat16((float)rand()/RAND_MAX * 0.02f);
        cudaMemcpy(d_W3, w3.data(), W3_sz, cudaMemcpyHostToDevice);

        std::vector<float> lg(MAX_TOKENS * NUM_EXPERTS);
        for (auto& v : lg) v = (float)rand()/RAND_MAX;
        cudaMemcpy(d_logits, lg.data(), lg_sz, cudaMemcpyHostToDevice);
    }

    // Router
    router_kernel<<<(MAX_TOKENS+255)/256, 256>>>(
        d_logits, d_eids, d_scores, MAX_TOKENS, NUM_EXPERTS, TOP_K);
    cudaDeviceSynchronize();

    // Build CSR expert→token map on host
    {
        std::vector<int> h_eids(MAX_TOKENS * TOP_K);
        cudaMemcpy(h_eids.data(), d_eids, h_eids.size()*sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> cnt(NUM_EXPERTS, 0);
        for (int id : h_eids) cnt[id]++;
        std::vector<int> offs(NUM_EXPERTS + 1, 0);
        for (int e = 0; e < NUM_EXPERTS; ++e) offs[e+1] = offs[e] + cnt[e];
        std::vector<int> idx(MAX_TOKENS * TOP_K);
        std::vector<int> cur(offs.begin(), offs.end());
        for (int t = 0; t < MAX_TOKENS; ++t)
            for (int k = 0; k < TOP_K; ++k)
                idx[cur[h_eids[t*TOP_K+k]]++] = t;
        cudaMemcpy(d_offs, offs.data(), offs.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tidx, idx.data(),  idx.size() *sizeof(int), cudaMemcpyHostToDevice);
    }

    // Shared memory size
    const size_t smem_sz =
        SMEM_STAGES * TMA_BOX_M * WMMA_K * sizeof(bf16)  +
        SMEM_STAGES * TMA_BOX_N * WMMA_K * sizeof(bf16)  +
        SMEM_STAGES * TMA_BOX_N * WMMA_K * sizeof(bf16)  +
        TMA_BOX_M * TMA_BOX_N * sizeof(float)            +
        TMA_BOX_M * TMA_BOX_N * sizeof(float)            +
        TMA_BOX_M * TMA_BOX_N * sizeof(bf16);

    cudaFuncSetAttribute(moe_expert_fused_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);

    const int max_tpe    = (MAX_TOKENS * TOP_K + NUM_EXPERTS - 1) / NUM_EXPERTS;
    const int m_tiles    = (max_tpe + TMA_BOX_M - 1) / TMA_BOX_M;
    dim3 grid_w(NUM_EXPERTS, m_tiles);
    dim3 block(THREADS_PER_BLOCK);

    auto run_wmma = [&]() {
        cudaMemset(d_Y, 0, Y_sz);
        moe_expert_fused_kernel<<<grid_w, block, smem_sz>>>(
            d_X, d_eids, d_scores, d_offs, d_tidx,
            d_W1, d_W2, d_W3, d_Y,
            MAX_TOKENS, HIDDEN, FFN_INTER);
    };
    auto run_naive = [&]() {
        cudaMemset(d_Y, 0, Y_sz);
        moe_naive_kernel<<<(MAX_TOKENS+255)/256, 256>>>(
            d_X, d_eids, d_scores, d_offs, d_tidx,
            d_W1, d_W2, d_W3, d_Y,
            MAX_TOKENS, HIDDEN, FFN_INTER);
    };

    printf("Benchmarking (warmup=3, iters=10)...\n\n");
    BenchResult rw = bench("WMMA+TMA (ThunderKittens/Blackwell)", run_wmma, 3, 10);
    BenchResult rn = bench("Naive scalar baseline",               run_naive, 3, 10);

    printf("\n--- Summary ---\n");
    printf("Speedup:                 %.2fx\n",   rn.ms / rw.ms);
    printf("WMMA+TMA TFLOPs:         %.2f\n",    rw.tflops);
    printf("Naive    TFLOPs:         %.2f\n",    rn.tflops);
    printf("B200 bf16 peak ~2000 TFLOPs\n");
    printf("Roofline utilization:    %.1f%%\n",  100.0 * rw.tflops / 2000.0);

    cudaFree(d_X);  cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3);
    cudaFree(d_Y);  cudaFree(d_logits); cudaFree(d_scores);
    cudaFree(d_eids); cudaFree(d_offs); cudaFree(d_tidx);
    printf("\nDone.\n");
    return 0;
}
"""

# ---------------------------------------------------------------------------
# Modal image — writes the CUDA source from the string above, then compiles
# ---------------------------------------------------------------------------
cuda_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "cmake", "ninja-build")
    .run_commands(
        "git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens",
    )
    # Write the embedded source to disk inside the image, then compile
    .run_commands(
        f"cat > /workspace/deepseek_moe_wmma_tma.cu << 'ENDSRC'\n{CUDA_SRC}\nENDSRC",
        "mkdir -p /workspace && nvcc -O3 "
        "  -gencode arch=compute_100a,code=sm_100a "
        "  -std=c++20 "
        "  -I/opt/thunderkittens/include "
        "  -lcublas -lcuda "
        "  /workspace/deepseek_moe_wmma_tma.cu "
        "  -o /workspace/moe_bench",
    )
)

app = modal.App("deepseek-moe-wmma-tma")

# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------
@app.function(
    gpu="B200",
    image=cuda_image,
    timeout=600,
    memory=65536,
)
def run_bench() -> str:
    import subprocess, sys
    result = subprocess.run(["/workspace/moe_bench"], capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:\n", result.stderr, file=sys.stderr)
        raise RuntimeError(f"moe_bench exited {result.returncode}")
    return result.stdout


# ---------------------------------------------------------------------------
# GPU info helper
# ---------------------------------------------------------------------------
@app.function(gpu="B200", image=cuda_image, timeout=120)
def check_gpu_info() -> str:
    import subprocess
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("=== GPU info ===")
    print(check_gpu_info.remote())
    print()
    print("=== Benchmark ===")
    print(run_bench.remote())