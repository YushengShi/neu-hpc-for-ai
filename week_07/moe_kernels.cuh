#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Elementwise SiLU in-place
// ---------------------------------------------------------------------------
__global__ void kernel_silu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ---------------------------------------------------------------------------
// Row-major matvec:  y[r] = sum_c W[r,c] * x[c]
// ---------------------------------------------------------------------------
__global__ void kernel_matvec(
    const float* __restrict__ W,   // [rows, cols]
    const float* __restrict__ x,   // [cols]
    float*       y,                // [rows]
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    double acc = 0.0;
    for (int c = 0; c < cols; c++)
        acc += (double)W[r * cols + c] * (double)x[c];
    y[r] = (float)acc;
}

// ---------------------------------------------------------------------------
// Elementwise multiply:  c[i] = a[i] * b[i]
// ---------------------------------------------------------------------------
__global__ void kernel_elemwise_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* c, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

// ---------------------------------------------------------------------------
// Weighted accumulate:  out[h] += weight * src[h]
// ---------------------------------------------------------------------------
__global__ void kernel_weighted_add(
    float*       out,
    const float* src,
    float        weight,
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < H) out[h] += weight * src[h];
}

// ---------------------------------------------------------------------------
// Elementwise add:  c[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
__global__ void kernel_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* c, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Gate scores:  scores[t,e] = sigmoid( gate_weight[e] dot x[t] )
// Launch: grid(T, ceil(E/BLOCK)), block(BLOCK)
// ---------------------------------------------------------------------------
__global__ void kernel_gate_scores(
    const float* __restrict__ x,           // [T, H]
    const float* __restrict__ gate_weight, // [E, H]
    float*       scores,                   // [T, E]
    int T, int H, int E
) {
    int t = blockIdx.x;
    int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || e >= E) return;

    const float* xt = x + (long)t * H;
    const float* we = gate_weight + (long)e * H;
    double acc = 0.0;
    for (int h = 0; h < H; h++)
        acc += (double)xt[h] * (double)we[h];
    scores[(long)t * E + e] = 1.0f / (1.0f + expf(-(float)acc));
}

// ---------------------------------------------------------------------------
// Top-K + weight normalization.  One thread per token.
// ---------------------------------------------------------------------------
#define MAX_EXPERTS_TOPK 128
#define MAX_TOPK_K        32

__global__ void kernel_topk(
    const float* __restrict__ scores,  // [T, E]
    int*   topk_idx,                   // [T, K]
    float* topk_weight,                // [T, K]
    int T, int E, int K,
    float routed_scaling_factor
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const float* s = scores + (long)t * E;

    // selection-sort top-K  (K ≤ 32, E ≤ 128 → fast enough)
    int   order[MAX_TOPK_K];
    float best [MAX_TOPK_K];
    bool  used [MAX_EXPERTS_TOPK];
    for (int e = 0; e < E; e++) used[e] = false;

    for (int k = 0; k < K; k++) {
        float bv = -1.f;
        int   bi = -1;
        for (int e = 0; e < E; e++) {
            if (!used[e] && (s[e] > bv || (s[e] == bv && (bi < 0 || e < bi)))) {
                bv = s[e]; bi = e;
            }
        }
        order[k] = bi;
        best[k]  = bv;
        used[bi] = true;
    }

    double denom = 0.0;
    for (int k = 0; k < K; k++) denom += best[k];
    if (denom < 1e-20) denom = 1e-20;

    for (int k = 0; k < K; k++) {
        topk_idx   [(long)t * K + k] = order[k];
        topk_weight[(long)t * K + k] = (float)((best[k] / denom) * routed_scaling_factor);
    }
}
