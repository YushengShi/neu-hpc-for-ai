// moe_single_gpu.cu
// Steps 1-2 of checklist: single-GPU MoE forward, verified against test cases.
// Compile:
//   nvcc -O2 -std=c++14 moe_single_gpu.cu -o moe_single_gpu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "moe_kernels.cuh"
#include "testcase.h"

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CK(x) do {                                                        \
    cudaError_t _e = (x);                                                 \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA %s:%d: %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        exit(1);                                                          \
    }                                                                     \
} while(0)

// ---------------------------------------------------------------------------
// Run gated MLP for ONE token through ONE expert.
//   gate_out = gate_proj @ x        [I]
//   up_out   = up_proj   @ x        [I]
//   mid      = silu(gate_out) * up  [I]
//   out      = down_proj @ mid      [H]
// Scratch buffers d_g/d_u/d_m must be size [I].
// ---------------------------------------------------------------------------
static void run_gated_mlp(
    const float* d_x,          // [H]  device
    int H, int I,
    const float* d_gate_proj,  // [I, H]
    const float* d_up_proj,    // [I, H]
    const float* d_down_proj,  // [H, I]
    float* d_out,              // [H]  device
    float* d_g,                // [I]  scratch
    float* d_u,                // [I]  scratch
    float* d_m                 // [I]  scratch
) {
    int tb = 128;
    kernel_matvec<<<(I+tb-1)/tb, tb>>>(d_gate_proj, d_x, d_g, I, H);
    kernel_matvec<<<(I+tb-1)/tb, tb>>>(d_up_proj,   d_x, d_u, I, H);
    kernel_silu  <<<(I+tb-1)/tb, tb>>>(d_g, I);
    kernel_elemwise_mul<<<(I+tb-1)/tb, tb>>>(d_g, d_u, d_m, I);
    kernel_matvec<<<(H+tb-1)/tb, tb>>>(d_down_proj, d_m, d_out, H, I);
}

// ---------------------------------------------------------------------------
// Full single-GPU MoE forward.
// Computes: gate -> topk_idx/weight, shared_out, routed_out, final_out
// ---------------------------------------------------------------------------
static void moe_forward(
    const float* d_x,              // [T, H]
    int T, int H, int SI, int I,   // SI = shared intermediate = n_shared * I
    int E, int K,
    float scale,
    const float* d_gate_w,         // [E, H]
    const float* d_rgp,            // [E, I, H]  routed gate proj
    const float* d_rup,            // [E, I, H]  routed up proj
    const float* d_rdp,            // [E, H, I]  routed down proj
    const float* d_sgp,            // [SI, H]    shared gate proj
    const float* d_sup,            // [SI, H]    shared up proj
    const float* d_sdp,            // [H, SI]    shared down proj
    // outputs
    float* d_shared_out,           // [T, H]
    float* d_routed_out,           // [T, H]
    int*   d_topk_idx,             // [T, K]
    float* d_topk_wt               // [T, K]
) {
    int tb = 128;

    // ------------------------------------------------------------------
    // Step 5: Compute gate scores and top-K routing
    // ------------------------------------------------------------------
    float* d_scores;
    CK(cudaMalloc(&d_scores, sizeof(float) * T * E));

    dim3 gs_grid(T, (E + tb - 1) / tb);
    kernel_gate_scores<<<gs_grid, tb>>>(d_x, d_gate_w, d_scores, T, H, E);

    kernel_topk<<<(T+tb-1)/tb, tb>>>(d_scores, d_topk_idx, d_topk_wt,
                                      T, E, K, scale);
    CK(cudaFree(d_scores));

    // ------------------------------------------------------------------
    // Shared experts: always-on, runs for every token
    // ------------------------------------------------------------------
    float *d_sg, *d_su, *d_sm;
    CK(cudaMalloc(&d_sg, sizeof(float)*SI));
    CK(cudaMalloc(&d_su, sizeof(float)*SI));
    CK(cudaMalloc(&d_sm, sizeof(float)*SI));

    for (int t = 0; t < T; t++) {
        run_gated_mlp(d_x + (long)t*H, H, SI,
                      d_sgp, d_sup, d_sdp,
                      d_shared_out + (long)t*H,
                      d_sg, d_su, d_sm);
    }
    CK(cudaFree(d_sg)); CK(cudaFree(d_su)); CK(cudaFree(d_sm));

    // ------------------------------------------------------------------
    // Steps 6-10: Routed experts
    //   - pull topk to host so we can loop (T*K is tiny in test cases)
    //   - for each (token, expert) pair run the expert and accumulate
    // ------------------------------------------------------------------
    CK(cudaMemset(d_routed_out, 0, sizeof(float) * T * H));

    int*   h_idx = (int*)  malloc(sizeof(int)  *T*K);
    float* h_wt  = (float*)malloc(sizeof(float)*T*K);
    CK(cudaMemcpy(h_idx, d_topk_idx, sizeof(int)  *T*K, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_wt,  d_topk_wt,  sizeof(float)*T*K, cudaMemcpyDeviceToHost));

    float *d_eg, *d_eu, *d_em, *d_eout;
    CK(cudaMalloc(&d_eg,   sizeof(float)*I));
    CK(cudaMalloc(&d_eu,   sizeof(float)*I));
    CK(cudaMalloc(&d_em,   sizeof(float)*I));
    CK(cudaMalloc(&d_eout, sizeof(float)*H));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            int   e = h_idx[(long)t*K + k];
            float w = h_wt [(long)t*K + k];

            const float* gp = d_rgp + (long)e*I*H;
            const float* up = d_rup + (long)e*I*H;
            const float* dp = d_rdp + (long)e*H*I;

            run_gated_mlp(d_x + (long)t*H, H, I,
                          gp, up, dp, d_eout,
                          d_eg, d_eu, d_em);

            kernel_weighted_add<<<(H+tb-1)/tb, tb>>>(
                d_routed_out + (long)t*H, d_eout, w, H);
        }
    }

    free(h_idx); free(h_wt);
    CK(cudaFree(d_eg)); CK(cudaFree(d_eu));
    CK(cudaFree(d_em)); CK(cudaFree(d_eout));
}

// ---------------------------------------------------------------------------
// main: load tests.txt, run forward, verify, print results
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tests.txt>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    int num_cases = 0;
    fscanf(f, "%d", &num_cases);

    const float atol = 2e-4f;
    int all_ok = 1;

    for (int cid = 0; cid < num_cases; cid++) {
        TestCase tc;
        memset(&tc, 0, sizeof(tc));
        if (!tc_load(f, &tc)) {
            fprintf(stderr, "Failed to load test case %d\n", cid);
            break;
        }

        int T=tc.num_tokens, H=tc.hidden_size, I=tc.moe_intermediate_size;
        int E=tc.n_routed_experts, S=tc.n_shared_experts, K=tc.top_k;
        int SI = S * I;

        // Upload all weights and input to GPU
        float *d_x,*d_gw,*d_rgp,*d_rup,*d_rdp,*d_sgp,*d_sup,*d_sdp;
        CK(cudaMalloc(&d_x,   sizeof(float)*T*H));
        CK(cudaMalloc(&d_gw,  sizeof(float)*E*H));
        CK(cudaMalloc(&d_rgp, sizeof(float)*E*I*H));
        CK(cudaMalloc(&d_rup, sizeof(float)*E*I*H));
        CK(cudaMalloc(&d_rdp, sizeof(float)*E*H*I));
        CK(cudaMalloc(&d_sgp, sizeof(float)*SI*H));
        CK(cudaMalloc(&d_sup, sizeof(float)*SI*H));
        CK(cudaMalloc(&d_sdp, sizeof(float)*H*SI));

        CK(cudaMemcpy(d_x,   tc.x,              sizeof(float)*T*H,   cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gw,  tc.gate_weight,    sizeof(float)*E*H,   cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rgp, tc.routed_gate_proj,sizeof(float)*E*I*H,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rup, tc.routed_up_proj,  sizeof(float)*E*I*H,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rdp, tc.routed_down_proj,sizeof(float)*E*H*I,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sgp, tc.shared_gate_proj,sizeof(float)*SI*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sup, tc.shared_up_proj,  sizeof(float)*SI*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sdp, tc.shared_down_proj,sizeof(float)*H*SI, cudaMemcpyHostToDevice));

        float *d_shared_out, *d_routed_out;
        int*   d_topk_idx;
        float* d_topk_wt;
        CK(cudaMalloc(&d_shared_out, sizeof(float)*T*H));
        CK(cudaMalloc(&d_routed_out, sizeof(float)*T*H));
        CK(cudaMalloc((void**)&d_topk_idx, sizeof(int)*T*K));
        CK(cudaMalloc(&d_topk_wt,    sizeof(float)*T*K));

        // Run forward
        moe_forward(d_x, T, H, SI, I, E, K, tc.routed_scaling_factor,
                    d_gw, d_rgp, d_rup, d_rdp, d_sgp, d_sup, d_sdp,
                    d_shared_out, d_routed_out, d_topk_idx, d_topk_wt);

        CK(cudaDeviceSynchronize());

        // Compute final = shared + routed on device
        float* d_final;
        CK(cudaMalloc(&d_final, sizeof(float)*T*H));
        int tb = 128;
        kernel_add<<<(T*H+tb-1)/tb, tb>>>(d_shared_out, d_routed_out, d_final, T*H);
        CK(cudaDeviceSynchronize());

        // Copy back to host
        float* got_shared = (float*)malloc(sizeof(float)*T*H);
        float* got_routed = (float*)malloc(sizeof(float)*T*H);
        float* got_final  = (float*)malloc(sizeof(float)*T*H);
        int*   got_idx    = (int*)  malloc(sizeof(int)  *T*K);
        float* got_wt     = (float*)malloc(sizeof(float)*T*K);

        CK(cudaMemcpy(got_shared, d_shared_out, sizeof(float)*T*H, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_routed, d_routed_out, sizeof(float)*T*H, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_final,  d_final,      sizeof(float)*T*H, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_idx,    d_topk_idx,   sizeof(int)  *T*K, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_wt,     d_topk_wt,    sizeof(float)*T*K, cudaMemcpyDeviceToHost));

        // Verify
        int   idx_ok = tc_exact_int_match(got_idx,    tc.expected_topk_idx,    (long)T*K);
        float w_err  = tc_max_abs_diff   (got_wt,     tc.expected_topk_weight, (long)T*K);
        float s_err  = tc_max_abs_diff   (got_shared, tc.expected_shared_out,  (long)T*H);
        float r_err  = tc_max_abs_diff   (got_routed, tc.expected_routed_out,  (long)T*H);
        float f_err  = tc_max_abs_diff   (got_final,  tc.expected_final_out,   (long)T*H);

        int ok = idx_ok && (w_err<=atol) && (s_err<=atol) && (r_err<=atol) && (f_err<=atol);
        if (!ok) all_ok = 0;

        printf("case=%d idx_ok=%d w_err=%.6f s_err=%.6f r_err=%.6f f_err=%.6f [%s]\n",
               cid, idx_ok, w_err, s_err, r_err, f_err, ok ? "PASS" : "FAIL");

        // Free
        CK(cudaFree(d_x)); CK(cudaFree(d_gw));
        CK(cudaFree(d_rgp)); CK(cudaFree(d_rup)); CK(cudaFree(d_rdp));
        CK(cudaFree(d_sgp)); CK(cudaFree(d_sup)); CK(cudaFree(d_sdp));
        CK(cudaFree(d_shared_out)); CK(cudaFree(d_routed_out));
        CK(cudaFree(d_topk_idx));   CK(cudaFree(d_topk_wt));
        CK(cudaFree(d_final));
        free(got_shared); free(got_routed); free(got_final); free(got_idx); free(got_wt);
        tc_free(&tc);
    }

    fclose(f);
    printf("\nOverall: %s\n", all_ok ? "ALL PASS" : "SOME FAILURES");
    return all_ok ? 0 : 2;
}
