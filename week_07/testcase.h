#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Test case struct  (all host memory)
// ---------------------------------------------------------------------------
typedef struct {
    int   num_tokens;
    int   hidden_size;
    int   moe_intermediate_size;
    int   n_routed_experts;
    int   n_shared_experts;
    int   top_k;
    float routed_scaling_factor;

    float* x;                   // [T, H]
    float* gate_weight;         // [E, H]
    float* routed_gate_proj;    // [E, I, H]
    float* routed_up_proj;      // [E, I, H]
    float* routed_down_proj;    // [E, H, I]
    float* shared_gate_proj;    // [S*I, H]
    float* shared_up_proj;      // [S*I, H]
    float* shared_down_proj;    // [H, S*I]

    int*   expected_topk_idx;    // [T, K]
    float* expected_topk_weight; // [T, K]
    float* expected_shared_out;  // [T, H]
    float* expected_routed_out;  // [T, H]
    float* expected_final_out;   // [T, H]
} TestCase;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline int tc_load_floats(FILE* f, float* a, long n) {
    for (long i = 0; i < n; i++)
        if (fscanf(f, "%f", &a[i]) != 1) return 0;
    return 1;
}
static inline int tc_load_ints(FILE* f, int* a, long n) {
    for (long i = 0; i < n; i++)
        if (fscanf(f, "%d", &a[i]) != 1) return 0;
    return 1;
}

static inline int tc_load(FILE* f, TestCase* tc) {
    if (fscanf(f, "%d %d %d %d %d %d %f",
               &tc->num_tokens, &tc->hidden_size, &tc->moe_intermediate_size,
               &tc->n_routed_experts, &tc->n_shared_experts, &tc->top_k,
               &tc->routed_scaling_factor) != 7)
        return 0;

    long T=tc->num_tokens, H=tc->hidden_size, I=tc->moe_intermediate_size,
         E=tc->n_routed_experts, S=tc->n_shared_experts, K=tc->top_k;

    tc->x                  = (float*)malloc(sizeof(float)*T*H);
    tc->gate_weight        = (float*)malloc(sizeof(float)*E*H);
    tc->routed_gate_proj   = (float*)malloc(sizeof(float)*E*I*H);
    tc->routed_up_proj     = (float*)malloc(sizeof(float)*E*I*H);
    tc->routed_down_proj   = (float*)malloc(sizeof(float)*E*H*I);
    tc->shared_gate_proj   = (float*)malloc(sizeof(float)*(S*I)*H);
    tc->shared_up_proj     = (float*)malloc(sizeof(float)*(S*I)*H);
    tc->shared_down_proj   = (float*)malloc(sizeof(float)*H*(S*I));
    tc->expected_topk_idx    = (int*)  malloc(sizeof(int)  *T*K);
    tc->expected_topk_weight = (float*)malloc(sizeof(float)*T*K);
    tc->expected_shared_out  = (float*)malloc(sizeof(float)*T*H);
    tc->expected_routed_out  = (float*)malloc(sizeof(float)*T*H);
    tc->expected_final_out   = (float*)malloc(sizeof(float)*T*H);

    if (!tc_load_floats(f,tc->x,T*H))            return 0;
    if (!tc_load_floats(f,tc->gate_weight,E*H))   return 0;
    if (!tc_load_floats(f,tc->routed_gate_proj,E*I*H)) return 0;
    if (!tc_load_floats(f,tc->routed_up_proj,E*I*H))   return 0;
    if (!tc_load_floats(f,tc->routed_down_proj,E*H*I))  return 0;
    if (!tc_load_floats(f,tc->shared_gate_proj,(S*I)*H)) return 0;
    if (!tc_load_floats(f,tc->shared_up_proj,(S*I)*H))   return 0;
    if (!tc_load_floats(f,tc->shared_down_proj,H*(S*I)))  return 0;
    if (!tc_load_ints  (f,tc->expected_topk_idx,T*K))    return 0;
    if (!tc_load_floats(f,tc->expected_topk_weight,T*K))  return 0;
    if (!tc_load_floats(f,tc->expected_shared_out,T*H))   return 0;
    if (!tc_load_floats(f,tc->expected_routed_out,T*H))   return 0;
    if (!tc_load_floats(f,tc->expected_final_out,T*H))    return 0;
    return 1;
}

static inline void tc_free(TestCase* tc) {
    free(tc->x); free(tc->gate_weight);
    free(tc->routed_gate_proj); free(tc->routed_up_proj); free(tc->routed_down_proj);
    free(tc->shared_gate_proj); free(tc->shared_up_proj); free(tc->shared_down_proj);
    free(tc->expected_topk_idx); free(tc->expected_topk_weight);
    free(tc->expected_shared_out); free(tc->expected_routed_out); free(tc->expected_final_out);
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------
static inline float tc_max_abs_diff(const float* a, const float* b, long n) {
    float m = 0.f;
    for (long i = 0; i < n; i++) {
        float d = fabsf(a[i]-b[i]);
        if (d > m) m = d;
    }
    return m;
}
static inline int tc_exact_int_match(const int* a, const int* b, long n) {
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}