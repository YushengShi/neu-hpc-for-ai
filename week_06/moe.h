#ifndef MOE_H
#define MOE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int hidden_size;
    int moe_intermediate_size;
    int n_routed_experts;
    int n_shared_experts;
    int n_group;
    int topk_group;
    int num_experts_per_tok;
    int norm_topk_prob;
    float routed_scaling_factor;
} MoEConfig;

typedef struct {
    float *gate_weight;          // [n_routed_experts, hidden_size]
    float *gate_bias_correction; // [n_routed_experts]

    float **expert_gate_proj;    // [n_routed_experts][moe_intermediate_size, hidden_size]
    float **expert_up_proj;      // [n_routed_experts][moe_intermediate_size, hidden_size]
    float **expert_down_proj;    // [n_routed_experts][hidden_size, moe_intermediate_size]

    float *shared_gate_proj;     // [shared_intermediate, hidden_size]
    float *shared_up_proj;       // [shared_intermediate, hidden_size]
    float *shared_down_proj;     // [hidden_size, shared_intermediate]
} MoEWeights;

float sigmoidf_approx(float x);
float siluf(float x);

void linear_rowmajor(
    const float *x,
    const float *w,
    float *y,
    int out_dim,
    int in_dim
);

void canonicalize_topk_pairs(int *idx, float *weight, int k);

void moe_gate_forward(
    const MoEConfig *cfg,
    const MoEWeights *wt,
    const float *x,
    int tokens,
    float *logits_out,
    float *scores_out,
    int *topk_idx_out,
    float *topk_weight_out
);

void expert_mlp_forward(
    const float *x,
    const float *gate_proj,
    const float *up_proj,
    const float *down_proj,
    int hidden,
    int mid,
    float *y
);

void moe_forward(
    const MoEConfig *cfg,
    const MoEWeights *wt,
    const float *x,
    int tokens,
    float *y
);

#ifdef __cplusplus
}
#endif

#endif