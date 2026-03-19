#include "moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float sigmoidf_approx(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

float siluf(float x) {
    return x * sigmoidf_approx(x);
}

void linear_rowmajor(
    const float *x,
    const float *w,
    float *y,
    int out_dim,
    int in_dim
) {
    for (int o = 0; o < out_dim; ++o) {
        float sum = 0.0f;
        const float *row = w + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            sum += row[i] * x[i];
        }
        y[o] = sum;
    }
}

void canonicalize_topk_pairs(int *idx, float *weight, int k) {
    for (int i = 0; i < k - 1; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (idx[j] < idx[i]) {
                int ti = idx[i];
                idx[i] = idx[j];
                idx[j] = ti;

                float tw = weight[i];
                weight[i] = weight[j];
                weight[j] = tw;
            }
        }
    }
}

static void topk_indices_desc(const float *arr, int n, int k, int *out_idx) {
    float *best_vals = (float *)malloc((size_t)k * sizeof(float));
    int *best_idx = (int *)malloc((size_t)k * sizeof(int));
    if (!best_vals || !best_idx) {
        fprintf(stderr, "malloc failed in topk_indices_desc\n");
        exit(1);
    }

    for (int i = 0; i < k; ++i) {
        best_vals[i] = -INFINITY;
        best_idx[i] = -1;
    }

    for (int i = 0; i < n; ++i) {
        float v = arr[i];
        int pos = -1;
        for (int j = 0; j < k; ++j) {
            if (v > best_vals[j]) {
                pos = j;
                break;
            }
        }
        if (pos >= 0) {
            for (int j = k - 1; j > pos; --j) {
                best_vals[j] = best_vals[j - 1];
                best_idx[j] = best_idx[j - 1];
            }
            best_vals[pos] = v;
            best_idx[pos] = i;
        }
    }

    for (int i = 0; i < k; ++i) {
        out_idx[i] = best_idx[i];
    }

    free(best_vals);
    free(best_idx);
}

void moe_gate_forward(
    const MoEConfig *cfg,
    const MoEWeights *wt,
    const float *x,
    int tokens,
    float *logits_out,
    float *scores_out,
    int *topk_idx_out,
    float *topk_weight_out
) {
    const int H = cfg->hidden_size;
    const int E = cfg->n_routed_experts;
    const int topk = cfg->num_experts_per_tok;

    float *scores_for_choice = (float *)malloc((size_t)E * sizeof(float));
    if (!scores_for_choice) {
        fprintf(stderr, "malloc failed in moe_gate_forward\n");
        exit(1);
    }

    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * H;
        float *logits_t = logits_out + (size_t)t * E;
        float *scores_t = scores_out + (size_t)t * E;
        int *topk_idx_t = topk_idx_out + (size_t)t * topk;
        float *topk_w_t = topk_weight_out + (size_t)t * topk;

        linear_rowmajor(xt, wt->gate_weight, logits_t, E, H);

        for (int e = 0; e < E; ++e) {
            scores_t[e] = sigmoidf_approx(logits_t[e]);
            scores_for_choice[e] = scores_t[e] + wt->gate_bias_correction[e];
        }

        // Match export_testcases.py exactly: global top-k on corrected scores
        topk_indices_desc(scores_for_choice, E, topk, topk_idx_t);

        for (int k = 0; k < topk; ++k) {
            topk_w_t[k] = scores_t[topk_idx_t[k]];
        }

        float denom = 1e-9f;
        for (int k = 0; k < topk; ++k) {
            denom += topk_w_t[k];
        }
        for (int k = 0; k < topk; ++k) {
            topk_w_t[k] = (topk_w_t[k] / denom) * cfg->routed_scaling_factor;
        }
    }

    free(scores_for_choice);
}

void expert_mlp_forward(
    const float *x,
    const float *gate_proj,
    const float *up_proj,
    const float *down_proj,
    int hidden,
    int mid,
    float *y
) {
    float *gate = (float *)malloc((size_t)mid * sizeof(float));
    float *up = (float *)malloc((size_t)mid * sizeof(float));
    float *tmp = (float *)malloc((size_t)mid * sizeof(float));

    if (!gate || !up || !tmp) {
        fprintf(stderr, "malloc failed in expert_mlp_forward\n");
        exit(1);
    }

    linear_rowmajor(x, gate_proj, gate, mid, hidden);
    linear_rowmajor(x, up_proj, up, mid, hidden);

    for (int i = 0; i < mid; ++i) {
        tmp[i] = siluf(gate[i]) * up[i];
    }

    linear_rowmajor(tmp, down_proj, y, hidden, mid);

    free(gate);
    free(up);
    free(tmp);
}

void moe_forward(
    const MoEConfig *cfg,
    const MoEWeights *wt,
    const float *x,
    int tokens,
    float *y
) {
    const int H = cfg->hidden_size;
    const int M = cfg->moe_intermediate_size;
    const int topk = cfg->num_experts_per_tok;
    const int E = cfg->n_routed_experts;
    const int shared_mid = cfg->moe_intermediate_size * cfg->n_shared_experts;

    float *logits = (float *)malloc((size_t)tokens * E * sizeof(float));
    float *scores = (float *)malloc((size_t)tokens * E * sizeof(float));
    int *topk_idx = (int *)malloc((size_t)tokens * topk * sizeof(int));
    float *topk_weight = (float *)malloc((size_t)tokens * topk * sizeof(float));
    float *tmp = (float *)malloc((size_t)H * sizeof(float));
    float *shared = (float *)malloc((size_t)H * sizeof(float));

    if (!logits || !scores || !topk_idx || !topk_weight || !tmp || !shared) {
        fprintf(stderr, "malloc failed in moe_forward\n");
        exit(1);
    }

    moe_gate_forward(cfg, wt, x, tokens, logits, scores, topk_idx, topk_weight);

    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * H;
        float *yt = y + (size_t)t * H;

        for (int h = 0; h < H; ++h) {
            yt[h] = 0.0f;
        }

        for (int k = 0; k < topk; ++k) {
            int eidx = topk_idx[(size_t)t * topk + k];
            float w = topk_weight[(size_t)t * topk + k];

            expert_mlp_forward(
                xt,
                wt->expert_gate_proj[eidx],
                wt->expert_up_proj[eidx],
                wt->expert_down_proj[eidx],
                H,
                M,
                tmp
            );

            for (int h = 0; h < H; ++h) {
                yt[h] += w * tmp[h];
            }
        }

        if (cfg->n_shared_experts > 0) {
            expert_mlp_forward(
                xt,
                wt->shared_gate_proj,
                wt->shared_up_proj,
                wt->shared_down_proj,
                H,
                shared_mid,
                shared
            );

            for (int h = 0; h < H; ++h) {
                yt[h] += shared[h];
            }
        }
    }

    free(logits);
    free(scores);
    free(topk_idx);
    free(topk_weight);
    free(tmp);
    free(shared);
}