#include "moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return p;
}

static void read_bytes(const char *path, void *dst, size_t nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(1);
    }
    size_t got = fread(dst, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) {
        fprintf(stderr, "failed reading %s\n", path);
        exit(1);
    }
}

static float *read_f32(const char *path, size_t count) {
    float *buf = (float *)xmalloc(count * sizeof(float));
    read_bytes(path, buf, count * sizeof(float));
    return buf;
}

static int *read_i32(const char *path, size_t count) {
    int *buf = (int *)xmalloc(count * sizeof(int));
    read_bytes(path, buf, count * sizeof(int));
    return buf;
}

static int max_abs_diff_f32(const float *a, const float *b, size_t n, float *out_max) {
    float mx = 0.0f;
    size_t arg = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) {
            mx = d;
            arg = i;
        }
    }
    *out_max = mx;
    return (int)arg;
}

static int compare_i32(const int *a, const int *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const char *base = "testdata/case_001";

    MoEConfig cfg = {
        .hidden_size = 8,
        .moe_intermediate_size = 6,
        .n_routed_experts = 8,
        .n_shared_experts = 1,
        .n_group = 4,
        .topk_group = 2,
        .num_experts_per_tok = 2,
        .norm_topk_prob = 1,
        .routed_scaling_factor = 2.5f,
    };

    const int batch_size = 2;
    const int seq_len = 3;
    const int tokens = batch_size * seq_len;
    const int H = cfg.hidden_size;
    const int E = cfg.n_routed_experts;
    const int M = cfg.moe_intermediate_size;
    const int topk = cfg.num_experts_per_tok;
    const int shared_mid = cfg.moe_intermediate_size * cfg.n_shared_experts;

    char path[512];

    MoEWeights wt;
    memset(&wt, 0, sizeof(wt));

    snprintf(path, sizeof(path), "%s/gate_weight.bin", base);
    wt.gate_weight = read_f32(path, (size_t)E * H);

    snprintf(path, sizeof(path), "%s/gate_bias_correction.bin", base);
    wt.gate_bias_correction = read_f32(path, (size_t)E);

    wt.expert_gate_proj = (float **)xmalloc((size_t)E * sizeof(float *));
    wt.expert_up_proj = (float **)xmalloc((size_t)E * sizeof(float *));
    wt.expert_down_proj = (float **)xmalloc((size_t)E * sizeof(float *));

    for (int e = 0; e < E; ++e) {
        snprintf(path, sizeof(path), "%s/expert_%d_gate_proj.bin", base, e);
        wt.expert_gate_proj[e] = read_f32(path, (size_t)M * H);

        snprintf(path, sizeof(path), "%s/expert_%d_up_proj.bin", base, e);
        wt.expert_up_proj[e] = read_f32(path, (size_t)M * H);

        snprintf(path, sizeof(path), "%s/expert_%d_down_proj.bin", base, e);
        wt.expert_down_proj[e] = read_f32(path, (size_t)H * M);
    }

    snprintf(path, sizeof(path), "%s/shared_gate_proj.bin", base);
    wt.shared_gate_proj = read_f32(path, (size_t)shared_mid * H);

    snprintf(path, sizeof(path), "%s/shared_up_proj.bin", base);
    wt.shared_up_proj = read_f32(path, (size_t)shared_mid * H);

    snprintf(path, sizeof(path), "%s/shared_down_proj.bin", base);
    wt.shared_down_proj = read_f32(path, (size_t)H * shared_mid);

    snprintf(path, sizeof(path), "%s/input.bin", base);
    float *x3d = read_f32(path, (size_t)batch_size * seq_len * H);
    float *x = x3d;

    snprintf(path, sizeof(path), "%s/logits.bin", base);
    float *ref_logits = read_f32(path, (size_t)tokens * E);

    snprintf(path, sizeof(path), "%s/scores.bin", base);
    float *ref_scores = read_f32(path, (size_t)tokens * E);

    snprintf(path, sizeof(path), "%s/topk_idx.bin", base);
    int *ref_topk_idx = read_i32(path, (size_t)tokens * topk);

    snprintf(path, sizeof(path), "%s/topk_weight.bin", base);
    float *ref_topk_weight = read_f32(path, (size_t)tokens * topk);

    snprintf(path, sizeof(path), "%s/final_out.bin", base);
    float *ref_final = read_f32(path, (size_t)batch_size * seq_len * H);

    float *logits = (float *)xmalloc((size_t)tokens * E * sizeof(float));
    float *scores = (float *)xmalloc((size_t)tokens * E * sizeof(float));
    int *topk_idx = (int *)xmalloc((size_t)tokens * topk * sizeof(int));
    float *topk_weight = (float *)xmalloc((size_t)tokens * topk * sizeof(float));
    float *y = (float *)xmalloc((size_t)tokens * H * sizeof(float));

    moe_gate_forward(&cfg, &wt, x, tokens, logits, scores, topk_idx, topk_weight);
    moe_forward(&cfg, &wt, x, tokens, y);

    float mx;
    int arg;

    arg = max_abs_diff_f32(logits, ref_logits, (size_t)tokens * E, &mx);
    printf("logits max_abs_diff = %.8f at %d\n", mx, arg);

    arg = max_abs_diff_f32(scores, ref_scores, (size_t)tokens * E, &mx);
    printf("scores max_abs_diff = %.8f at %d\n", mx, arg);

    printf("topk_idx exact_match = %s\n",
           compare_i32(topk_idx, ref_topk_idx, (size_t)tokens * topk) ? "YES" : "NO");

    arg = max_abs_diff_f32(topk_weight, ref_topk_weight, (size_t)tokens * topk, &mx);
    printf("topk_weight max_abs_diff = %.8f at %d\n", mx, arg);

    arg = max_abs_diff_f32(y, ref_final, (size_t)tokens * H, &mx);
    printf("final_out max_abs_diff = %.8f at %d\n", mx, arg);

    const float tol = 1e-4f;
    int ok = 1;

    if (!compare_i32(topk_idx, ref_topk_idx, (size_t)tokens * topk)) {
        ok = 0;
    }

    max_abs_diff_f32(logits, ref_logits, (size_t)tokens * E, &mx);
    if (mx > tol) ok = 0;

    max_abs_diff_f32(scores, ref_scores, (size_t)tokens * E, &mx);
    if (mx > tol) ok = 0;

    max_abs_diff_f32(topk_weight, ref_topk_weight, (size_t)tokens * topk, &mx);
    if (mx > tol) ok = 0;

    max_abs_diff_f32(y, ref_final, (size_t)tokens * H, &mx);
    if (mx > tol) ok = 0;

    printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");

    free(wt.gate_weight);
    free(wt.gate_bias_correction);

    for (int e = 0; e < E; ++e) {
        free(wt.expert_gate_proj[e]);
        free(wt.expert_up_proj[e]);
        free(wt.expert_down_proj[e]);
    }

    free(wt.expert_gate_proj);
    free(wt.expert_up_proj);
    free(wt.expert_down_proj);

    free(wt.shared_gate_proj);
    free(wt.shared_up_proj);
    free(wt.shared_down_proj);

    free(x3d);
    free(ref_logits);
    free(ref_scores);
    free(ref_topk_idx);
    free(ref_topk_weight);
    free(ref_final);

    free(logits);
    free(scores);
    free(topk_idx);
    free(topk_weight);
    free(y);

    return ok ? 0 : 1;
}