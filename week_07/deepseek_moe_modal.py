import json
import math
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("deepseek-moe-c-reference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "git")
    .pip_install(
        "numpy>=1.26.0",
        "torch>=2.2.0",
        "git+https://github.com/huggingface/transformers.git",
    )
)

C_SOURCE = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TESTS 64
#define MAX_TOKENS 512
#define MAX_HIDDEN 512
#define MAX_INTER 512
#define MAX_EXPERTS 128
#define MAX_TOPK 32
#define EPS 1e-12f

typedef struct {
    int num_tokens;
    int hidden_size;
    int moe_intermediate_size;
    int n_routed_experts;
    int n_shared_experts;
    int top_k;
    float routed_scaling_factor;
    float* x;                            // [num_tokens, hidden_size]
    float* gate_weight;                  // [n_routed_experts, hidden_size]
    float* routed_gate_proj;             // [n_routed_experts, inter, hidden]
    float* routed_up_proj;               // [n_routed_experts, inter, hidden]
    float* routed_down_proj;             // [n_routed_experts, hidden, inter]
    float* shared_gate_proj;             // [n_shared_experts*inter, hidden]
    float* shared_up_proj;               // [n_shared_experts*inter, hidden]
    float* shared_down_proj;             // [hidden, n_shared_experts*inter]

    int* expected_topk_idx;              // [num_tokens, top_k]
    float* expected_topk_weight;         // [num_tokens, top_k]
    float* expected_shared_out;          // [num_tokens, hidden_size]
    float* expected_routed_out;          // [num_tokens, hidden_size]
    float* expected_final_out;           // [num_tokens, hidden_size]
} TestCase;

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int exact_int_match(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void matvec_rowmajor(
    const float* W,   // [rows, cols]
    const float* x,   // [cols]
    float* y,         // [rows]
    int rows,
    int cols
) {
    for (int r = 0; r < rows; r++) {
        double acc = 0.0;
        for (int c = 0; c < cols; c++) {
            acc += (double)W[r * cols + c] * (double)x[c];
        }
        y[r] = (float)acc;
    }
}

static void gated_mlp_single_expert(
    const float* x,                 // [hidden]
    int hidden_size,
    int intermediate_size,
    const float* gate_proj,         // [inter, hidden]
    const float* up_proj,           // [inter, hidden]
    const float* down_proj,         // [hidden, inter]
    float* out                      // [hidden]
) {
    float* gate = (float*)calloc(intermediate_size, sizeof(float));
    float* up   = (float*)calloc(intermediate_size, sizeof(float));
    float* mid  = (float*)calloc(intermediate_size, sizeof(float));

    matvec_rowmajor(gate_proj, x, gate, intermediate_size, hidden_size);
    matvec_rowmajor(up_proj,   x, up,   intermediate_size, hidden_size);

    for (int i = 0; i < intermediate_size; i++) {
        mid[i] = silu(gate[i]) * up[i];
    }

    // down_proj: [hidden, inter]
    for (int h = 0; h < hidden_size; h++) {
        double acc = 0.0;
        for (int i = 0; i < intermediate_size; i++) {
            acc += (double)down_proj[h * intermediate_size + i] * (double)mid[i];
        }
        out[h] = (float)acc;
    }

    free(gate);
    free(up);
    free(mid);
}

static void compute_gate(
    const TestCase* tc,
    int* topk_idx,            // [num_tokens, top_k]
    float* topk_weight        // [num_tokens, top_k]
) {
    int T = tc->num_tokens;
    int H = tc->hidden_size;
    int E = tc->n_routed_experts;
    int K = tc->top_k;

    float* scores = (float*)calloc(E, sizeof(float));
    int* order = (int*)calloc(E, sizeof(int));

    for (int t = 0; t < T; t++) {
        const float* xt = tc->x + t * H;

        for (int e = 0; e < E; e++) {
            double logit = 0.0;
            const float* w = tc->gate_weight + e * H;
            for (int h = 0; h < H; h++) {
                logit += (double)w[h] * (double)xt[h];
            }
            float s = 1.0f / (1.0f + expf((float)(-logit))); // sigmoid
            scores[e] = s;
            order[e] = e;
        }

        // top-k by score descending, tie-break smaller index
        for (int i = 0; i < K; i++) {
            int best = i;
            for (int j = i + 1; j < E; j++) {
                float sj = scores[order[j]];
                float sb = scores[order[best]];
                if (sj > sb || (sj == sb && order[j] < order[best])) {
                    best = j;
                }
            }
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }

        double denom = 0.0;
        for (int k = 0; k < K; k++) {
            denom += scores[order[k]];
        }
        if (denom < 1e-20) denom = 1e-20;

        for (int k = 0; k < K; k++) {
            topk_idx[t * K + k] = order[k];
            topk_weight[t * K + k] = (float)((scores[order[k]] / denom) * tc->routed_scaling_factor);
        }
    }

    free(scores);
    free(order);
}

static void compute_shared(
    const TestCase* tc,
    float* shared_out          // [num_tokens, hidden]
) {
    int T = tc->num_tokens;
    int H = tc->hidden_size;
    int I = tc->moe_intermediate_size * tc->n_shared_experts;

    for (int t = 0; t < T; t++) {
        gated_mlp_single_expert(
            tc->x + t * H,
            H,
            I,
            tc->shared_gate_proj,
            tc->shared_up_proj,
            tc->shared_down_proj,
            shared_out + t * H
        );
    }
}

static void compute_routed(
    const TestCase* tc,
    const int* topk_idx,           // [num_tokens, top_k]
    const float* topk_weight,      // [num_tokens, top_k]
    float* routed_out              // [num_tokens, hidden]
) {
    int T = tc->num_tokens;
    int H = tc->hidden_size;
    int I = tc->moe_intermediate_size;
    int K = tc->top_k;

    memset(routed_out, 0, sizeof(float) * T * H);

    float* tmp = (float*)calloc(H, sizeof(float));

    for (int t = 0; t < T; t++) {
        const float* xt = tc->x + t * H;
        float* yt = routed_out + t * H;

        for (int k = 0; k < K; k++) {
            int e = topk_idx[t * K + k];
            float w = topk_weight[t * K + k];

            const float* gate_proj = tc->routed_gate_proj + e * I * H;
            const float* up_proj   = tc->routed_up_proj   + e * I * H;
            const float* down_proj = tc->routed_down_proj + e * H * I;

            gated_mlp_single_expert(xt, H, I, gate_proj, up_proj, down_proj, tmp);

            for (int h = 0; h < H; h++) {
                yt[h] += w * tmp[h];
            }
        }
    }

    free(tmp);
}

static int load_floats(FILE* f, float* arr, int n) {
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &arr[i]) != 1) return 0;
    }
    return 1;
}

static int load_ints(FILE* f, int* arr, int n) {
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%d", &arr[i]) != 1) return 0;
    }
    return 1;
}

static void free_testcase(TestCase* tc) {
    free(tc->x);
    free(tc->gate_weight);
    free(tc->routed_gate_proj);
    free(tc->routed_up_proj);
    free(tc->routed_down_proj);
    free(tc->shared_gate_proj);
    free(tc->shared_up_proj);
    free(tc->shared_down_proj);
    free(tc->expected_topk_idx);
    free(tc->expected_topk_weight);
    free(tc->expected_shared_out);
    free(tc->expected_routed_out);
    free(tc->expected_final_out);
}

static int load_testcase(FILE* f, TestCase* tc) {
    if (fscanf(
        f,
        "%d %d %d %d %d %d %f",
        &tc->num_tokens,
        &tc->hidden_size,
        &tc->moe_intermediate_size,
        &tc->n_routed_experts,
        &tc->n_shared_experts,
        &tc->top_k,
        &tc->routed_scaling_factor
    ) != 7) {
        return 0;
    }

    int T = tc->num_tokens;
    int H = tc->hidden_size;
    int I = tc->moe_intermediate_size;
    int E = tc->n_routed_experts;
    int S = tc->n_shared_experts;
    int K = tc->top_k;

    tc->x = (float*)malloc(sizeof(float) * T * H);
    tc->gate_weight = (float*)malloc(sizeof(float) * E * H);
    tc->routed_gate_proj = (float*)malloc(sizeof(float) * E * I * H);
    tc->routed_up_proj = (float*)malloc(sizeof(float) * E * I * H);
    tc->routed_down_proj = (float*)malloc(sizeof(float) * E * H * I);
    tc->shared_gate_proj = (float*)malloc(sizeof(float) * (S * I) * H);
    tc->shared_up_proj = (float*)malloc(sizeof(float) * (S * I) * H);
    tc->shared_down_proj = (float*)malloc(sizeof(float) * H * (S * I));

    tc->expected_topk_idx = (int*)malloc(sizeof(int) * T * K);
    tc->expected_topk_weight = (float*)malloc(sizeof(float) * T * K);
    tc->expected_shared_out = (float*)malloc(sizeof(float) * T * H);
    tc->expected_routed_out = (float*)malloc(sizeof(float) * T * H);
    tc->expected_final_out = (float*)malloc(sizeof(float) * T * H);

    if (!load_floats(f, tc->x, T * H)) return 0;
    if (!load_floats(f, tc->gate_weight, E * H)) return 0;
    if (!load_floats(f, tc->routed_gate_proj, E * I * H)) return 0;
    if (!load_floats(f, tc->routed_up_proj, E * I * H)) return 0;
    if (!load_floats(f, tc->routed_down_proj, E * H * I)) return 0;
    if (!load_floats(f, tc->shared_gate_proj, (S * I) * H)) return 0;
    if (!load_floats(f, tc->shared_up_proj, (S * I) * H)) return 0;
    if (!load_floats(f, tc->shared_down_proj, H * (S * I))) return 0;

    if (!load_ints(f, tc->expected_topk_idx, T * K)) return 0;
    if (!load_floats(f, tc->expected_topk_weight, T * K)) return 0;
    if (!load_floats(f, tc->expected_shared_out, T * H)) return 0;
    if (!load_floats(f, tc->expected_routed_out, T * H)) return 0;
    if (!load_floats(f, tc->expected_final_out, T * H)) return 0;

    return 1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tests.txt>\\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int num_cases = 0;
    if (fscanf(f, "%d", &num_cases) != 1) {
        fprintf(stderr, "Failed to read number of test cases\\n");
        fclose(f);
        return 1;
    }

    const float atol = 2e-4f;
    int all_ok = 1;

    for (int case_id = 0; case_id < num_cases; case_id++) {
        TestCase tc;
        memset(&tc, 0, sizeof(TestCase));

        if (!load_testcase(f, &tc)) {
            fprintf(stderr, "Failed to load test case %d\\n", case_id);
            fclose(f);
            return 1;
        }

        int T = tc.num_tokens;
        int H = tc.hidden_size;
        int K = tc.top_k;

        int* got_topk_idx = (int*)malloc(sizeof(int) * T * K);
        float* got_topk_weight = (float*)malloc(sizeof(float) * T * K);
        float* got_shared_out = (float*)malloc(sizeof(float) * T * H);
        float* got_routed_out = (float*)malloc(sizeof(float) * T * H);
        float* got_final_out = (float*)malloc(sizeof(float) * T * H);

        compute_gate(&tc, got_topk_idx, got_topk_weight);
        compute_shared(&tc, got_shared_out);
        compute_routed(&tc, got_topk_idx, got_topk_weight, got_routed_out);

        for (int i = 0; i < T * H; i++) {
            got_final_out[i] = got_shared_out[i] + got_routed_out[i];
        }

        int idx_ok = exact_int_match(got_topk_idx, tc.expected_topk_idx, T * K);
        float w_err = max_abs_diff(got_topk_weight, tc.expected_topk_weight, T * K);
        float s_err = max_abs_diff(got_shared_out, tc.expected_shared_out, T * H);
        float r_err = max_abs_diff(got_routed_out, tc.expected_routed_out, T * H);
        float f_err = max_abs_diff(got_final_out, tc.expected_final_out, T * H);

        int case_ok = idx_ok && (w_err <= atol) && (s_err <= atol) && (r_err <= atol) && (f_err <= atol);

        printf(
            "case=%d idx_ok=%d weight_max_abs=%.8f shared_max_abs=%.8f routed_max_abs=%.8f final_max_abs=%.8f status=%s\\n",
            case_id,
            idx_ok,
            w_err,
            s_err,
            r_err,
            f_err,
            case_ok ? "PASS" : "FAIL"
        );

        if (!case_ok) all_ok = 0;

        free(got_topk_idx);
        free(got_topk_weight);
        free(got_shared_out);
        free(got_routed_out);
        free(got_final_out);
        free_testcase(&tc);
    }

    fclose(f);
    return all_ok ? 0 : 2;
}
"""


@app.function(image=image, timeout=60 * 20)
def build_and_test():
    import numpy as np
    import torch

    try:
        from transformers import DeepseekV3Config
    except Exception:
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    except Exception as e:
        raise RuntimeError(f"Could not import DeepseekV3MoE from transformers: {e}")

    out_dir = Path("/tmp/deepseek_moe_modal")
    out_dir.mkdir(parents=True, exist_ok=True)

    c_path = out_dir / "deepseek_moe.c"
    exe_path = out_dir / "deepseek_moe"
    tests_path = out_dir / "tests.txt"
    summary_path = out_dir / "summary.json"

    c_path.write_text(C_SOURCE)

    # Small config so tests run fast, but keeps the same structure as the HF MoE block.
    def make_config():
        cfg = DeepseekV3Config(
            hidden_size=8,
            intermediate_size=32,         # unused by MoE block here, but harmless
            moe_intermediate_size=4,
            n_routed_experts=6,
            n_shared_experts=2,
            num_experts_per_tok=3,
            hidden_act="silu",
            routed_scaling_factor=1.0,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=3,
            topk_group=2,
            norm_topk_prob=True,
        )
        cfg.ep_size = 1
        return cfg

    def tensor_to_list(x):
        return x.detach().cpu().float().reshape(-1).tolist()

    def ints_to_list(x):
        return x.detach().cpu().int().reshape(-1).tolist()

    def compute_reference_parts(moe, x):
        """
        Reconstruct the exact blocks we want to test:
        1. gate -> topk_idx, topk_weight
        2. shared expert output
        3. routed expert weighted sum
        4. final output
        """
        with torch.no_grad():
            topk_idx, topk_weight = moe.gate(x)
            flat_x = x.view(-1, x.shape[-1])

            # shared
            shared_out = moe.shared_experts(x).view(-1, x.shape[-1])

            # routed
            num_tokens = flat_x.shape[0]
            hidden = flat_x.shape[1]
            routed_out = torch.zeros((num_tokens, hidden), dtype=flat_x.dtype)

            for token_i in range(num_tokens):
                xt = flat_x[token_i : token_i + 1]
                for k in range(moe.num_experts_per_tok):
                # for k in range(moe.config.num_experts_per_tok):
                    expert_id = int(topk_idx.view(-1, moe.num_experts_per_tok)[token_i, k].item())
                    weight = topk_weight.view(-1, moe.num_experts_per_tok)[token_i, k]
                    routed_out[token_i] += weight * moe.experts[expert_id](xt).squeeze(0)

            final_out = routed_out + shared_out
            return topk_idx.view(-1, moe.num_experts_per_tok), topk_weight.view(-1, moe.num_experts_per_tok), shared_out, routed_out, final_out

    # Make deterministic test cases.
    num_cases = 4
    torch.manual_seed(1234)
    np.random.seed(1234)

    cases = []

    for case_id in range(num_cases):
        seed = 100 + case_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = make_config()
        moe = DeepseekV3MoE(cfg).eval()

        # deterministic but varied token counts
        num_tokens = 2 + case_id  # 2,3,4,5
        x = torch.randn(1, num_tokens, cfg.hidden_size, dtype=torch.float32)

        topk_idx, topk_weight, shared_out, routed_out, final_out = compute_reference_parts(moe, x)

        case = {
            "num_tokens": num_tokens,
            "hidden_size": cfg.hidden_size,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "n_routed_experts": cfg.n_routed_experts,
            "n_shared_experts": cfg.n_shared_experts,
            "top_k": cfg.num_experts_per_tok,
            "routed_scaling_factor": float(cfg.routed_scaling_factor),
            "x": tensor_to_list(x.view(-1, cfg.hidden_size)),
            "gate_weight": tensor_to_list(moe.gate.weight),
            "routed_gate_proj": [],
            "routed_up_proj": [],
            "routed_down_proj": [],
            "shared_gate_proj": tensor_to_list(moe.shared_experts.gate_proj.weight),
            "shared_up_proj": tensor_to_list(moe.shared_experts.up_proj.weight),
            "shared_down_proj": tensor_to_list(moe.shared_experts.down_proj.weight),
            "expected_topk_idx": ints_to_list(topk_idx),
            "expected_topk_weight": tensor_to_list(topk_weight),
            "expected_shared_out": tensor_to_list(shared_out),
            "expected_routed_out": tensor_to_list(routed_out),
            "expected_final_out": tensor_to_list(final_out),
        }

        for expert in moe.experts:
            case["routed_gate_proj"].extend(tensor_to_list(expert.gate_proj.weight))
            case["routed_up_proj"].extend(tensor_to_list(expert.up_proj.weight))
            case["routed_down_proj"].extend(tensor_to_list(expert.down_proj.weight))

        cases.append(case)

    # Write simple text format for C.
    with open(tests_path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(
                f"{c['num_tokens']} {c['hidden_size']} {c['moe_intermediate_size']} "
                f"{c['n_routed_experts']} {c['n_shared_experts']} {c['top_k']} "
                f"{c['routed_scaling_factor']}\n"
            )

            def write_list(vals):
                f.write(" ".join(f"{float(v):.9g}" for v in vals) + "\n")

            def write_int_list(vals):
                f.write(" ".join(str(int(v)) for v in vals) + "\n")

            write_list(c["x"])
            write_list(c["gate_weight"])
            write_list(c["routed_gate_proj"])
            write_list(c["routed_up_proj"])
            write_list(c["routed_down_proj"])
            write_list(c["shared_gate_proj"])
            write_list(c["shared_up_proj"])
            write_list(c["shared_down_proj"])

            write_int_list(c["expected_topk_idx"])
            write_list(c["expected_topk_weight"])
            write_list(c["expected_shared_out"])
            write_list(c["expected_routed_out"])
            write_list(c["expected_final_out"])

    # Compile C.
    subprocess.run(
        ["gcc", "-O2", "-std=c99", str(c_path), "-lm", "-o", str(exe_path)],
        check=True,
    )

    # Run tests.
    result = subprocess.run(
        [str(exe_path), str(tests_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    summary = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "files": {
            "c_source": str(c_path),
            "compiled_binary": str(exe_path),
            "tests": str(tests_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("===== C TEST OUTPUT =====")
    print(result.stdout)
    if result.stderr:
        print("===== STDERR =====")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"C tests failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    return {
        "message": "All tests passed",
        "summary_path": str(summary_path),
        "tests_path": str(tests_path),
        "c_source_path": str(c_path),
    }


@app.local_entrypoint()
def main():
    result = build_and_test.remote()
    print(json.dumps(result, indent=2))