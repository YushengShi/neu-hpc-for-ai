import numpy as np
import json
from pathlib import Path

np.random.seed(1234)

OUT_DIR = Path("testdata/case_001")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config (same as before)
B, S, H = 2, 3, 8
TOKENS = B * S
E = 8
TOPK = 2
MID = 6

def save_f32(name, arr):
    arr.astype(np.float32).tofile(OUT_DIR / name)

def save_i32(name, arr):
    arr.astype(np.int32).tofile(OUT_DIR / name)

def silu(x):
    return x / (1 + np.exp(-x))

def expert_forward(x, w1, w2, w3):
    g = silu(x @ w1.T)
    u = x @ w2.T
    return (g * u) @ w3.T

# ===== generate weights =====
gate_weight = np.random.randn(E, H) * 0.2
bias = np.random.randn(E) * 0.05

experts = []
for _ in range(E):
    experts.append({
        "w1": np.random.randn(MID, H) * 0.1,
        "w2": np.random.randn(MID, H) * 0.1,
        "w3": np.random.randn(H, MID) * 0.1
    })

shared = {
    "w1": np.random.randn(MID, H) * 0.1,
    "w2": np.random.randn(MID, H) * 0.1,
    "w3": np.random.randn(H, MID) * 0.1
}

# ===== input =====
x = np.random.randn(TOKENS, H) * 0.2

# ===== gate =====
logits = x @ gate_weight.T
scores = 1 / (1 + np.exp(-logits))
scores_choice = scores + bias

topk_idx = np.argsort(-scores_choice, axis=1)[:, :TOPK]
topk_weight = np.take_along_axis(scores, topk_idx, axis=1)

# normalize
topk_weight = topk_weight / (topk_weight.sum(axis=1, keepdims=True) + 1e-9)
topk_weight *= 2.5

# ===== forward =====
y = np.zeros((TOKENS, H))

for t in range(TOKENS):
    for k in range(TOPK):
        e = topk_idx[t, k]
        out = expert_forward(x[t:t+1], experts[e]["w1"], experts[e]["w2"], experts[e]["w3"])
        y[t] += topk_weight[t, k] * out.squeeze()

    shared_out = expert_forward(x[t:t+1], shared["w1"], shared["w2"], shared["w3"])
    y[t] += shared_out.squeeze()

# ===== save =====
save_f32("input.bin", x)
save_f32("gate_weight.bin", gate_weight)
save_f32("gate_bias_correction.bin", bias)
save_f32("logits.bin", logits)
save_f32("scores.bin", scores)
save_i32("topk_idx.bin", topk_idx)
save_f32("topk_weight.bin", topk_weight)
save_f32("final_out.bin", y)

for i, e in enumerate(experts):
    save_f32(f"expert_{i}_gate_proj.bin", e["w1"])
    save_f32(f"expert_{i}_up_proj.bin", e["w2"])
    save_f32(f"expert_{i}_down_proj.bin", e["w3"])

save_f32("shared_gate_proj.bin", shared["w1"])
save_f32("shared_up_proj.bin", shared["w2"])
save_f32("shared_down_proj.bin", shared["w3"])

print("✅ Testcase generated")