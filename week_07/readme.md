# Week 7 – DeepSeekV3 MoE Operator

## Overview

This project implements the Mixture-of-Experts (MoE) operator from DeepSeekV3 in pure C (CPU-only, no parallelism, no CUDA) and verifies correctness using automatically generated test cases from Hugging Face Transformers.

The objectives of this assignment are:
- Understand the DeepSeekMoE architecture
- Break down the MoE operator into components
- Generate test cases using a reference implementation
- Reimplement the operator in C
- Verify correctness

---

## 1. DeepSeekMoE Paper Summary

DeepSeekMoE is a sparse mixture-of-experts architecture designed to scale model capacity efficiently.

### Key Ideas

Routed Experts  
- Each token is routed to a subset of experts  
- A gating network selects the top-k experts  

Shared Experts  
- A shared expert processes all tokens  
- Improves stability and generalization  

Sparse Computation  
- Only a few experts are activated per token  
- Reduces compute while maintaining performance  

Gating Mechanism  
- Linear projection → sigmoid → top-k → normalization  

---

## 2. MoE Operator Breakdown

The DeepSeekV3 MoE operator consists of four main components:

### 2.1 Gating (Routing)

Input:
    x: [num_tokens, hidden_size]

Steps:
- Compute logits using linear projection  
- Apply sigmoid activation  
- Select top-k experts  
- Normalize weights  

Outputs:
- topk_idx  
- topk_weight  

---

### 2.2 Shared Experts

    shared_out = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

---

### 2.3 Routed Experts

    routed_out = sum(weight_i * Expert_i(x))

Each expert:

    Expert(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

---

### 2.4 Final Output

    output = shared_out + routed_out

---

## 3. Test Case Generation (Hugging Face)

We use Hugging Face Transformers as the reference implementation.

### Step 1 — Load DeepSeekV3 MoE

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    from transformers import DeepseekV3Config

---

### Step 2 — Create a small config

    cfg = DeepseekV3Config(
        hidden_size=8,
        moe_intermediate_size=4,
        n_routed_experts=6,
        n_shared_experts=2,
        num_experts_per_tok=3
    )

---

### Step 3 — Generate input

    x = torch.randn(1, num_tokens, cfg.hidden_size)

---

### Step 4 — Extract reference outputs

- Top-k indices  
- Top-k weights  
- Shared output  
- Routed output  
- Final output  

---

### Step 5 — Save to file

All values are written to tests.txt and used by the C program.

---

## 4. C Implementation (No Parallelism)

The MoE operator is implemented entirely in pure C.

### Design Constraints

- No CUDA  
- No multithreading  
- Deterministic execution  
- Matches Hugging Face outputs  

---

### Key Components

Matrix multiplication:

    matvec_rowmajor(W, x, y)

Activation (SiLU):

    silu(x) = x / (1 + exp(-x))

Expert computation:

    out = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

Routing logic:
- Compute scores  
- Select top-k  
- Normalize  

Aggregation:

    output += weight * expert_output

---

## 5. Verification Strategy

Each component is validated independently:

- Gating indices → exact match  
- Gating weights → max absolute error  
- Shared output → max absolute error  
- Routed output → max absolute error  
- Final output → max absolute error  

Error tolerance:

    2e-4

---

## 6. Running on Modal

Run:

    modal run deepseek_moe_modal.py

Pipeline:

1. Build container  
2. Install dependencies  
3. Generate test cases  
4. Compile C code (gcc)  
5. Run tests  
6. Compare outputs  

---

## 7. Example Output

    case=0 idx_ok=1 weight_max_abs=0.00000012 shared_max_abs=0.00000021 routed_max_abs=0.00000019 final_max_abs=0.00000025 status=PASS

---

## 8. Challenges and Solutions

Floating point precision  
- Used double accumulation in C  

Top-k consistency  
- Implemented deterministic sorting  

Tensor layout mismatch  
- Carefully matched indexing  

Debugging  
- Verified intermediate outputs step-by-step  

---

## 9. Key Learnings

- MoE enables scaling with sparse compute  
- Routing is critical in modern LLMs  
- Real implementations differ from theory  
- Numerical validation is essential  

---

## 10. Conclusion

This project successfully:
- Reimplemented the DeepSeekV3 MoE operator  
- Built a pure C version  
- Verified correctness against Hugging Face  

---

## Files

    deepseek_moe_modal.py
    deepseek_moe.c
    tests.txt
    README.md

---

## References

- DeepSeekMoE Paper  
- Hugging Face Transformers  
- HPC course materials  