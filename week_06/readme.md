# Week 6 – DeepSeekV3 MoE Operator in Pure C

## Overview

This project implements the **Mixture-of-Experts (MoE) operator for DeepSeekV3** in **pure C**, with:

- **no CUDA**
- **no parallelism**
- **no OpenMP / pthreads**
- **no GPU dependency in the C implementation**

The project also includes a **Python reference exporter** that uses the Hugging Face DeepSeekV3 implementation to automatically generate:

- test inputs
- intermediate expected outputs
- final expected outputs

Those exported files are then used by the C test harness to verify correctness.

---

## Assignment Questions Answered

### 1. Read the DeepSeekMoE paper

The DeepSeekMoE paper motivates why MoE models can be improved by:

- using **fine-grained expert segmentation**, where experts are split into smaller specialized experts
- using **shared experts**, which capture common knowledge that should always be available
- combining routed experts and shared experts so the model benefits from both specialization and stable shared capacity

In practice for this assignment, the **paper gives the architectural intuition**, while the **Hugging Face DeepSeekV3 implementation is treated as the behavioral reference** for exact inputs, routing, and outputs.

---

### 2. Generate test cases for each block in the MoE operator using Hugging Face as the reference

This project does that through `export_testcases.py`.

It generates a small deterministic DeepSeekV3-style MoE test case and exports:

- `input.bin`
- `gate_weight.bin`
- `gate_bias_correction.bin`
- `logits.bin`
- `scores.bin`
- `topk_idx.bin`
- `topk_weight.bin`
- `expert_out.bin`
- `shared_out.bin`
- `final_out.bin`

It also exports each routed expert’s MLP weights and the shared expert’s weights.

This makes Hugging Face the **ground-truth implementation**, and the C code must match those outputs.

---

### 3. Implement the MoE operator for DeepSeekV3 in pure C with no parallelism and no CUDA

This project implements the following parts in pure C:

- router linear projection
- sigmoid routing scores
- grouped top-k expert selection
- normalized/scaled routing weights
- expert MLP forward pass
- weighted accumulation of routed expert outputs
- shared expert forward pass
- final MoE output assembly

The C code is contained in:

- `moe.h`
- `moe.c`

The implementation uses only:

- standard C
- loops
- heap allocations
- math library (`expf`, etc.)

No GPU or parallel runtime is used.

---

### 4. Check that the implementation passes the generated test cases

This is done by `test_moe.c`.

It loads the generated test data and compares the C outputs against the Hugging Face reference outputs for:

- router logits
- routing scores
- selected top-k expert indices
- routing weights
- final MoE output

If everything matches within tolerance, it prints:

```text
RESULT: PASS