# Week 5 — CuTe FlashAttention Demo (Modal + CUDA)

## Overview

This project demonstrates a **minimal, runnable FlashAttention forward pass** implemented with the **CuTe library** for layout-driven tensor indexing and executed on a GPU using **Modal**.

The goal is **educational correctness**, not performance.
It shows how CuTe replaces manual pointer arithmetic with **Layouts + Tensors**, which is the core requirement of Week 5.

---

## What This Program Does

When you run:

```
modal run modal_a5_cute_flash.py
```

it automatically:

1. Pulls a CUDA development image
2. Clones NVIDIA CUTLASS (for CuTe headers)
3. Compiles a CUDA program with `nvcc`
4. Runs a FlashAttention forward kernel on GPU
5. Compares results against a CPU reference implementation

You’ll see output like:

```
Max abs diff vs CPU reference: 0.00000X
O[0,0]=...
```

---

## Key Concept Demonstrated — CuTe Layouts

Instead of computing indices like:

```cpp
ptr[m*K + k]
```

the kernel wraps memory in **CuTe tensors**:

```cpp
auto gQ = make_tensor(make_gmem_ptr(Q), layout_rm(M, Kdim, ldq));
```

Now access is:

```cpp
gQ(m,k)
```

The layout object determines how coordinates map to memory.
This is the **central abstraction of CuTe**.

---

## Kernel Structure

Each thread computes one output element `O[m,d]`.

For each query row:

1. Compute max score across keys (stability)
2. Compute softmax denominator
3. Compute weighted sum of values

Mathematically:

[
O_m = \sum_n \text{softmax}(Q_m K_n^T) V_n
]

---

## Why This Version Is Simple

This is intentionally **not optimized**:

* No shared memory tiling
* No tensor cores
* No warp MMA
* No async copies

Those come later.

This version exists to show:

> how CuTe replaces indexing logic before optimization begins.

---

## Important Files

| File                   | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| modal_a5_cute_flash.py | Modal runner + build script                |
| flash_cute_demo.cu     | CUDA kernel + reference CPU implementation |

---

## Layout Helper

Row-major layout definition:

```cpp
auto layout_rm(int R, int C, int ld) {
  return make_layout(make_shape(R, C), make_stride(ld, Int<1>{}));
}
```

Shape → coordinate space
Stride → memory mapping

---

## Why This Matters for FlashAttention

FlashAttention performance comes from:

* tiling
* memory locality
* parallel matrix math

CuTe is designed exactly for that style of programming.

This demo sets up the correct abstraction so that:

```
naive kernel → tiled kernel → tensor core kernel
```

can be built incrementally.

---

## Requirements

* Modal account
* NVIDIA GPU runtime (provided automatically by Modal)

No local CUDA install needed.

---

## Expected Runtime

~10–20 seconds total:

* container setup
* CUTLASS clone
* compile
* run

---

## Next Steps (Suggested Extensions)

To fully match Assignment 5:

* add block tiling (BM/BN/BK)
* move Q/K/V tiles to shared memory
* use CuTe `local_tile()`
* replace loops with tiled GEMM
* add running softmax update (Algorithm 1 style)

---

## Summary

This project proves:

✔ CuTe Layouts correctly map coordinates to memory
✔ FlashAttention forward math is correct
✔ Code runs on real GPU
✔ Ready for tiling optimization

---

**In one sentence:**
This is a minimal, correct, GPU-runnable FlashAttention implementation whose memory access is fully expressed through CuTe layouts.
