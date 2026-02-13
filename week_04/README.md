# week_04 — FlashAttention-2 (Algorithm 1) Forward Pass  
**Deliverables:**  
1) **Unparallelized C** implementation of FlashAttention-2 **Algorithm 1** (forward pass)  
2) **Parallelized CUDA** implementation of the same algorithm

This repo implements both parts and includes a built-in correctness check against a naive attention reference.

---

## 1. Problem Summary (What the assignment asks)

We implement the forward pass of attention:

\[
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

where \(Q, K, V, O \in \mathbb{R}^{N \times d}\).

The assignment requires:
- **(1) CPU:** Implement FlashAttention-2 Algorithm 1 in **unparallelized C**
- **(2) GPU:** Implement FlashAttention-2 Algorithm 1 in **parallelized CUDA**

Both implementations must be **correct** (optimization is not the focus yet).

---

## 2. Files and What They Contain

- `flash.h`  
  Declares the three entry points:
  - `attention_naive_cpu(...)` (reference)
  - `flash_forward_cpu(...)` (Part 1)
  - `flash_forward_cuda(...)` (Part 2)

- `flash_cpu.c`  
  Contains:
  - `attention_naive_cpu(...)`: baseline attention for correctness checking  
  - `flash_forward_cpu(...)`: **unparallelized C** implementation of FlashAttention-2 Algorithm 1

- `flash_cuda.cu`  
  Contains:
  - `flash_forward_kernel<<<...>>>(...)`: **parallelized CUDA** implementation
  - `flash_forward_cuda(...)`: kernel launcher

- `main.c`  
  Test harness:
  - generates random Q/K/V
  - runs naive CPU
  - runs flash CPU
  - runs flash CUDA
  - prints max absolute errors

---

## 3. How the code answers Question (1): Unparallelized C

### 3.1 Approach (Algorithm 1 structure)
The CPU implementation `flash_forward_cpu(...)` follows FlashAttention-2 Algorithm 1 by tiling:

- Split `Q` into row tiles of size `Br × d`  
- Split `K` and `V` into column tiles of size `Bc × d`

For each Q tile `Qi`:
- Maintain per-row running values:
  - `m` = running maximum of logits (for numerical stability)
  - `l` = running sum of exponentials (online softmax normalizer)
  - `Oacc` = running unnormalized output accumulator

### 3.2 Online softmax update (core correctness)
For each tile `(Qi, Kj, Vj)`:

1. Compute local scores:
   \[
   S^{(j)} = \frac{Qi Kj^T}{\sqrt{d}}
   \]

2. Update rowwise max:
   \[
   m^{(j)} = \max(m^{(j-1)}, \text{rowmax}(S^{(j)}))
   \]

3. Update rowwise normalizer:
   \[
   l^{(j)} = l^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}} + \text{rowsum}(e^{S^{(j)} - m^{(j)}})
   \]

4. Update output accumulator:
   \[
   O^{(j)} = O^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}} + e^{S^{(j)} - m^{(j)}}V_j
   \]

After all K/V tiles:
- Normalize:
  \[
  O_i = \frac{O^{(T_c)}}{l^{(T_c)}}
  \]
- Logsumexp output:
  \[
  L_i = m^{(T_c)} + \log(l^{(T_c)})
  \]

### 3.3 Where this is implemented in code
In `flash_cpu.c`:
- tiling loops: `for (ti ...)` over `Tr`, and inside `for (tj ...)` over `Tc`
- online softmax state: arrays `m[r]`, `l[r]`
- output accumulator: `Oacc[r*d + k]`

This is **fully unparallelized**: everything is standard C loops.

✅ Therefore, **Question (1) is answered** by `flash_forward_cpu(...)`.

---

## 4. How the code answers Question (2): Parallelized CUDA

### 4.1 Parallelization strategy
The CUDA implementation uses a correctness-first mapping:

- **One CUDA block handles one Q tile** (one `Qi`)
- **One thread handles one row** within that tile

So for a tile size `Br`, we launch:
- `grid.x = ceil(N / Br)` blocks
- `block.x = Br` threads per block

### 4.2 Shared memory tiling
Each block iterates over all `Kj, Vj` tiles.  
For each tile, the block loads:
- `Kj` into shared memory
- `Vj` into shared memory

Shared memory layout:
- `Ksh`: `Bc * d`
- `Vsh`: `Bc * d`

This matches the FlashAttention idea: keep working tiles on-chip while streaming through all K/V blocks.

### 4.3 Per-thread online softmax state
Each thread keeps:
- `m` and `l` scalars for its row
- `Orow[d]` accumulator (unnormalized)

Then it performs the same online update equations tile-by-tile as the CPU.

After all tiles:
- normalize `Orow` by `l`
- write `O` and `L` back to global memory (HBM)

### 4.4 Where this is implemented in code
In `flash_cuda.cu`:
- kernel: `flash_forward_kernel(...)`
- launcher: `flash_forward_cuda(...)`

Parallel work:
- cooperative loading to shared memory: loop `for (idx = t; idx < total; idx += blockDim.x)`
- per-thread compute for its row `i`

✅ Therefore, **Question (2) is answered** by `flash_forward_kernel` + `flash_forward_cuda(...)`.

---

## 5. Correctness Verification

We verify both implementations against a naive reference:

1. `attention_naive_cpu(...)` computes:
   \[
   O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
   \]
   and `L = logsumexp` per row.

2. Compare results:
- CPU flash vs naive
- GPU flash vs naive

In `main.c` we report:
- `max|O_naive - O_flash|`
- `max|L_naive - L_flash|`

Typical expected tolerance (float32):
- `~1e-3` or smaller

---

## 6. How to Build and Run

### Option A: Local build (with NVCC)
If you have CUDA locally:

```bash
nvcc -O2 main.c flash_cpu.c flash_cuda.cu -o flash_test -lm
./flash_test 256 64 64 64
