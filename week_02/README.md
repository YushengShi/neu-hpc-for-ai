# Week_02 Assignment — Naive GEMM Kernel (CUDA on Modal)

## Overview

This assignment implements a naive GEMM (Generalized Matrix Multiplication) kernel in CUDA, executed on a remote NVIDIA GPU using Modal.

The computation performed is:

C ← α · op(A) · op(B) + β · C

Where:
- op(A) is either A or Aᵀ
- op(B) is either B or Bᵀ
- α and β are scalar coefficients
- C is updated in place

Constraints:
- Global memory only (no shared memory, no tiling)
- No high-level CUDA libraries (cuBLAS, cuDNN)
- Focus on correctness and clarity

---

## Supported Operations

The kernel supports all four GEMM variants:

- C ← α · A · B + β · C
- C ← α · Aᵀ · B + β · C
- C ← α · A · Bᵀ + β · C
- C ← α · Aᵀ · Bᵀ + β · C

---

## Files

This Modal-based implementation only requires the following file:

- modal_gemm.py  
  A Modal script that:
  - creates a GPU-enabled container
  - generates CUDA source files inside the container
  - compiles them using nvcc
  - runs the GEMM kernel and validates results

No Makefile or local CUDA installation is required.

---

## How to Run

Step 1: Install Modal (one time only)

Run the following commands:

    pip install modal
    modal setup

Step 2: Run the assignment

From the project directory, run:

    modal run modal_gemm.py

This command performs the following steps:

1. Launches a GPU-enabled container on Modal
2. Generates CUDA source files inside the container
3. Compiles the code using nvcc
4. Executes the GEMM kernel on the GPU
5. Validates GPU results against a CPU reference implementation

---

## Output

The program prints the maximum absolute difference between CPU and GPU results for each transpose configuration.

Example output:

    [C = a*A *B + b*C] max |GPU-CPU| = 7.15e-07
    [C = a*A^T*B + b*C] max |GPU-CPU| = 7.15e-07
    [C = a*A *B^T + b*C] max |GPU-CPU| = 7.15e-07
    [C = a*A^T*B^T + b*C] max |GPU-CPU| = 7.15e-07
    All cases passed.

Small floating-point differences are expected.

---

## Design Notes

- Each CUDA thread computes one output element C[row, col]
- A two-dimensional grid and two-dimensional thread blocks are used
- Matrix transposition is handled through indexing logic
- The implementation is intentionally naive and not performance-optimized

---

## Summary

This assignment demonstrates a correct and naive CUDA GEMM implementation that supports optional transposition of input matrices, updates C in place, and executes on a remote NVIDIA GPU using Modal.

---

## Explicit Answers to Assignment Questions

### 1. Implement a GEMM kernel in CUDA. Do not use cuBLAS or cuDNN.

**Answer:**  
A custom CUDA kernel is implemented using `__global__` functions. No high-level CUDA libraries such as cuBLAS or cuDNN are used. All matrix multiplication logic is written manually.

---

### 2. Only implement a naive version that reads and writes global memory.

**Answer:**  
The implementation is fully naive:
- All matrix elements are read from global memory
- All output values are written to global memory
- No shared memory, tiling, or other optimizations are used

---

### 3. What does GEMM stand for?

**Answer:**  
GEMM stands for **Generalized Matrix Multiplication**.

---

### 4. Define standard matrix multiplication: C = AB

**Answer:**  
Given:
- A is an m × k matrix
- B is a k × n matrix

The result:
- C = AB is an m × n matrix

This case is directly supported by the kernel.

---

### 5. Define generalized matrix multiplication: D = αAB + βC

**Answer:**  
The kernel computes the generalized form:

C ← α · op(A) · op(B) + β · C

Where α and β are scalar coefficients.  
The result is written **in place** to C, so no separate matrix D is allocated.

---

### 6. Are α and β read-only?

**Answer:**  
Yes. The scalar coefficients α and β are passed as input parameters and are never modified inside the kernel.

---

### 7. Extend the kernel to optionally transpose A or B.

**Answer:**  
The kernel supports optional transposition via boolean flags:
- `transposeA` controls whether A or Aᵀ is used
- `transposeB` controls whether B or Bᵀ is used

Transposition is handled through indexing logic only, without creating temporary matrices.

---

### 8. Update C in place instead of allocating D.

**Answer:**  
The kernel updates C directly:

C[row, col] = α · (op(A) · op(B)) + β · C[row, col]

No additional output matrix is allocated.

---

### 9. Supported forms required by the assignment

**Answer:**  
All required forms are supported:

- C ← α · A · B + β · C  
- C ← α · Aᵀ · B + β · C  
- C ← α · A · Bᵀ + β · C  
- C ← α · Aᵀ · Bᵀ + β · C  

Each form is tested and validated.

---

### 10. How is correctness verified?

**Answer:**  
A CPU reference implementation computes the same GEMM operation.  
The GPU result is compared against the CPU result, and the maximum absolute difference is reported.

---

### 11. Execution environment

**Answer:**  
The code is executed on a real NVIDIA GPU using Modal.  
Compilation and execution occur inside a GPU-enabled container using `nvcc`.

---

### 12. Final confirmation

**Answer:**  
This implementation satisfies **all functional and conceptual requirements** stated in the assignment.
