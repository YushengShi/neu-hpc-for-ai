# Week_01 Assignment: Threaded Matrix Multiplication in C

This directory contains the implementation and evaluation of matrix multiplication in C, including a single-threaded version, a multi-threaded version using pthreads, correctness tests, and performance benchmarking.

All questions from the assignment are answered explicitly below, with references to the corresponding source files.

---

## Files

- matmul.h  
  Declares:
  - matmul_serial(...)
  - matmul_pthread(..., num_threads)

- matmul.c  
  Implements:
  - matmul_serial (single-threaded baseline)
  - matmul_pthread (multi-threaded using pthreads)

- test.c  
  Runs correctness tests by comparing:
  - C_serial = matmul_serial(...)
  - C_par    = matmul_pthread(...)  
  Prints PASS for each test case and exits with an error if any mismatch is found.

- benchmark.c  
  Measures runtime and prints speedup:
  - serial runtime T1
  - parallel runtime Tn
  - speedup = T1 / Tn

- Makefile  
  Builds the test and benchmark executables and provides a clean target.

---

## 1. Single-threaded Matrix Multiplication

**Question:**  
Implement a single-threaded version of matrix multiplication in C.

**Answer:**  
The single-threaded implementation is provided by the function:

void matmul_serial(int m, int n, int p, const double *A, const double *B, double *C)

This function is implemented in matmul.c and declared in matmul.h.

The implementation uses the standard triple-nested loop to compute:

C = A × B

where:
- A is m × n
- B is n × p
- C is m × p

Matrices are stored as flat row-major arrays.

---

## 2. Test Cases for Correctness

**Question:**  
Write test cases and check a wide range of matrix dimensions, including corner cases.

**Answer:**  
Correctness tests are implemented in test.c.

The following test cases are explicitly included:
- A = 1×1, B = 1×1
- A = 1×1, B = 1×5
- A = 2×1, B = 1×3
- A = 2×2, B = 2×2
- Additional non-square matrices
- Randomly generated matrix sizes

For each test case:
1. The serial result is computed using matmul_serial.
2. The parallel result is computed using matmul_pthread.
3. All output elements are compared within floating-point tolerance.

Each successful test prints:
PASS: A=mxn, B=nxp, threads=t

If any mismatch is found, the program prints the failing index and exits with an error.

Code location: test.c

---

## 3. Multi-threaded Matrix Multiplication Using pthreads

**Question:**  
Implement a multi-threaded version of matrix multiplication in C using pthreads.

**Answer:**  
Yes, pthreads are used.

The function:

void matmul_pthread(int m, int n, int p,
                    const double *A, const double *B, double *C,
                    int num_threads)

is implemented in matmul.c and declared in matmul.h.

Parallelization strategy:
- The output matrix C is partitioned by rows.
- Each thread computes a disjoint subset of rows.
- No two threads write to the same output elements.

Because each thread writes to a unique region of C, no mutexes or locks are required.

---

## 4. Verifying Correctness of the Multi-threaded Version

**Question:**  
Check that the test cases pass for the multi-threaded version.

**Answer:**  
Correctness is verified by directly comparing the multi-threaded output against the single-threaded reference output for all test cases.

Running:

./test

confirms correctness for both implementations. The program exits immediately if any mismatch is detected.

---

## 5. Measuring Speedup with Different Thread Counts

**Question:**  
Measure the speedup for different thread counts: 1, 4, 16, 32, 64, 128.

**Answer:**  
Speedup measurements are implemented in benchmark.c.

The benchmark:
- Measures serial runtime T1.
- Measures parallel runtime Tn for each thread count.
- Computes speedup as:

speedup = T1 / Tn

The following thread counts are evaluated:
1, 4, 16, 32, 64, 128

Results are printed to stdout.

---

## 6. Use of Large Matrices

**Question:**  
Use large matrices so that speedup is measurable.

**Answer:**  
The benchmark uses large matrix sizes by default (e.g., 1024×1024), ensuring computation time dominates thread overhead.

Custom matrix sizes can be specified when running:

./benchmark m n p

This allows evaluation of scalability across different problem sizes.

---

## Build Instructions

Build all executables:

make

---

## Running Correctness Tests

Run:

./test

Each test prints a PASS message. Any failure terminates the program with an error.

---

## Running the Benchmark

Run with default matrix sizes:

./benchmark

Run with custom matrix sizes:

./benchmark 512 512 512

---

## Cleaning Build Artifacts

Before submission, remove object files and executables using:

make clean

This leaves only source files in the directory.

---

## Notes on Performance

Observed speedup may be limited by:
- Number of available CPU cores
- Memory bandwidth
- Thread creation and scheduling overhead

Such behavior is expected on real systems.
