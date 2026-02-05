import modal

app = modal.App("cuda-matmul")

# CUDA kernel source code
cuda_source = """
// Kernel 1: Naive Implementation
extern "C" __global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// Kernel 2: Global Memory Coalescing
extern "C" __global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
                                 const float *B, float beta, float *C) {
    const int BLOCKSIZE = 32;
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// Kernel 3: Shared Memory Cache-Blocking
extern "C" __global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    
    __shared__ float As[1024];
    __shared__ float Bs[1024];

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    const unsigned int threadRow = threadIdx.x / BN;
    const unsigned int threadCol = threadIdx.x % BN;

    const float *A_ptr = A + cRow * BM * K;
    const float *B_ptr = B + cCol * BN;
    float *C_ptr = C + cRow * BM * N + cCol * BN;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[threadRow * BK + threadCol] = A_ptr[threadRow * K + threadCol];
        Bs[threadRow * BN + threadCol] = B_ptr[threadRow * N + threadCol];

        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            tmp += As[threadRow * BK + dotIdx] * Bs[dotIdx * BN + threadCol];
        }
        __syncthreads();
    }
    C_ptr[threadRow * N + threadCol] = alpha * tmp + beta * C_ptr[threadRow * N + threadCol];
}

// Kernel 4: 1D Blocktiling
extern "C" __global__ void sgemm_1D_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    
    __shared__ float As[512];
    __shared__ float Bs[512];

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    const float *A_ptr = A + cRow * BM * K;
    const float *B_ptr = B + cCol * BN;
    float *C_ptr = C + cRow * BM * N + cCol * BN;

    float threadResults[8] = {0.0};

    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A_ptr[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B_ptr[innerRowB * N + innerColB];
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (unsigned int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    for (unsigned int resIdx = 0; resIdx < TM; ++resIdx) {
        C_ptr[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C_ptr[(threadRow * TM + resIdx) * N + threadCol];
    }
}

// Kernel 5: 2D Blocktiling
extern "C" __global__ void sgemm_2D_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    __shared__ float As[512];
    __shared__ float Bs[512];

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    const unsigned int threadCol = threadIdx.x % (BN / TN);
    const unsigned int threadRow = threadIdx.x / (BN / TN);

    const unsigned int innerRowA = threadIdx.x / BK;
    const unsigned int innerColA = threadIdx.x % BK;
    const unsigned int strideA = (BM * BK) / 64;

    const unsigned int innerRowB = threadIdx.x / BN;
    const unsigned int innerColB = threadIdx.x % BN;
    const unsigned int strideB = (BK * BN) / 64;

    const float *A_ptr = A + cRow * BM * K;
    const float *B_ptr = B + cCol * BN;
    float *C_ptr = C + cRow * BM * N + cCol * BN;

    float threadResults[64] = {0.0};
    float regM[8] = {0.0};
    float regN[8] = {0.0};

    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (unsigned int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A_ptr[(innerRowA + loadOffset) * K + innerColA];
        }
        for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B_ptr[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (unsigned int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (unsigned int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C_ptr[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C_ptr[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

// Kernel 6: Vectorized (Simplified - using same logic as Kernel 5)
extern "C" __global__ void sgemm_vectorize(int M, int N, int K, float alpha,
                                const float *A, const float *B,
                                float beta, float *C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    __shared__ float As[1024];  // BM * BK
    __shared__ float Bs[1024];  // BK * BN

    const unsigned int cRow = blockIdx.y;
    const unsigned int cCol = blockIdx.x;

    const unsigned int threadCol = threadIdx.x % (BN / TN);
    const unsigned int threadRow = threadIdx.x / (BN / TN);

    const unsigned int innerRowA = threadIdx.x / BK;
    const unsigned int innerColA = threadIdx.x % BK;
    const unsigned int strideA = (BM * BK) / 256;

    const unsigned int innerRowB = threadIdx.x / BN;
    const unsigned int innerColB = threadIdx.x % BN;
    const unsigned int strideB = (BK * BN) / 256;

    const float *A_ptr = A + cRow * BM * K;
    const float *B_ptr = B + cCol * BN;
    float *C_ptr = C + cRow * BM * N + cCol * BN;

    float threadResults[64] = {0.0};
    float regM[8] = {0.0};
    float regN[8] = {0.0};

    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (unsigned int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A_ptr[(innerRowA + loadOffset) * K + innerColA];
        }
        for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B_ptr[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (unsigned int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (unsigned int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C_ptr[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C_ptr[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}
"""

# Use Modal's official CUDA image with full runtime
image = modal.Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04",
    add_python="3.11"
).pip_install("cupy-cuda12x", "numpy")

@app.function(gpu="A10G", image=image, timeout=600)
def run_matmul_benchmarks():
    import cupy as cp
    import numpy as np
    
    # Compile kernels
    module = cp.RawModule(code=cuda_source)
    
    kernel_naive = module.get_function('sgemm_naive')
    kernel_coalescing = module.get_function('sgemm_coalescing')
    kernel_shared_mem = module.get_function('sgemm_shared_mem_block')
    kernel_1d_tiling = module.get_function('sgemm_1D_blocktiling')
    kernel_2d_tiling = module.get_function('sgemm_2D_blocktiling')
    kernel_vectorize = module.get_function('sgemm_vectorize')
    
    def ceil_div(a, b):
        return (a + b - 1) // b
    
    def benchmark_kernel(kernel, M, N, K, grid, block, kernel_name):
        """Benchmark a kernel and return GFLOPS"""
        alpha = np.float32(1.0)
        beta = np.float32(0.0)
        
        # Create random matrices
        A = cp.random.randn(M, K, dtype=cp.float32)
        B = cp.random.randn(K, N, dtype=cp.float32)
        C = cp.zeros((M, N), dtype=cp.float32)
        
        # Warmup
        kernel(grid, block, (M, N, K, alpha, A, B, beta, C))
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        kernel(grid, block, (M, N, K, alpha, A, B, beta, C))
        end.record()
        end.synchronize()
        
        elapsed_time = cp.cuda.get_elapsed_time(start, end) / 1000.0  # Convert to seconds
        flops = 2.0 * M * N * K
        gflops = (flops / elapsed_time) / 1e9
        
        print(f"{kernel_name}: {gflops:.2f} GFLOPS ({elapsed_time*1000:.3f} ms)")
        
        return gflops
    
    M = N = K = 4096
    
    print(f"Matrix size: {M}x{K} @ {K}x{N}\n")
    print("=" * 60)
    
    # Kernel 1: Naive
    grid = (ceil_div(M, 32), ceil_div(N, 32))
    block = (32, 32)
    benchmark_kernel(kernel_naive, M, N, K, grid, block, "Kernel 1: Naive")
    
    # Kernel 2: Coalescing
    grid = (ceil_div(M, 32), ceil_div(N, 32))
    block = (32 * 32,)
    benchmark_kernel(kernel_coalescing, M, N, K, grid, block, "Kernel 2: Coalescing")
    
    # Kernel 3: Shared Memory
    grid = (ceil_div(N, 32), ceil_div(M, 32))
    block = (32 * 32,)
    benchmark_kernel(kernel_shared_mem, M, N, K, grid, block, "Kernel 3: Shared Memory")
    
    # Kernel 4: 1D Blocktiling
    BM, BN, TM = 64, 64, 8
    grid = (ceil_div(N, BN), ceil_div(M, BM))
    block = ((BM * BN) // TM,)
    benchmark_kernel(kernel_1d_tiling, M, N, K, grid, block, "Kernel 4: 1D Blocktiling")
    
    # Kernel 5: 2D Blocktiling
    BM, BN, TM, TN = 64, 64, 8, 8
    grid = (ceil_div(N, BN), ceil_div(M, BM))
    block = ((BM * BN) // (TM * TN),)
    benchmark_kernel(kernel_2d_tiling, M, N, K, grid, block, "Kernel 5: 2D Blocktiling")
    
    # Kernel 6: Vectorized
    BM, BN, TM, TN = 128, 128, 8, 8
    grid = (ceil_div(N, BN), ceil_div(M, BM))
    block = ((BM * BN) // (TM * TN),)
    benchmark_kernel(kernel_vectorize, M, N, K, grid, block, "Kernel 6: Vectorized")
    
    print("=" * 60)
    print("\nBenchmark complete!")

@app.local_entrypoint()
def main():
    run_matmul_benchmarks.remote()