# modal_moe_bench.py
# Week 9: DeepSeekMoE — cuBLAS GemmEx (tensor cores) vs naive, on B200
# Run with: modal run modal_moe_bench.py

import modal
import base64 as _b64

CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Dims — kept small enough to fit in Modal's 64 GB CPU RAM + 192 GB GPU HBM.
// E=16: weights = 3*16*18432*7168*2 = 12.7 GB GPU, 0 GB CPU (GPU-init only)
// E=256 needs 203 GB GPU — exceeds B200 HBM; real DeepSeek V3 shards across GPUs.
// ---------------------------------------------------------------------------
constexpr int HIDDEN       = 7168;
constexpr int FFN_INTER    = 18432;
constexpr int NUM_EXPERTS  = 16;
constexpr int TOP_K        = 8;
constexpr int MAX_TOKENS   = 4096;
constexpr int NAIVE_TOKENS = 32;

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s=(x); \
    if(_s!=CUBLAS_STATUS_SUCCESS){ \
        printf("cuBLAS error %d line %d\n",(int)_s,__LINE__);exit(1);} \
} while(0)
#define CURAND_CHECK(x) do { \
    curandStatus_t _s=(x); \
    if(_s!=CURAND_STATUS_SUCCESS){ \
        printf("cuRAND error %d line %d\n",(int)_s,__LINE__);exit(1);} \
} while(0)
#define CUDA_CHECK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ \
        printf("CUDA error %s line %d\n",cudaGetErrorString(_e),__LINE__);exit(1);} \
} while(0)

// ---------------------------------------------------------------------------
// GPU-side random init: fill bf16 buffer using cuRAND float + cast kernel
// Avoids allocating large host vectors (which OOM the CPU container)
// ---------------------------------------------------------------------------
__global__ void float_to_bf16_kernel(const float* src, bf16* dst, int n, float scale, float bias) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) dst[i] = __float2bfloat16(src[i]*scale+bias);
}

// Forward declaration
void progress(int step, int total, const char* label);

void gpu_rand_bf16(curandGenerator_t gen, bf16* dst, size_t n, float scale, float bias) {
    float* tmp; CUDA_CHECK(cudaMalloc(&tmp, n*sizeof(float)));
    CURAND_CHECK(curandGenerateUniform(gen, tmp, n));
    float_to_bf16_kernel<<<(n+255)/256,256>>>(tmp,dst,(int)n,scale,bias);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tmp);
}

void gpu_rand_float(curandGenerator_t gen, float* dst, size_t n) {
    CURAND_CHECK(curandGenerateUniform(gen, dst, n));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
__global__ void router_kernel(
    const float* __restrict__ logits,
    int* __restrict__ expert_ids, float* __restrict__ scores,
    int T, int E, int K
) {
    int t = blockIdx.x*blockDim.x+threadIdx.x;
    if (t>=T) return;
    const float* row = logits+t*E;
    int   top_idx[8]; float top_val[8];
    for(int k=0;k<K;++k){top_idx[k]=k;top_val[k]=row[k];}
    for(int e=K;e<E;++e){
        float v=row[e]; int mk=0; float mv=top_val[0];
        for(int k=1;k<K;++k) if(top_val[k]<mv){mv=top_val[k];mk=k;}
        if(v>mv){top_val[mk]=v;top_idx[mk]=e;}
    }
    float sum=0.f;
    for(int k=0;k<K;++k) sum+=expf(top_val[k]);
    for(int k=0;k<K;++k){
        expert_ids[t*K+k]=top_idx[k];
        scores[t*K+k]=expf(top_val[k])/(sum+1e-12f);
    }
}

// ---------------------------------------------------------------------------
// Gather / Scatter / SiLU
// ---------------------------------------------------------------------------
__global__ void gather_kernel(
    const bf16* __restrict__ X, const int* __restrict__ tok_idx,
    bf16* __restrict__ out, int H, int n
) {
    int i=blockIdx.x; if(i>=n) return;
    int tok=tok_idx[i];
    for(int h=threadIdx.x;h<H;h+=blockDim.x)
        out[(size_t)i*H+h]=X[(size_t)tok*H+h];
}

__global__ void silu_gate_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    bf16* __restrict__ mid, int n
) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float g=gate[i],u=up[i];
    mid[i]=__float2bfloat16((g/(1.f+expf(-g)))*u);
}

__global__ void scatter_add_kernel(
    const float* __restrict__ out, const int* __restrict__ tok_idx,
    const float* __restrict__ scores_flat,
    float* __restrict__ Y, int H, int n
) {
    int i=blockIdx.x; if(i>=n) return;
    int   tok  =tok_idx[i];
    float score=scores_flat[i];
    for(int h=threadIdx.x;h<H;h+=blockDim.x)
        atomicAdd(&Y[(size_t)tok*H+h], out[(size_t)i*H+h]*score);
}

// ---------------------------------------------------------------------------
// Naive baseline
// ---------------------------------------------------------------------------
__global__ void moe_naive_kernel(
    const bf16* __restrict__ X, const int* __restrict__ expert_ids,
    const float* __restrict__ scores,
    const bf16* __restrict__ W1, const bf16* __restrict__ W2,
    const bf16* __restrict__ W3, float* __restrict__ Y,
    int T, int H, int I, int K
) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=T) return;
    for(int k=0;k<K;++k){
        int   e =expert_ids[t*K+k];
        float sc=scores    [t*K+k];
        for(int n=0;n<I;++n){
            float gate=0.f,up=0.f;
            for(int h=0;h<H;++h){
                float xv=__bfloat162float(X[t*H+h]);
                gate+=xv*__bfloat162float(W1[(size_t)e*I*H+n*H+h]);
                up  +=xv*__bfloat162float(W2[(size_t)e*I*H+n*H+h]);
            }
            float hv=(gate/(1.f+expf(-gate)))*up;
            for(int o=0;o<H;++o)
                atomicAdd(&Y[t*H+o],
                    sc*hv*__bfloat162float(W3[(size_t)e*H*I+o*I+n]));
        }
    }
}

// ---------------------------------------------------------------------------
// Routing table (device->host->device, small arrays only)
// ---------------------------------------------------------------------------
void build_routing(
    int* d_eids,int* d_offs,int* d_tidx,
    float* d_scores,float* d_scores_flat,
    int T,int K,int E,std::vector<int>& h_offs
){
    std::vector<int>   he(T*K); std::vector<float> hs(T*K);
    cudaMemcpy(he.data(),d_eids,  T*K*sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(),d_scores,T*K*sizeof(float),cudaMemcpyDeviceToHost);
    std::vector<int> cnt(E,0);
    for(int id:he) cnt[id]++;
    h_offs.resize(E+1,0);
    for(int e=0;e<E;++e) h_offs[e+1]=h_offs[e]+cnt[e];
    std::vector<int>   idx(T*K);
    std::vector<float> sflt(T*K);
    std::vector<int>   cur(h_offs.begin(),h_offs.end());
    for(int t=0;t<T;++t)
        for(int k=0;k<K;++k){
            int e=he[t*K+k],pos=cur[e]++;
            idx[pos]=t; sflt[pos]=hs[t*K+k];
        }
    cudaMemcpy(d_offs, h_offs.data(),(E+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_tidx, idx.data(),   T*K *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores_flat,sflt.data(),T*K*sizeof(float),cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// cuBLAS MoE forward (expert-serial, shared per-expert scratch)
// ---------------------------------------------------------------------------
void cublas_moe_fwd(
    cublasHandle_t handle,
    const bf16* d_X,const int* d_tidx,const float* d_scf,
    const bf16* d_W1,const bf16* d_W2,const bf16* d_W3,
    float* d_Y,
    bf16* d_Xe,float* d_gate,float* d_up,bf16* d_mid,float* d_out,
    const std::vector<int>& h_offs,
    int T,int H,int I,int K,int E,bool show_progress=false
){
    const float al=1.f,be=0.f;
    for(int e=0;e<E;++e){
        int n=h_offs[e+1]-h_offs[e]; if(!n) continue;
        const int*   tidx=d_tidx+h_offs[e];
        const float* sflt=d_scf +h_offs[e];
        const bf16* Wg=d_W1+(size_t)e*I*H;
        const bf16* Wu=d_W2+(size_t)e*I*H;
        const bf16* Wd=d_W3+(size_t)e*H*I;
        gather_kernel<<<n,min(H,256)>>>(d_X,tidx,d_Xe,H,n);
        CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,I,n,H,&al,
            Wg,CUDA_R_16BF,H,d_Xe,CUDA_R_16BF,H,&be,d_gate,CUDA_R_32F,I,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,I,n,H,&al,
            Wu,CUDA_R_16BF,H,d_Xe,CUDA_R_16BF,H,&be,d_up,CUDA_R_32F,I,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        silu_gate_kernel<<<(n*I+255)/256,256>>>(d_gate,d_up,d_mid,n*I);
        CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,H,n,I,&al,
            Wd,CUDA_R_16BF,I,d_mid,CUDA_R_16BF,I,&be,d_out,CUDA_R_32F,H,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        scatter_add_kernel<<<n,min(H,256)>>>(d_out,tidx,sflt,d_Y,H,n);
        if(show_progress) progress(e+1, E, "running experts");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
struct BR{double ms,tflops;};
BR bench(const char* lbl,std::function<void()> fn,int wu,int iters,double flops=0){
    printf("  %s\n",lbl); fflush(stdout);
    int total = wu + iters;
    for(int i=0;i<wu;++i){
        progress(i, total, "warmup");
        fn(); cudaDeviceSynchronize();
    }
    cudaEvent_t t0,t1; cudaEventCreate(&t0);cudaEventCreate(&t1);
    std::vector<float> ts;
    for(int i=0;i<iters;++i){
        progress(wu+i, total, "benchmarking");
        cudaEventRecord(t0);fn();cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);ts.push_back(ms);
    }
    progress(total, total, "done");
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    double mean=std::accumulate(ts.begin(),ts.end(),0.)/iters;
    double tf=flops>0?flops/(mean*1e-3)/1e12:0;
    if(flops>0) printf("  -> %.2f ms   %.3f TFLOPs\n",mean,tf);
    else        printf("  -> %.2f ms\n",mean);
    fflush(stdout);
    return {mean,tf};
}

// ---------------------------------------------------------------------------
// Progress bar: prints  [=====>    ] 55%  label
// Call with step=0..total; step==total prints newline
// ---------------------------------------------------------------------------
void progress(int step, int total, const char* label) {
    int width = 30;
    int filled = (step * width) / total;
    int pct    = (step * 100) / total;
    printf("\r  [");
    for (int i=0;i<width;++i) printf(i<filled?"=": (i==filled?">": " "));
    printf("] %3d%%  %s", pct, label);
    if (step==total) printf("\n");
    fflush(stdout);
}

// ===========================================================================
// CORRECTNESS TEST  (small dims, CPU-safe init)
// ===========================================================================
bool correctness_test(){
    printf("\n=== CORRECTNESS TEST ===\n");
    // Small enough that host vectors are fine (~8 MB each)
    const int T=32,H=64,I=128,K=2,E=4;
    printf("  dims: T=%d H=%d I=%d E=%d K=%d\n",T,H,I,E,K);

    bf16 *dX,*dW1,*dW2,*dW3,*dXe,*dMid;
    float *dYc,*dYn,*dLg,*dSc,*dScf,*dGate,*dUp,*dOut;
    int *dEids,*dOffs,*dTidx;
    int mtp=T*K;

    CUDA_CHECK(cudaMalloc(&dX,  (size_t)T*H*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW1, (size_t)E*I*H*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW2, (size_t)E*I*H*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW3, (size_t)E*H*I*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dYc, (size_t)T*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dYn, (size_t)T*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLg, (size_t)T*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t)T*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dScf,(size_t)T*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dEids,(size_t)T*K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOffs,(E+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dTidx,(size_t)T*K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dXe,  (size_t)mtp*H*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dGate,(size_t)mtp*I*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dUp,  (size_t)mtp*I*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMid, (size_t)mtp*I*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)mtp*H*sizeof(float)));

    // Small arrays — host init is fine
    srand(42);
    auto rb=[](){return __float2bfloat16((float)rand()/RAND_MAX*.2f-.1f);};
    auto rf=[](){return (float)rand()/RAND_MAX;};
    std::vector<bf16> hX(T*H),hW1(E*I*H),hW2(E*I*H),hW3(E*H*I);
    std::vector<float> hLg(T*E);
    for(auto&v:hX)  v=rb(); for(auto&v:hW1)v=rb();
    for(auto&v:hW2) v=rb(); for(auto&v:hW3)v=rb();
    for(auto&v:hLg) v=rf();
    cudaMemcpy(dX, hX.data(), hX.size()*sizeof(bf16),  cudaMemcpyHostToDevice);
    cudaMemcpy(dW1,hW1.data(),hW1.size()*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2.data(),hW2.size()*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(dW3,hW3.data(),hW3.size()*sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(dLg,hLg.data(),hLg.size()*sizeof(float),cudaMemcpyHostToDevice);

    router_kernel<<<(T+255)/256,256>>>(dLg,dEids,dSc,T,E,K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_offs;
    build_routing(dEids,dOffs,dTidx,dSc,dScf,T,K,E,h_offs);

    cublasHandle_t h; cublasCreate(&h);

    // cuBLAS run
    CUDA_CHECK(cudaMemset(dYc,0,(size_t)T*H*sizeof(float)));
    cublas_moe_fwd(h,dX,dTidx,dScf,dW1,dW2,dW3,dYc,
                   dXe,dGate,dUp,dMid,dOut,h_offs,T,H,I,K,E);

    // Naive run
    CUDA_CHECK(cudaMemset(dYn,0,(size_t)T*H*sizeof(float)));
    moe_naive_kernel<<<(T+255)/256,256>>>(dX,dEids,dSc,dW1,dW2,dW3,dYn,T,H,I,K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hYc(T*H),hYn(T*H);
    cudaMemcpy(hYc.data(),dYc,T*H*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(hYn.data(),dYn,T*H*sizeof(float),cudaMemcpyDeviceToHost);

    double maxe=0,meane=0; int nz=0;
    for(int i=0;i<T*H;++i){
        double ref=fabs((double)hYn[i]);
        if(ref<1e-6) continue;
        double rel=fabs((double)hYc[i]-(double)hYn[i])/ref;
        if(rel>maxe)maxe=rel; meane+=rel; nz++;
    }
    meane/=std::max(1,nz);
    bool ok=maxe<0.05;

    printf("  max rel error : %.5f\n",maxe);
    printf("  mean rel error: %.5f  (%d non-zero)\n",meane,nz);
    printf("  spot check tok=0: cuBLAS[%.4f %.4f]  naive[%.4f %.4f]\n",
           hYc[0],hYc[1],hYn[0],hYn[1]);
    printf("  TEST %s\n",ok?"PASSED ✓":"FAILED ✗");

    cublasDestroy(h);
    cudaFree(dX);cudaFree(dW1);cudaFree(dW2);cudaFree(dW3);
    cudaFree(dYc);cudaFree(dYn);cudaFree(dLg);cudaFree(dSc);cudaFree(dScf);
    cudaFree(dEids);cudaFree(dOffs);cudaFree(dTidx);
    cudaFree(dXe);cudaFree(dGate);cudaFree(dUp);cudaFree(dMid);cudaFree(dOut);
    return ok;
}

// ===========================================================================
// PERF BENCHMARK  (full dims, GPU-only init via cuRAND)
// ===========================================================================
void perf_benchmark(){
    printf("\n=== PERF BENCHMARK ===\n");
    printf("  H=%d I=%d E=%d K=%d T=%d\n",
           HIDDEN,FFN_INTER,NUM_EXPERTS,TOP_K,MAX_TOKENS);
    double wt_gb=3.*(double)NUM_EXPERTS*FFN_INTER*HIDDEN*2/1e9;
    printf("  Weights: %.1f GB on GPU (init via cuRAND, no host alloc)\n\n",wt_gb);
    fflush(stdout);

    // All weights initialized directly on GPU — no host vectors
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng,CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng,42));

    int mtp=(MAX_TOKENS*TOP_K+NUM_EXPERTS-1)/NUM_EXPERTS+64;

    bf16 *dX,*dW1,*dW2,*dW3,*dXe,*dMid;
    float *dY,*dLg,*dSc,*dScf,*dGate,*dUp,*dOut;
    int *dEids,*dOffs,*dTidx;

    printf("  Allocating GPU memory...\n"); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&dX,  (size_t)MAX_TOKENS*HIDDEN*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW1, (size_t)NUM_EXPERTS*FFN_INTER*HIDDEN*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW2, (size_t)NUM_EXPERTS*FFN_INTER*HIDDEN*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dW3, (size_t)NUM_EXPERTS*HIDDEN*FFN_INTER*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dY,  (size_t)MAX_TOKENS*HIDDEN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLg, (size_t)MAX_TOKENS*NUM_EXPERTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t)MAX_TOKENS*TOP_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dScf,(size_t)MAX_TOKENS*TOP_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dEids,(size_t)MAX_TOKENS*TOP_K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOffs,(NUM_EXPERTS+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dTidx,(size_t)MAX_TOKENS*TOP_K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dXe,  (size_t)mtp*HIDDEN*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dGate,(size_t)mtp*FFN_INTER*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dUp,  (size_t)mtp*FFN_INTER*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMid, (size_t)mtp*FFN_INTER*sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)mtp*HIDDEN*sizeof(float)));
    printf("  Allocation done.\n"); fflush(stdout);

    printf("  Initializing weights on GPU (cuRAND):\n"); fflush(stdout);
    progress(0,5,"X");   gpu_rand_bf16(rng,dX, (size_t)MAX_TOKENS*HIDDEN,         1.f,-.5f);
    progress(1,5,"W1");  gpu_rand_bf16(rng,dW1,(size_t)NUM_EXPERTS*FFN_INTER*HIDDEN,.02f,-.01f);
    progress(2,5,"W2");  gpu_rand_bf16(rng,dW2,(size_t)NUM_EXPERTS*FFN_INTER*HIDDEN,.02f,-.01f);
    progress(3,5,"W3");  gpu_rand_bf16(rng,dW3,(size_t)NUM_EXPERTS*HIDDEN*FFN_INTER,.02f,-.01f);
    progress(4,5,"logits"); gpu_rand_float(rng,dLg,(size_t)MAX_TOKENS*NUM_EXPERTS);
    progress(5,5,"done");

    router_kernel<<<(MAX_TOKENS+255)/256,256>>>(
        dLg,dEids,dSc,MAX_TOKENS,NUM_EXPERTS,TOP_K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h_offs;
    build_routing(dEids,dOffs,dTidx,dSc,dScf,
                  MAX_TOKENS,TOP_K,NUM_EXPERTS,h_offs);

    cublasHandle_t handle; cublasCreate(&handle);

    double flops=2.*2.*MAX_TOKENS*TOP_K*(double)HIDDEN*FFN_INTER;

    auto run_cublas=[&](){
        CUDA_CHECK(cudaMemset(dY,0,(size_t)MAX_TOKENS*HIDDEN*sizeof(float)));
        cublas_moe_fwd(handle,dX,dTidx,dScf,dW1,dW2,dW3,dY,
                       dXe,dGate,dUp,dMid,dOut,h_offs,
                       MAX_TOKENS,HIDDEN,FFN_INTER,TOP_K,NUM_EXPERTS,true);
    };
    auto run_naive=[&](){
        CUDA_CHECK(cudaMemset(dY,0,(size_t)MAX_TOKENS*HIDDEN*sizeof(float)));
        moe_naive_kernel<<<(NAIVE_TOKENS+255)/256,256>>>(
            dX,dEids,dSc,dW1,dW2,dW3,dY,
            NAIVE_TOKENS,HIDDEN,FFN_INTER,TOP_K);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    BR rc=bench("cuBLAS GemmEx (tensor cores, T=4096)",run_cublas,1,5,flops);
    BR rn=bench("Naive scalar baseline      (T=32)",   run_naive, 0,1);

    double extrap=rn.ms*((double)MAX_TOKENS/NAIVE_TOKENS);
    printf("\n=== SUMMARY ===\n");
    printf("cuBLAS tensor cores : %8.2f ms   %.3f TFLOPs  (T=%d E=%d)\n",
           rc.ms,rc.tflops,MAX_TOKENS,NUM_EXPERTS);
    printf("Naive scalar        : %8.2f ms              (T=%d)\n",
           rn.ms,NAIVE_TOKENS);
    printf("Naive extrap        : %8.2f ms   (est. T=%d)\n",extrap,MAX_TOKENS);
    printf("Speedup             : %.1fx\n",extrap/rc.ms);
    printf("B200 bf16 peak      : ~2000 TFLOPs\n");
    printf("Roofline util       : %.1f%%\n",100.*rc.tflops/2000.);
    printf("\nNote: E=16 used (E=256 needs 203 GB > B200's 192 GB HBM;\n");
    printf("      real DeepSeek V3 shards across 32 GPUs in production)\n");

    curandDestroyGenerator(rng);
    cublasDestroy(handle);
    cudaFree(dX);cudaFree(dW1);cudaFree(dW2);cudaFree(dW3);
    cudaFree(dY);cudaFree(dLg);cudaFree(dSc);cudaFree(dScf);
    cudaFree(dEids);cudaFree(dOffs);cudaFree(dTidx);
    cudaFree(dXe);cudaFree(dGate);cudaFree(dUp);cudaFree(dMid);cudaFree(dOut);
}

// ===========================================================================
int main(){
    printf("=== DeepSeek MoE cuBLAS Benchmark [B200] ===\n"); fflush(stdout);
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
    printf("GPU: %s  SM%d.%d  %dSMs  %.0fGB HBM\n\n",
           p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem/1e9);
    fflush(stdout);
    bool ok=correctness_test();
    perf_benchmark();
    return ok?0:1;
}
"""

_CUDA_SRC_B64 = _b64.b64encode(CUDA_SRC.encode()).decode()

def _write_and_compile():
    import base64, os, pathlib, subprocess, time
    src = base64.b64decode(os.environ["CUDA_SRC_B64"]).decode()
    pathlib.Path("/workspace").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/workspace/moe_bench.cu").write_text(src)
    print("Compiling...", flush=True)
    t0 = time.time()
    r = subprocess.run([
        "nvcc", "-O2",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-lcublas", "-lcuda", "-lcurand",
        "/workspace/moe_bench.cu",
        "-o", "/workspace/moe_bench",
    ], capture_output=True, text=True)
    print(f"nvcc: {time.time()-t0:.1f}s", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stderr}")
    print("OK", flush=True)

cuda_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "cmake", "ninja-build")
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/HazyResearch/ThunderKittens.git /opt/thunderkittens",
    )
    .env({"CUDA_SRC_B64": _CUDA_SRC_B64})
    .run_function(_write_and_compile)
)

app = modal.App("deepseek-moe-cublas-b200")

@app.function(gpu="B200", image=cuda_image, timeout=300, memory=32768)
def run_bench() -> str:
    import subprocess, sys
    r = subprocess.run(
        ["/workspace/moe_bench"], capture_output=True, text=True, timeout=240)
    if r.returncode not in (0,1):
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr, file=sys.stderr)
        raise RuntimeError(f"crashed (exit {r.returncode})")
    return r.stdout

@app.function(gpu="B200", image=cuda_image, timeout=60)
def check_gpu_info() -> str:
    import subprocess
    r = subprocess.run(
        ["nvidia-smi","--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    return r.stdout.strip()

@app.local_entrypoint()
def main():
    print("=== GPU Info ===")
    print(check_gpu_info.remote())
    print()
    print("=== Benchmark ===")
    print(run_bench.remote())
