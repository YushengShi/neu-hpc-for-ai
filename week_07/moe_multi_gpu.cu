// moe_multi_gpu.cu
// Steps 3-10 of checklist: multi-GPU MoE with data + expert parallelism.
//
// Parallelism layout:
//   DATA parallelism   : tokens split evenly across ranks
//   EXPERT parallelism : routed experts split evenly across ranks
//
// NCCL communication:
//   AllToAll  (step 7)  : dispatch packed token vectors to owner GPU
//   AllToAll  (step 9)  : send expert outputs back to origin GPU
//   AllReduce (shared)  : not needed - shared experts run independently per rank
//
// Compile (two GPUs example):
//   nvcc -O2 -std=c++14 moe_multi_gpu.cu -lnccl -o moe_multi_gpu
// Run:
//   mpirun -np 2 ./moe_multi_gpu <tests.txt>
//   -- OR via the Python launcher which sets RANK/WORLD_SIZE env vars --

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "moe_kernels.cuh"
#include "testcase.h"

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------
#define CK(x) do {                                                          \
    cudaError_t _e=(x);                                                     \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",                  \
    __FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}                     \
} while(0)

#define NK(x) do {                                                          \
    ncclResult_t _r=(x);                                                    \
    if(_r!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",                  \
    __FILE__,__LINE__,ncclGetErrorString(_r));exit(1);}                     \
} while(0)

// ---------------------------------------------------------------------------
// Run gated MLP (same as single-GPU version)
// ---------------------------------------------------------------------------
static void run_gated_mlp(
    const float* d_x, int H, int I,
    const float* d_gp, const float* d_up, const float* d_dp,
    float* d_out, float* d_g, float* d_u, float* d_m
) {
    int tb = 128;
    kernel_matvec<<<(I+tb-1)/tb,tb>>>(d_gp, d_x, d_g, I, H);
    kernel_matvec<<<(I+tb-1)/tb,tb>>>(d_up, d_x, d_u, I, H);
    kernel_silu  <<<(I+tb-1)/tb,tb>>>(d_g, I);
    kernel_elemwise_mul<<<(I+tb-1)/tb,tb>>>(d_g, d_u, d_m, I);
    kernel_matvec<<<(H+tb-1)/tb,tb>>>(d_dp, d_m, d_out, H, I);
}

// ---------------------------------------------------------------------------
// Per-rank MoE forward with AllToAll expert dispatch
//
// Each rank owns:
//   - a slice of tokens  [Tlocal, H]              (data parallelism)
//   - a slice of experts [E_local experts]         (expert parallelism)
//
// Protocol (steps 6-10):
//   1. Compute gate scores + topk on local tokens (all experts, not just local)
//   2. Build send_counts[g] = how many (token,slot) pairs go to GPU g
//   3. Pack those token vectors into a contiguous send buffer  [sum, H]
//   4. AllToAll counts  → recv_counts
//   5. AllToAllv        → recv_buf  (tokens arrive from other GPUs)
//   6. Run local experts on received tokens
//   7. AllToAllv        → send back expert outputs in reverse
//   8. Unpack and weighted-sum into routed_out
// ---------------------------------------------------------------------------
static void moe_forward_rank(
    // local token slice
    const float* d_x_local,       // [Tlocal, H]
    int Tlocal, int T_global,
    int H, int SI, int I,
    int E_global, int E_local, int expert_offset,
    int K, float scale,
    int rank, int world_size,
    ncclComm_t comm,
    // weights on this GPU
    const float* d_gate_w,        // [E_global, H]
    const float* d_rgp_local,     // [E_local, I, H]
    const float* d_rup_local,     // [E_local, I, H]
    const float* d_rdp_local,     // [E_local, H, I]
    const float* d_sgp,           // [SI, H]
    const float* d_sup,           // [SI, H]
    const float* d_sdp,           // [H, SI]
    // outputs
    float* d_shared_out,          // [Tlocal, H]
    float* d_routed_out           // [Tlocal, H]  ← filled by this function
) {
    int tb = 128;

    // ------------------------------------------------------------------
    // Step 5: Gate scores + top-K for local tokens
    // ------------------------------------------------------------------
    float* d_scores;
    CK(cudaMalloc(&d_scores, sizeof(float)*Tlocal*E_global));
    {
        dim3 grid(Tlocal, (E_global+tb-1)/tb);
        kernel_gate_scores<<<grid, tb>>>(d_x_local, d_gate_w, d_scores,
                                          Tlocal, H, E_global);
    }
    int*   d_topk_idx; float* d_topk_wt;
    CK(cudaMalloc((void**)&d_topk_idx, sizeof(int)  *Tlocal*K));
    CK(cudaMalloc(&d_topk_wt,          sizeof(float)*Tlocal*K));
    kernel_topk<<<(Tlocal+tb-1)/tb, tb>>>(d_scores, d_topk_idx, d_topk_wt,
                                           Tlocal, E_global, K, scale);
    CK(cudaFree(d_scores));
    CK(cudaDeviceSynchronize());

    // Pull topk to host for packing logic
    int*   h_idx = (int*)  malloc(sizeof(int)  *Tlocal*K);
    float* h_wt  = (float*)malloc(sizeof(float)*Tlocal*K);
    CK(cudaMemcpy(h_idx, d_topk_idx, sizeof(int)  *Tlocal*K, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_wt,  d_topk_wt,  sizeof(float)*Tlocal*K, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_topk_idx)); CK(cudaFree(d_topk_wt));

    // ------------------------------------------------------------------
    // Step 6: Pack tokens by destination GPU
    //
    // For every (token t, slot k):
    //   dest_rank = expert_id / (E_global / world_size)
    //   pack token t's vector into send_buf[dest_rank]
    //
    // We store metadata so we can unpack after receiving outputs back.
    // ------------------------------------------------------------------
    int experts_per_rank = E_global / world_size;

    // send_counts[g]  = number of (token,slot) dispatches going to GPU g
    int* send_counts = (int*)calloc(world_size, sizeof(int));
    // For each (t,k) record the destination rank
    int* dispatch_dest = (int*)malloc(sizeof(int)*Tlocal*K);
    for (int t = 0; t < Tlocal; t++) {
        for (int k = 0; k < K; k++) {
            int e  = h_idx[t*K+k];
            int dr = e / experts_per_rank;
            if (dr >= world_size) dr = world_size - 1; // last rank handles remainder
            dispatch_dest[t*K+k] = dr;
            send_counts[dr]++;
        }
    }

    // Compute send displacements
    int* send_displs = (int*)calloc(world_size, sizeof(int));
    for (int g = 1; g < world_size; g++)
        send_displs[g] = send_displs[g-1] + send_counts[g-1];
    int total_send = send_displs[world_size-1] + send_counts[world_size-1];

    // Pack: for each rank g, consecutively write token vectors + metadata
    // send_buf layout: one float[H] per dispatch, ordered by dest rank
    float* send_buf = (float*)malloc(sizeof(float)*total_send*H);
    // Also store original (t, k, expert_id, weight) for reconstruction
    int*   send_meta_t  = (int*)  malloc(sizeof(int)  *total_send);
    int*   send_meta_k  = (int*)  malloc(sizeof(int)  *total_send);
    int*   send_meta_e  = (int*)  malloc(sizeof(int)  *total_send);
    float* send_meta_w  = (float*)malloc(sizeof(float)*total_send);

    int* cursor = (int*)calloc(world_size, sizeof(int));
    // Pull x_local to host for packing
    float* h_x = (float*)malloc(sizeof(float)*Tlocal*H);
    CK(cudaMemcpy(h_x, d_x_local, sizeof(float)*Tlocal*H, cudaMemcpyDeviceToHost));

    for (int t = 0; t < Tlocal; t++) {
        for (int k = 0; k < K; k++) {
            int e  = h_idx[t*K+k];
            float w = h_wt[t*K+k];
            int dr = dispatch_dest[t*K+k];
            int pos = send_displs[dr] + cursor[dr];
            memcpy(send_buf + pos*H, h_x + t*H, sizeof(float)*H);
            send_meta_t[pos] = t;
            send_meta_k[pos] = k;
            send_meta_e[pos] = e;
            send_meta_w[pos] = w;
            cursor[dr]++;
        }
    }
    free(cursor); free(h_x);

    // ------------------------------------------------------------------
    // Step 7a: AllToAll send_counts → recv_counts
    // (use CPU AllToAll via MPI-like ring since NCCL doesn't have AllToAllv
    //  for integers; we use ncclAllToAll on floats for the data itself)
    //
    // For count exchange we use a simple allgather of the full count matrix.
    // ------------------------------------------------------------------
    // Gather all send_counts so every rank knows recv_counts
    int* all_counts = (int*)malloc(sizeof(int)*world_size*world_size);
    // Upload send_counts to device, allgather, download
    int* d_send_counts; int* d_all_counts;
    CK(cudaMalloc((void**)&d_send_counts, sizeof(int)*world_size));
    CK(cudaMalloc((void**)&d_all_counts,  sizeof(int)*world_size*world_size));
    CK(cudaMemcpy(d_send_counts, send_counts, sizeof(int)*world_size,
                  cudaMemcpyHostToDevice));
    NK(ncclAllGather(d_send_counts, d_all_counts, world_size,
                     ncclInt32, comm, (cudaStream_t)0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(all_counts, d_all_counts,
                  sizeof(int)*world_size*world_size, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_send_counts)); CK(cudaFree(d_all_counts));

    // recv_counts[g] = how many dispatches GPU g sends to me (rank)
    int* recv_counts = (int*)malloc(sizeof(int)*world_size);
    for (int g = 0; g < world_size; g++)
        recv_counts[g] = all_counts[g*world_size + rank];
    free(all_counts);

    int* recv_displs = (int*)calloc(world_size, sizeof(int));
    for (int g = 1; g < world_size; g++)
        recv_displs[g] = recv_displs[g-1] + recv_counts[g-1];
    int total_recv = recv_displs[world_size-1] + recv_counts[world_size-1];

    // ------------------------------------------------------------------
    // Step 7b: AllToAllv — ship token vectors to owner GPUs
    //
    // NCCL's ncclSend/ncclRecv inside a group acts as AllToAllv.
    // ------------------------------------------------------------------
    float* d_send_buf; float* d_recv_buf;
    CK(cudaMalloc(&d_send_buf, sizeof(float)*(total_send > 0 ? total_send : 1)*H));
    CK(cudaMalloc(&d_recv_buf, sizeof(float)*(total_recv > 0 ? total_recv : 1)*H));
    if (total_send > 0)
        CK(cudaMemcpy(d_send_buf, send_buf, sizeof(float)*total_send*H,
                      cudaMemcpyHostToDevice));

    cudaStream_t stream = (cudaStream_t)0;

    NK(ncclGroupStart());
    for (int g = 0; g < world_size; g++) {
        if (send_counts[g] > 0)
            NK(ncclSend(d_send_buf + (long)send_displs[g]*H,
                        (size_t)send_counts[g]*H, ncclFloat, g, comm, stream));
        if (recv_counts[g] > 0)
            NK(ncclRecv(d_recv_buf + (long)recv_displs[g]*H,
                        (size_t)recv_counts[g]*H, ncclFloat, g, comm, stream));
    }
    NK(ncclGroupEnd());
    CK(cudaStreamSynchronize(stream));

    // Also exchange the expert IDs of received tokens
    // (so we know which local expert to run on each received vector)
    // Pack expert IDs as floats for NCCL transport
    float* h_send_eid = (float*)malloc(sizeof(float)*(total_send > 0 ? total_send : 1));
    for (int i = 0; i < total_send; i++)
        h_send_eid[i] = (float)send_meta_e[i];

    float* d_send_eid; float* d_recv_eid;
    CK(cudaMalloc(&d_send_eid, sizeof(float)*(total_send > 0 ? total_send : 1)));
    CK(cudaMalloc(&d_recv_eid, sizeof(float)*(total_recv > 0 ? total_recv : 1)));
    if (total_send > 0)
        CK(cudaMemcpy(d_send_eid, h_send_eid, sizeof(float)*total_send,
                      cudaMemcpyHostToDevice));
    free(h_send_eid);

    NK(ncclGroupStart());
    for (int g = 0; g < world_size; g++) {
        if (send_counts[g] > 0)
            NK(ncclSend(d_send_eid + send_displs[g],
                        (size_t)send_counts[g], ncclFloat, g, comm, stream));
        if (recv_counts[g] > 0)
            NK(ncclRecv(d_recv_eid + recv_displs[g],
                        (size_t)recv_counts[g], ncclFloat, g, comm, stream));
    }
    NK(ncclGroupEnd());
    CK(cudaStreamSynchronize(stream));

    float* h_recv_eid = (float*)malloc(sizeof(float)*(total_recv > 0 ? total_recv : 1));
    if (total_recv > 0)
        CK(cudaMemcpy(h_recv_eid, d_recv_eid, sizeof(float)*total_recv,
                      cudaMemcpyDeviceToHost));
    CK(cudaFree(d_send_eid)); CK(cudaFree(d_recv_eid));

    // ------------------------------------------------------------------
    // Step 8: Run LOCAL experts on received token vectors
    // ------------------------------------------------------------------
    float* h_recv_buf = (float*)malloc(sizeof(float)*(total_recv > 0 ? total_recv : 1)*H);
    if (total_recv > 0)
        CK(cudaMemcpy(h_recv_buf, d_recv_buf, sizeof(float)*total_recv*H,
                      cudaMemcpyDeviceToHost));

    float* h_expert_out = (float*)malloc(sizeof(float)*(total_recv > 0 ? total_recv : 1)*H);
    memset(h_expert_out, 0, sizeof(float)*(total_recv > 0 ? total_recv : 1)*H);

    float *d_tok, *d_eout, *d_eg, *d_eu, *d_em;
    CK(cudaMalloc(&d_tok,  sizeof(float)*H));
    CK(cudaMalloc(&d_eout, sizeof(float)*H));
    CK(cudaMalloc(&d_eg,   sizeof(float)*I));
    CK(cudaMalloc(&d_eu,   sizeof(float)*I));
    CK(cudaMalloc(&d_em,   sizeof(float)*I));

    for (int i = 0; i < total_recv; i++) {
        int global_e = (int)h_recv_eid[i];
        int local_e  = global_e - expert_offset;
        if (local_e < 0 || local_e >= E_local) {
            // shouldn't happen if routing is correct
            fprintf(stderr, "rank=%d got token for expert %d (local %d) — skip\n",
                    rank, global_e, local_e);
            continue;
        }
        CK(cudaMemcpy(d_tok, h_recv_buf + i*H, sizeof(float)*H, cudaMemcpyHostToDevice));

        const float* gp = d_rgp_local + (long)local_e*I*H;
        const float* up = d_rup_local + (long)local_e*I*H;
        const float* dp = d_rdp_local + (long)local_e*H*I;
        run_gated_mlp(d_tok, H, I, gp, up, dp, d_eout, d_eg, d_eu, d_em);
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_expert_out + i*H, d_eout, sizeof(float)*H,
                      cudaMemcpyDeviceToHost));
    }
    CK(cudaFree(d_tok)); CK(cudaFree(d_eout));
    CK(cudaFree(d_eg));  CK(cudaFree(d_eu)); CK(cudaFree(d_em));
    free(h_recv_buf); free(h_recv_eid);

    // ------------------------------------------------------------------
    // Step 9: Send expert outputs BACK to originating GPUs
    // AllToAllv in reverse: recv_counts/displs ↔ send_counts/displs
    // ------------------------------------------------------------------
    float* d_expert_out;
    CK(cudaMalloc(&d_expert_out, sizeof(float)*(total_recv > 0 ? total_recv : 1)*H));
    if (total_recv > 0)
        CK(cudaMemcpy(d_expert_out, h_expert_out, sizeof(float)*total_recv*H,
                      cudaMemcpyHostToDevice));
    free(h_expert_out);

    // Results go back: what was recv now becomes send, and vice versa
    float* d_result_buf;
    CK(cudaMalloc(&d_result_buf, sizeof(float)*(total_send > 0 ? total_send : 1)*H));

    NK(ncclGroupStart());
    for (int g = 0; g < world_size; g++) {
        // send back to g what we received from g
        if (recv_counts[g] > 0)
            NK(ncclSend(d_expert_out + (long)recv_displs[g]*H,
                        (size_t)recv_counts[g]*H, ncclFloat, g, comm, stream));
        // receive from g what we originally sent to g
        if (send_counts[g] > 0)
            NK(ncclRecv(d_result_buf + (long)send_displs[g]*H,
                        (size_t)send_counts[g]*H, ncclFloat, g, comm, stream));
    }
    NK(ncclGroupEnd());
    CK(cudaStreamSynchronize(stream));

    CK(cudaFree(d_expert_out));
    CK(cudaFree(d_recv_buf));
    CK(cudaFree(d_send_buf));

    float* h_result_buf = (float*)malloc(sizeof(float)*(total_send > 0 ? total_send : 1)*H);
    if (total_send > 0)
        CK(cudaMemcpy(h_result_buf, d_result_buf, sizeof(float)*total_send*H,
                      cudaMemcpyDeviceToHost));
    CK(cudaFree(d_result_buf));

    // ------------------------------------------------------------------
    // Step 10: Combine outputs in original token order
    //
    // Iterate over the send metadata (which maps result positions back
    // to original (token, weight)) and accumulate weighted sums.
    // ------------------------------------------------------------------
    float* h_routed = (float*)calloc(Tlocal*H, sizeof(float));
    for (int pos = 0; pos < total_send; pos++) {
        int   t = send_meta_t[pos];
        float w = send_meta_w[pos];
        for (int h = 0; h < H; h++)
            h_routed[t*H+h] += w * h_result_buf[pos*H+h];
    }
    free(h_result_buf);

    CK(cudaMemcpy(d_routed_out, h_routed, sizeof(float)*Tlocal*H,
                  cudaMemcpyHostToDevice));
    free(h_routed);

    // ------------------------------------------------------------------
    // Shared experts: run independently on every rank (no communication)
    // ------------------------------------------------------------------
    float *d_sg, *d_su, *d_sm;
    CK(cudaMalloc(&d_sg, sizeof(float)*SI));
    CK(cudaMalloc(&d_su, sizeof(float)*SI));
    CK(cudaMalloc(&d_sm, sizeof(float)*SI));
    for (int t = 0; t < Tlocal; t++) {
        run_gated_mlp(d_x_local + (long)t*H, H, SI,
                      d_sgp, d_sup, d_sdp,
                      d_shared_out + (long)t*H,
                      d_sg, d_su, d_sm);
    }
    CK(cudaFree(d_sg)); CK(cudaFree(d_su)); CK(cudaFree(d_sm));

    // Cleanup
    free(h_idx); free(h_wt); free(dispatch_dest);
    free(send_counts); free(recv_counts);
    free(send_displs); free(recv_displs);
    free(send_buf);
    free(send_meta_t); free(send_meta_k);
    free(send_meta_e); free(send_meta_w);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tests.txt>\n", argv[0]);
        return 1;
    }

    // Rank / world_size come from env vars set by the Python launcher
    int rank       = 0;
    int world_size = 1;
    const char* r = getenv("RANK");
    const char* w = getenv("WORLD_SIZE");
    if (r) rank       = atoi(r);
    if (w) world_size = atoi(w);

    // Each rank owns one GPU
    CK(cudaSetDevice(rank));

    // ------------------------------------------------------------------
    // Init NCCL: rank 0 creates the unique id and broadcasts it.
    // We broadcast via a shared file (simple, no MPI dependency).
    // ------------------------------------------------------------------
    ncclUniqueId nccl_id;
    const char* id_file = "/tmp/nccl_unique_id.bin";

    if (rank == 0) {
        NK(ncclGetUniqueId(&nccl_id));
        FILE* f = fopen(id_file, "wb");
        fwrite(&nccl_id, sizeof(nccl_id), 1, f);
        fclose(f);
    } else {
        // Spin-wait until rank 0 writes the file
        FILE* f = NULL;
        for (int tries = 0; tries < 200 && !f; tries++) {
            f = fopen(id_file, "rb");
            if (!f) { struct timespec ts={0,50000000}; nanosleep(&ts,NULL); }
        }
        if (!f) { fprintf(stderr, "rank=%d: timeout waiting for NCCL id\n", rank); return 1; }
        fread(&nccl_id, sizeof(nccl_id), 1, f);
        fclose(f);
    }

    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, world_size, nccl_id, rank));

    // ------------------------------------------------------------------
    // Load test cases (every rank loads the same file)
    // ------------------------------------------------------------------
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }
    int num_cases = 0;
    fscanf(f, "%d", &num_cases);

    const float atol = 2e-4f;
    int all_ok = 1;

    for (int cid = 0; cid < num_cases; cid++) {
        TestCase tc;
        memset(&tc, 0, sizeof(tc));
        if (!tc_load(f, &tc)) {
            fprintf(stderr, "rank=%d failed to load case %d\n", rank, cid);
            break;
        }

        int T=tc.num_tokens, H=tc.hidden_size, I=tc.moe_intermediate_size;
        int E=tc.n_routed_experts, S=tc.n_shared_experts, K=tc.top_k;
        int SI = S * I;

        // ---- Data parallelism: token split ----
        int base_T  = T / world_size;
        int extra_T = T % world_size;
        int Tlocal  = base_T + (rank < extra_T ? 1 : 0);
        int tok_off = rank * base_T + (rank < extra_T ? rank : extra_T);

        // ---- Expert parallelism: expert split ----
        int base_E  = E / world_size;
        int extra_E = E % world_size;
        int E_local = base_E + (rank < extra_E ? 1 : 0);
        int exp_off = rank * base_E + (rank < extra_E ? rank : extra_E);

        if (rank == 0)
            printf("case=%d T=%d H=%d E=%d K=%d world=%d\n",
                   cid, T, H, E, K, world_size);

        // Upload local token slice + full gate weights + local expert weights
        float *d_x, *d_gw, *d_rgp, *d_rup, *d_rdp, *d_sgp, *d_sup, *d_sdp;
        CK(cudaMalloc(&d_x,   sizeof(float)*Tlocal*H));
        CK(cudaMalloc(&d_gw,  sizeof(float)*E*H));
        CK(cudaMalloc(&d_rgp, sizeof(float)*E_local*I*H));
        CK(cudaMalloc(&d_rup, sizeof(float)*E_local*I*H));
        CK(cudaMalloc(&d_rdp, sizeof(float)*E_local*H*I));
        CK(cudaMalloc(&d_sgp, sizeof(float)*SI*H));
        CK(cudaMalloc(&d_sup, sizeof(float)*SI*H));
        CK(cudaMalloc(&d_sdp, sizeof(float)*H*SI));

        CK(cudaMemcpy(d_x,   tc.x + tok_off*H,  sizeof(float)*Tlocal*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gw,  tc.gate_weight,     sizeof(float)*E*H,      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rgp, tc.routed_gate_proj + exp_off*I*H,  sizeof(float)*E_local*I*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rup, tc.routed_up_proj   + exp_off*I*H,  sizeof(float)*E_local*I*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rdp, tc.routed_down_proj + exp_off*H*I,  sizeof(float)*E_local*H*I, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sgp, tc.shared_gate_proj, sizeof(float)*SI*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sup, tc.shared_up_proj,   sizeof(float)*SI*H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sdp, tc.shared_down_proj, sizeof(float)*H*SI, cudaMemcpyHostToDevice));

        float *d_shared_out, *d_routed_out;
        CK(cudaMalloc(&d_shared_out, sizeof(float)*Tlocal*H));
        CK(cudaMalloc(&d_routed_out, sizeof(float)*Tlocal*H));

        // Run multi-GPU MoE forward
        moe_forward_rank(
            d_x, Tlocal, T, H, SI, I,
            E, E_local, exp_off, K, tc.routed_scaling_factor,
            rank, world_size, comm,
            d_gw, d_rgp, d_rup, d_rdp, d_sgp, d_sup, d_sdp,
            d_shared_out, d_routed_out
        );
        CK(cudaDeviceSynchronize());

        // Compute final = shared + routed
        float* d_final;
        CK(cudaMalloc(&d_final, sizeof(float)*Tlocal*H));
        int tb = 128;
        kernel_add<<<(Tlocal*H+tb-1)/tb, tb>>>(d_shared_out, d_routed_out, d_final, Tlocal*H);
        CK(cudaDeviceSynchronize());

        // Copy back
        float* got_shared = (float*)malloc(sizeof(float)*Tlocal*H);
        float* got_routed = (float*)malloc(sizeof(float)*Tlocal*H);
        float* got_final  = (float*)malloc(sizeof(float)*Tlocal*H);
        CK(cudaMemcpy(got_shared, d_shared_out, sizeof(float)*Tlocal*H, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_routed, d_routed_out, sizeof(float)*Tlocal*H, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_final,  d_final,      sizeof(float)*Tlocal*H, cudaMemcpyDeviceToHost));

        // Verify against expected slice
        float s_err = tc_max_abs_diff(got_shared, tc.expected_shared_out + tok_off*H, (long)Tlocal*H);
        float r_err = tc_max_abs_diff(got_routed, tc.expected_routed_out + tok_off*H, (long)Tlocal*H);
        float f_err = tc_max_abs_diff(got_final,  tc.expected_final_out  + tok_off*H, (long)Tlocal*H);
        int ok = (s_err<=atol) && (r_err<=atol) && (f_err<=atol);
        if (!ok) all_ok = 0;

        printf("rank=%d case=%d s_err=%.6f r_err=%.6f f_err=%.6f [%s]\n",
               rank, cid, s_err, r_err, f_err, ok ? "PASS" : "FAIL");

        // Cleanup
        CK(cudaFree(d_x)); CK(cudaFree(d_gw));
        CK(cudaFree(d_rgp)); CK(cudaFree(d_rup)); CK(cudaFree(d_rdp));
        CK(cudaFree(d_sgp)); CK(cudaFree(d_sup)); CK(cudaFree(d_sdp));
        CK(cudaFree(d_shared_out)); CK(cudaFree(d_routed_out)); CK(cudaFree(d_final));
        free(got_shared); free(got_routed); free(got_final);
        tc_free(&tc);
    }

    fclose(f);
    ncclCommDestroy(comm);
    printf("rank=%d overall: %s\n", rank, all_ok ? "ALL PASS" : "SOME FAILURES");
    return all_ok ? 0 : 2;
}
