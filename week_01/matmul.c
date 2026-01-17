#include "matmul.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static inline size_t idx2(int row, int col, int stride) {
    return (size_t)row * (size_t)stride + (size_t)col;
}

void matmul_serial(int m, int n, int p,
                   const double *A, const double *B, double *C) {
    // C[m x p] = A[m x n] * B[n x p]
    // Row-major, flat arrays.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[idx2(i, k, n)] * B[idx2(k, j, p)];
            }
            C[idx2(i, j, p)] = sum;
        }
    }
}

typedef struct {
    int m, n, p;
    const double *A;
    const double *B;
    double *C;
    int row_start;   // inclusive
    int row_end;     // exclusive
} worker_args_t;

static void *worker_rows(void *arg) {
    worker_args_t *w = (worker_args_t *)arg;

    for (int i = w->row_start; i < w->row_end; i++) {
        for (int j = 0; j < w->p; j++) {
            double sum = 0.0;
            for (int k = 0; k < w->n; k++) {
                sum += w->A[idx2(i, k, w->n)] * w->B[idx2(k, j, w->p)];
            }
            w->C[idx2(i, j, w->p)] = sum;
        }
    }
    return NULL;
}

void matmul_pthread(int m, int n, int p,
                    const double *A, const double *B, double *C,
                    int num_threads) {
    if (num_threads < 1) num_threads = 1;
    if (num_threads > m) num_threads = m; // no point having more threads than rows

    pthread_t *threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    worker_args_t *args = (worker_args_t *)malloc((size_t)num_threads * sizeof(worker_args_t));
    if (!threads || !args) {
        // Fallback: run serial if allocation fails
        matmul_serial(m, n, p, A, B, C);
        free(threads);
        free(args);
        return;
    }

    // Split rows of C among threads as evenly as possible.
    int base = m / num_threads;
    int rem  = m % num_threads;

    int row = 0;
    for (int t = 0; t < num_threads; t++) {
        int take = base + (t < rem ? 1 : 0);
        args[t].m = m; args[t].n = n; args[t].p = p;
        args[t].A = A; args[t].B = B; args[t].C = C;
        args[t].row_start = row;
        args[t].row_end = row + take;
        row += take;

        pthread_create(&threads[t], NULL, worker_rows, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
}
