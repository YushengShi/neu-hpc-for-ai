#include "matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_constant(double *x, int len, double v) {
    for (int i = 0; i < len; i++) x[i] = v;
}

static double checksum(const double *x, int len) {
    double s = 0.0;
    for (int i = 0; i < len; i++) s += x[i];
    return s;
}

static double run_serial(int m, int n, int p, const double *A, const double *B, double *C, int iters) {
    double best = 1e300;
    for (int r = 0; r < iters; r++) {
        memset(C, 0, (size_t)m * (size_t)p * sizeof(double));
        double t0 = now_seconds();
        matmul_serial(m, n, p, A, B, C);
        double t1 = now_seconds();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double run_parallel(int m, int n, int p, const double *A, const double *B, double *C, int threads, int iters) {
    double best = 1e300;
    for (int r = 0; r < iters; r++) {
        memset(C, 0, (size_t)m * (size_t)p * sizeof(double));
        double t0 = now_seconds();
        matmul_pthread(m, n, p, A, B, C, threads);
        double t1 = now_seconds();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    // You can override sizes: ./benchmark 1024 1024 1024
    int m = 1024, n = 1024, p = 1024;
    if (argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
    }

    // Bigger matrices => measurable speedup
    int iters = 3;

    int sizeA = m * n;
    int sizeB = n * p;
    int sizeC = m * p;

    double *A = (double *)malloc((size_t)sizeA * sizeof(double));
    double *B = (double *)malloc((size_t)sizeB * sizeof(double));
    double *C = (double *)malloc((size_t)sizeC * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed. Try smaller sizes.\n");
        return 1;
    }

    // Use deterministic contents (also helps avoid any weird “uninitialized” effects)
    fill_constant(A, sizeA, 1.0);
    fill_constant(B, sizeB, 1.0);

    printf("Matrix sizes: A=%dx%d, B=%dx%d, C=%dx%d\n", m, n, n, p, m, p);
    printf("Timing: best of %d runs (CLOCK_MONOTONIC)\n\n", iters);

    double t1 = run_serial(m, n, p, A, B, C, iters);
    double chk = checksum(C, sizeC);

    printf("Threads\tTime(s)\t\tSpeedup\t\tChecksum\n");
    printf("1(s)\t%.6f\t1.000\t\t%.3f\n", t1, chk);

    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_counts = (int)(sizeof(thread_counts) / sizeof(thread_counts[0]));

    for (int i = 0; i < num_counts; i++) {
        int th = thread_counts[i];
        double tn = run_parallel(m, n, p, A, B, C, th, iters);
        double speedup = t1 / tn;
        double cchk = checksum(C, sizeC);
        printf("%d\t%.6f\t%.3f\t\t%.3f\n", th, tn, speedup, cchk);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
