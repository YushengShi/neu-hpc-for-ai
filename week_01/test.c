#include "matmul.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double rand_uniform(void) {
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0; // [-1, 1]
}

static void fill_random(double *x, int len) {
    for (int i = 0; i < len; i++) x[i] = rand_uniform();
}

static int nearly_equal(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= eps * scale;
}

static void compare_or_fail(const double *X, const double *Y, int rows, int cols, double eps,
                            int m, int n, int p, int threads) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double a = X[i * cols + j];
            double b = Y[i * cols + j];
            if (!nearly_equal(a, b, eps)) {
                fprintf(stderr,
                        "FAIL: A=%dx%d, B=%dx%d, threads=%d\n"
                        "Mismatch at C(%d,%d): got %.15g expected %.15g\n",
                        m, n, n, p, threads, i, j, a, b);
                exit(1);
            }
        }
    }
}

static void run_one_case(int m, int n, int p, int threads_to_test) {
    int sizeA = m * n;
    int sizeB = n * p;
    int sizeC = m * p;

    double *A = (double *)malloc((size_t)sizeA * sizeof(double));
    double *B = (double *)malloc((size_t)sizeB * sizeof(double));
    double *C_serial = (double *)malloc((size_t)sizeC * sizeof(double));
    double *C_par = (double *)malloc((size_t)sizeC * sizeof(double));

    if (!A || !B || !C_serial || !C_par) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    fill_random(A, sizeA);
    fill_random(B, sizeB);
    memset(C_serial, 0, (size_t)sizeC * sizeof(double));
    memset(C_par, 0, (size_t)sizeC * sizeof(double));

    matmul_serial(m, n, p, A, B, C_serial);
    matmul_pthread(m, n, p, A, B, C_par, threads_to_test);

    compare_or_fail(C_par, C_serial, m, p, 1e-10, m, n, p, threads_to_test);

    printf("PASS: A=%dx%d, B=%dx%d, threads=%d\n", m, n, n, p, threads_to_test);

    free(A);
    free(B);
    free(C_serial);
    free(C_par);
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("Running correctness tests (serial vs pthread)...\n\n");

    // Explicit cases mentioned by assignment style
    run_one_case(1, 1, 1, 1);   // A=1x1, B=1x1
    run_one_case(1, 1, 5, 4);   // A=1x1, B=1x5
    run_one_case(2, 1, 3, 4);   // A=2x1, B=1x3
    run_one_case(2, 2, 2, 2);   // A=2x2, B=2x2

    // Extra coverage: non-square
    run_one_case(3, 5, 4, 4);
    run_one_case(8, 3, 7, 8);
    run_one_case(5, 8, 2, 16);

    // Random variety tests
    for (int t = 0; t < 20; t++) {
        int m = 1 + rand() % 25;
        int n = 1 + rand() % 25;
        int p = 1 + rand() % 25;
        int threads = 1 + rand() % 16;
        run_one_case(m, n, p, threads);
    }

    printf("\nAll tests passed \n");
    return 0;
}
