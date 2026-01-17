#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

void matmul_serial(int m, int n, int p,
                   const double *A, const double *B, double *C);

void matmul_pthread(int m, int n, int p,
                    const double *A, const double *B, double *C,
                    int num_threads);

#endif
