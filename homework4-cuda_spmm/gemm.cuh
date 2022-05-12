//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"

#ifndef MATRIXMULTISAMPLES_GEMM_CUH
#define MATRIXMULTISAMPLES_GEMM_CUH

#include <openblas_config.h>
#include <cblas.h>

void gemm_ref(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n, double *time_val) {
    *time_val = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        memset(C,0,sizeof(VALUE_TYPE)*m*n);
        timeStart();
#pragma omp parallel for
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                for (int ki = 0; ki < k; ki++)
                    C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
        *time_val += timeCut();
    }
}


void gemm_yours(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n, double *time_val) {
    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(C,0,sizeof(VALUE_TYPE)*m*n);
        timeStart();
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int ki = 0; ki < k; ++ki) {
                for (int j = 0; j < n; ++j) {
                    C[i * n + j] += A[i * k + ki] * B[ki * n + j];
                }
            }
        }
        *time_val += timeCut();
    }
}

void gemm_OpenBlas(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n, double *time_val) {
    enum CBLAS_ORDER order = CblasRowMajor;
    enum CBLAS_TRANSPOSE transposeA = CblasNoTrans;
    enum CBLAS_TRANSPOSE transposeB = CblasNoTrans;
    VALUE_TYPE alpha = 1;
    VALUE_TYPE beta = 1;
    *time_val = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        memset(C,0,sizeof(VALUE_TYPE)*m*n);
        timeStart();
        cblas_dgemm(order, transposeA, transposeB, m, n, k, alpha, A, k, B, n, beta, C, n);
        *time_val += timeCut();
    }
}

#endif //MATRIXMULTISAMPLES_GEMM_CUH
