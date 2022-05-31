#include "gemm.h"

#include <openblas_config.h>
#include <generated/cblas.h>
#include <omp.h>
void gemm_ref(float *A, float *B, float *C, int m, int k, int n) {
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            for (int ki = 0; ki < k; ki++)
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
        }
    }
}


void gemm_yours(float *A, float *B, float *C, int m, int k, int n) {
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int ki = 0; ki < k; ++ki) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * k + ki] * B[ki * n + j];
            }
        }
    }
}

void gemm_OpenBlas(float *A, float *B, float *C, int m, int k, int n) {
    enum CBLAS_ORDER order = CblasRowMajor;
    enum CBLAS_TRANSPOSE transposeA = CblasNoTrans;
    enum CBLAS_TRANSPOSE transposeB = CblasNoTrans;
    float alpha = 1;
    float beta = 1;
    cblas_sgemm(order,transposeA,transposeB,m,n,k,alpha,A,k,B,n,beta,C,n);
}