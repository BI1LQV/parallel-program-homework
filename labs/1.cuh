//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"
#ifndef MATRIXMULTISAMPLES_GEMM_CUBLAS_CUH
#define MATRIXMULTISAMPLES_GEMM_CUBLAS_CUH
#include <cublas.h>
#define VALUE_TYPE float
void toColIndx_(int line, int ld, VALUE_TYPE *val)
{
    VALUE_TYPE *temp = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * line * ld);

    for (int i = 0; i < ld; ++i)
    {
        for (int j = 0; j < line; ++j)
        {
            temp[i * line + j] = val[j * ld + i];
        }
    }
    memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
    free(temp);
}

void toRowIndx_(int line, int ld, VALUE_TYPE *val)
{
    VALUE_TYPE *temp = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * line * ld);

    for (int i = 0; i < line; ++i)
    {
        for (int j = 0; j < ld; ++j)
        {
            temp[i * ld + j] = val[j * line + i];
        }
    }
    memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
    free(temp);
}

void gemm_cublas(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n, double *time_value)
{
    VALUE_TYPE *d_A, *d_B, *d_C;

    size_t size = m * k * sizeof(VALUE_TYPE);
    toColIndx_(m, k, A);
    toColIndx_(k, n, B);
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size,
               cudaMemcpyHostToDevice);

    size = k * n * sizeof(VALUE_TYPE);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size,
               cudaMemcpyHostToDevice);
    // Allocate C in device memory

    size = m * n * sizeof(VALUE_TYPE);
    cudaMalloc(&d_C, size);

    cublasHandle_t s;
    VALUE_TYPE al = 1, ve = 0;
    cublasCreate_v2(&s);

    int lda = m;
    int ldb = k;
    int d_R = m;

    printf("dddd%d,%d,%d,%d,%d,%d\n", m, n, k, lda, ldb, d_R);
    cublasSgemm_v2(s, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_A, lda, d_B, ldb, &ve, d_C, d_R);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size,
               cudaMemcpyDeviceToHost);
    toRowIndx_(m, n, C);
    toRowIndx_(m, k, A);
    toRowIndx_(k, n, B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
#endif // MATRIXMULTISAMPLES_GEMM_CUBLAS_CUH
