//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"

#ifndef MATRIXMULTISAMPLES_CUDA_GEMM_CUH
#define MATRIXMULTISAMPLES_CUDA_GEMM_CUH


__global__ void MatrixMulKernel(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n) {
// Each thread computes one element of C
// by accumulating results into Cvalue
    VALUE_TYPE Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        for (int e = 0; e < k; ++e)
            Cvalue += A[row * k + e]
                      * B[e * n + col];
        C[row * n + col] = Cvalue;
    }
}


void gemm_cuda(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n,double *time_value) {
    VALUE_TYPE *d_A, *d_B, *d_C;
    size_t size = m * k * sizeof(VALUE_TYPE);
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
// Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n / dimBlock.x) + (n % dimBlock.x != 0), (m / dimBlock.y) + (m % dimBlock.y != 0));
    for(int i = 0 ; i < WARMUP_TIMES ; ++i){
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
    }


    *time_value = 0;
    cudaDeviceSynchronize();

    for (int i = 0 ; i < BENCH_TIMES ; ++i) {

        timeStart();
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();

        *time_value += timeCut();
    }



// timer end
    // Read C from device memory
    cudaMemcpy(C, d_C, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
// Free device memory
}

#endif //MATRIXMULTISAMPLES_CUDA_GEMM_CUH
