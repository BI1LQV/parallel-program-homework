//
// Created by kouushou on 2021/5/12.
//
#include "gemm.h"


__global__ void MatrixMulKernel(float *A, float *B, float *C, int m, int k, int n) {
// Each thread computes one element of C
// by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        for (int e = 0; e < k; ++e)
            Cvalue += A[row * k + e]
                      * B[e * n + col];
        C[row * n + col] = Cvalue;
    }
}


void gemm_cuda(float *A, float *B, float *C, int m, int k, int n,double *time_value) {
    float *d_A, *d_B, *d_C;
    size_t size = m * k * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size,
               cudaMemcpyHostToDevice);

    size = k * n * sizeof(float);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory

    size = m * n * sizeof(float);
    cudaMalloc(&d_C, size);
// Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n / dimBlock.x) + (n % dimBlock.x != 0), (m / dimBlock.y) + (m % dimBlock.y != 0));
    for(int i = 0 ; i < WARMUP_TIMES ; ++i){
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
    }

    timeval t1,t2;
    *time_value = 0;
    cudaDeviceSynchronize();

    for (int i = 0 ; i < BENCH_TIMES ; ++i) {

        gettimeofday(&t1, nullptr);
        MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();
        gettimeofday(&t2, nullptr);
        *time_value += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;

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
// M(row, col) = *(M.elements + row * M.stride + col)


