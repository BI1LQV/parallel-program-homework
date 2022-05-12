//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"
#ifndef MATRIXMULTISAMPLES_CUDA_SPMM1_CUH
#define MATRIXMULTISAMPLES_CUDA_SPMM1_CUH
__global__ void SpMMKernel(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                           int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res) {
// Each thread computes one element of C
// by accumulating results into Cvalue
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    //for (int k = 0; k < width; ++k) {
    //  Res[row * width + k] = 0;
    //}
    if(row >= m || col >= width) return;
    double val = 0;
    for (int j = RowPtr[row]; j < RowPtr[row + 1]; ++j) {
        //for (int k = 0; k < width; ++k) {
        val += CsrVal[j] * denseRightMatrix[ColIdx[j] * width + col];
        //}
    }
    Res[row * width + col] = val;
}

void spMM_cuda1_yours(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                      int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_value) {

    int *d_RowPtr, *d_ColIdx;

    size_t size = (m + 1) * sizeof(int);
    cudaMalloc(&d_RowPtr, size);
    cudaMemcpy(d_RowPtr, RowPtr, size,
               cudaMemcpyHostToDevice);

    size = RowPtr[m] * sizeof(int);
    cudaMalloc(&d_ColIdx, size);
    cudaMemcpy(d_ColIdx, ColIdx, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory

    double *d_CsrVal, *d_denseRightMatrix, *d_Res;
    size = RowPtr[m] * sizeof(double);
    cudaMalloc(&d_CsrVal, size);
    cudaMemcpy(d_CsrVal, CsrVal, size,
               cudaMemcpyHostToDevice);

    size = sizeof(double) * m * width;

    cudaMalloc(&d_denseRightMatrix, size);
    cudaMemcpy(d_denseRightMatrix, denseRightMatrix, size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_Res, size);
    dim3 dimBlock(1, 1);
    dim3 dimGrid((m), (width));

    for (int i = 0; i < WARMUP_TIMES; ++i) {

        ///// edit your warmup code here

        SpMMKernel<<<dimGrid, dimBlock>>>(m, d_RowPtr, d_ColIdx, d_CsrVal,
                                          width, d_denseRightMatrix, d_Res);
        ////
    }

    cudaDeviceSynchronize();
    *time_value = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);
        timeStart();
        ///// edit your code here

        SpMMKernel<<<dimGrid, dimBlock>>>(m, d_RowPtr, d_ColIdx, d_CsrVal,
                                          width, d_denseRightMatrix, d_Res);


        ////

        cudaDeviceSynchronize();

        *time_value += timeCut();
    }


    *time_value /= BENCH_TIMES;
    cudaMemcpy(Res, d_Res, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_ColIdx);
    cudaFree(d_Res);
    cudaFree(d_CsrVal);
    cudaFree(d_RowPtr);
    cudaFree(d_denseRightMatrix);

}

void spMM_cuda16_yours(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                       int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_value) {

    int *d_RowPtr, *d_ColIdx;

    size_t size = (m + 1) * sizeof(int);
    cudaMalloc(&d_RowPtr, size);
    cudaMemcpy(d_RowPtr, RowPtr, size,
               cudaMemcpyHostToDevice);

    size = RowPtr[m] * sizeof(int);
    cudaMalloc(&d_ColIdx, size);
    cudaMemcpy(d_ColIdx, ColIdx, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory

    double *d_CsrVal, *d_denseRightMatrix, *d_Res;
    size = RowPtr[m] * sizeof(double);
    cudaMalloc(&d_CsrVal, size);
    cudaMemcpy(d_CsrVal, CsrVal, size,
               cudaMemcpyHostToDevice);

    size = sizeof(double) * m * width;

    cudaMalloc(&d_denseRightMatrix, size);
    cudaMemcpy(d_denseRightMatrix, denseRightMatrix, size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_Res, size);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m+BLOCK_SIZE-1)/BLOCK_SIZE, (width+BLOCK_SIZE-1)/BLOCK_SIZE);

    for (int i = 0; i < WARMUP_TIMES; ++i) {

        ///// edit your warmup code here

        SpMMKernel<<<dimGrid, dimBlock>>>(m, d_RowPtr, d_ColIdx, d_CsrVal,
                                          width, d_denseRightMatrix, d_Res);
        ////
    }

    cudaDeviceSynchronize();
    *time_value = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);
        timeStart();
        ///// edit your code here

        SpMMKernel<<<dimGrid, dimBlock>>>(m, d_RowPtr, d_ColIdx, d_CsrVal,
                                          width, d_denseRightMatrix, d_Res);


        ////

        cudaDeviceSynchronize();

        *time_value += timeCut();
    }


    *time_value /= BENCH_TIMES;
    cudaMemcpy(Res, d_Res, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_ColIdx);
    cudaFree(d_Res);
    cudaFree(d_CsrVal);
    cudaFree(d_RowPtr);
    cudaFree(d_denseRightMatrix);

}

#endif //MATRIXMULTISAMPLES_CUDA_SPMM1_CUH
