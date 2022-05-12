//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"
#ifndef MATRIXMULTISAMPLES_CUDA_GEMM_SHARED_CUH
#define MATRIXMULTISAMPLES_CUDA_GEMM_SHARED_CUH
//
// Created by kouushou on 2021/5/12.
//
#include "defines.h"

typedef struct {
    int width;
    int height;
    int stride;
    VALUE_TYPE *elements;
} Matrix;

// Get a matrix element
__device__ VALUE_TYPE GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           VALUE_TYPE value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width
            = BLOCK_SIZE;
    Asub.height
            = BLOCK_SIZE;
    Asub.stride
            = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                + BLOCK_SIZE * col];
    return Asub;

}

#define ifin(r, c, rb, cb, mat_A) \
((c + cb*BLOCK_SIZE < mat_A.width) && (r+rb*BLOCK_SIZE < mat_A.height))

__global__ void MatMulKernel_SharedMemory(Matrix A, Matrix B, Matrix C) {
// Block row and column

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    __shared__ VALUE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ VALUE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    Matrix subC = GetSubMatrix(C, blockRow, blockCol);
    int tot = A.width / BLOCK_SIZE + (A.width % BLOCK_SIZE != 0);
    VALUE_TYPE CValue = 0;
    for (int i = 0; i < tot; ++i) {
        Matrix Asub = GetSubMatrix(A, blockRow, i);

        Matrix Bsub = GetSubMatrix(B, i, blockCol);
        //if (i * BLOCK_SIZE + col < A.width)
        if(ifin(row,col,blockRow,i,A)) {
            As[row][col] = GetElement(Asub, row, col);
        } else As[row][col] = 0;

        if(ifin(row,col,i,blockCol,B)) {
            Bs[row][col] = GetElement(Bsub, row, col);
        }else Bs[row][col] = 0;
        __syncthreads();
// Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            CValue += As[row][e] * Bs[e][col];
// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
        __syncthreads();
        if (ifin(row, col, blockRow, blockCol, C))
            SetElement(subC, row, col, CValue);
    }

}

void gemm_cuda_shared(VALUE_TYPE *A, VALUE_TYPE *B, VALUE_TYPE *C, int m, int k, int n,double *time_value) {
// Load A and B to device memory
    Matrix d_A;

    d_A.width = d_A.stride = k;
    d_A.height = m;
    size_t size = k * m * sizeof(VALUE_TYPE);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = n;
    d_B.height = k;
    size = n * k * sizeof(VALUE_TYPE);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = n;
    d_C.height = m;
    size = n * m * sizeof(VALUE_TYPE);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n / dimBlock.x) + (n % dimBlock.x != 0),
                 (m / dimBlock.y) + (m % dimBlock.y != 0));

    for(int i = 0 ; i < WARMUP_TIMES ; ++i){
        MatMulKernel_SharedMemory<<<dimGrid, dimBlock>>>(
                d_A, d_B, d_C);
    }


    *time_value = 0;
    cudaDeviceSynchronize();
    for(int i = 0 ; i < BENCH_TIMES ; ++i) {
        timeStart();
        MatMulKernel_SharedMemory<<<dimGrid, dimBlock>>>(
                d_A, d_B, d_C);
        cudaDeviceSynchronize();

        *time_value+=timeCut();
    }

    // Read C from device memory
    cudaMemcpy(C, d_C.elements, size,
               cudaMemcpyDeviceToHost);

// Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


#endif //MATRIXMULTISAMPLES_CUDA_GEMM_SHARED_CUH
