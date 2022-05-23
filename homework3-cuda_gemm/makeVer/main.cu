
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <openblas_config.h>
#include <generated/cblas.h>
#include <omp.h>

#include <cublas.h>


#define BLOCK_SIZE 16
#define BENCH_TIMES 10
#define WARMUP_TIMES 5


void gemm_ref(float *A, float *B, float *C, int m, int k, int n) ;

void gemm_yours(float *A, float *B, float *C, int m, int k, int n) ;

void gemm_OpenBlas(float *A, float *B, float *C, int m, int k, int n) ;

void gemm_cuda(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cuda_shared(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cublas(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cuda_yours(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void compareUndPrint(const char *name,const float *C_Golden,const float *C_ref,int m,int n){

    int count1 = 0;
    for (int i = 0; i < m * n; i++)
        if (C_Golden[i] != C_ref[i]) {
            //    printf("%d %d %f %f\n",i/n,i%n,C[i],C_golden[i]);
            count1++;
        }
    if (count1 == 0)
        printf("GEMM (%s)(row-col, A and B are in row-major) PASS!\n\n",name);
    else
        printf("GEMM (%s)(row-col, A and B are in row-major) NOT PASS!\n\n",name);
}

int cir(int mmm)
{
    int m = mmm;
    int k = mmm;
    int n = mmm;
    printf("Matrix A is %i x %i, matrix B is %i x %i\n", m, k, k, n);
    float gflop = 2.0 * (float)m * (float)n * (float)k / 1000000000.0;

    // malloc A, B and Cs
    float *A = (float *)malloc(sizeof(float) * m * k);
    float *B = (float *)malloc(sizeof(float) * k * n);
    float *C_golden = (float *)malloc(sizeof(float) * m * n);
    float *C_ref = (float *)malloc(sizeof(float) * m * n);


    // randomly give values to elements in A and B
    srand(0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i * k + j] = 1;//rand() % 4;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = 1;//rand() % 4;

    // compute C_golden for validation
    memset(C_golden, 0, sizeof(float) * m * n);
#pragma omp parallel for
    for (int mi = 0; mi < m; mi++)
    {
        for (int ki = 0; ki < k; ki++)
        {
            for (int ni = 0; ni < n; ni++)
                C_golden[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
        }
    }


    struct timeval t1, t2;
    double time_value;
    const char*Name ;
    gflop = 2.0 * (float)m * (float)n * (float)k / 1000000000.0;

    memset(C_ref, 0, sizeof(float) * m * n);
    gettimeofday(&t1, NULL);

    gemm_yours(A,B,C_ref,m,k,n);

    gettimeofday(&t2, NULL);

    time_value = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    Name = "gemm_openMP";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s, %4.2f GFlop/s\n",
           Name,time_value, gflop/time_value);
    compareUndPrint(Name,C_ref,C_golden,m,n);


    memset(C_ref, 0, sizeof(float) * m * n);
    gettimeofday(&t1, NULL);

    gemm_OpenBlas(A,B,C_ref,m,k,n);

    gettimeofday(&t2, NULL);

    time_value = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    Name = "OpenBLAS";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s, %4.2f GFlop/s\n",
           Name,time_value, gflop/time_value);
    compareUndPrint(Name,C_ref,C_golden,m,n);







    memset(C_ref, 0, sizeof(float) * m * n);
    gemm_cuda(A,B,C_ref,m,k,n,&time_value);

    time_value /= BENCH_TIMES;
    Name = "cuda_global";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s for %d bench(s) in average, %4.2f GFlop/s\n",
           Name,time_value,BENCH_TIMES, gflop/time_value);

    compareUndPrint(Name,C_ref,C_golden,m,n);

    gemm_cuda_shared(A,B,C_ref,m,k,n,&time_value);

    time_value /= BENCH_TIMES;
    Name = "cuda_shared";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s for %d bench(s) in average, %4.2f GFlop/s\n",
           Name,time_value,BENCH_TIMES, gflop/time_value);

    compareUndPrint(Name,C_ref,C_golden,m,n);

    memset(C_ref, 0, sizeof(float) * m * n);
    gemm_cublas(A,B,C_ref,m,k,n,&time_value);

    time_value/=BENCH_TIMES;
    Name = "cublas";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s for %d bench(s) in average, %4.2f GFlop/s\n",
           Name,time_value,BENCH_TIMES, gflop/time_value);

    compareUndPrint(Name,C_ref,C_golden,m,n);


    memset(C_ref, 0, sizeof(float) * m * n);
    gemm_cuda_yours(A,B,C_ref,m,k,n,&time_value);

    time_value/=BENCH_TIMES;
    Name = "cuda_yours";
    printf("\nGEMM (%s)(row-col, A and B are in row-major)) used %4.5f s for %d bench(s) in average, %4.2f GFlop/s\n",
           Name,time_value,BENCH_TIMES, gflop/time_value);

    compareUndPrint(Name,C_ref,C_golden,m,n);

    // free memory
    free(A);
    free(B);
    free(C_golden);
    free(C_ref);

}

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


typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value) {
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
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    Matrix subC = GetSubMatrix(C, blockRow, blockCol);
    int tot = A.width / BLOCK_SIZE + (A.width % BLOCK_SIZE != 0);
    float CValue = 0;
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

void gemm_cuda_shared(float *A, float *B, float *C, int m, int k, int n,double *time_value) {
// Load A and B to device memory
    Matrix d_A;

    d_A.width = d_A.stride = k;
    d_A.height = m;
    size_t size = k * m * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = n;
    d_B.height = k;
    size = n * k * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B, size,
               cudaMemcpyHostToDevice);
// Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = n;
    d_C.height = m;
    size = n * m * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n / dimBlock.x) + (n % dimBlock.x != 0),
                 (m / dimBlock.y) + (m % dimBlock.y != 0));

    for(int i = 0 ; i < WARMUP_TIMES ; ++i){
        MatMulKernel_SharedMemory<<<dimGrid, dimBlock>>>(
                d_A, d_B, d_C);
    }

    timeval t1,t2;
    *time_value = 0;
    cudaDeviceSynchronize();
    for(int i = 0 ; i < BENCH_TIMES ; ++i) {
        gettimeofday(&t1, nullptr);
        MatMulKernel_SharedMemory<<<dimGrid, dimBlock>>>(
                d_A, d_B, d_C);
        cudaDeviceSynchronize();
        gettimeofday(&t2, nullptr);
        *time_value+=(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;

    }

    // Read C from device memory
    cudaMemcpy(C, d_C.elements, size,
               cudaMemcpyDeviceToHost);

// Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
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


void gemm_cublas(float *A, float *B, float *C, int m, int k, int n,double *time_value){
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



    cublasHandle_t s;
    float al = 1,ve=0;
    cublasCreate_v2(&s);

    for(int i = 0 ; i < WARMUP_TIMES ; ++i){
        cublasSgemm_v2(s,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&al,d_A,k,d_B,n,&ve,d_C,n);
    }
    timeval t1,t2;
    cudaDeviceSynchronize();

    for(int i = 0 ; i < BENCH_TIMES ; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);

        gettimeofday(&t1, nullptr);
        cublasSgemm_v2(s,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&al,d_A,k,d_B,n,&ve,d_C,n);
        cudaDeviceSynchronize();
        gettimeofday(&t2, nullptr);
        *time_value += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    }




    cudaMemcpy(C, d_C, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


void gemm_cuda_yours(float *A, float *B, float *C, int m, int k, int n,double *time_value){
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


    for(int i = 0 ; i < WARMUP_TIMES ; ++i){

        ///// edit your warmup code here

        ////
    }
    timeval t1,t2;
    cudaDeviceSynchronize();

    for(int i = 0 ; i < BENCH_TIMES ; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);

        gettimeofday(&t1, nullptr);
        ///// edit your code here



        ////

        cudaDeviceSynchronize();
        gettimeofday(&t2, nullptr);
        *time_value += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    }




    cudaMemcpy(C, d_C, size,
               cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    for(int p=100;p<=2000;p+=100){
        cir(p);
    }
    return 0;
}