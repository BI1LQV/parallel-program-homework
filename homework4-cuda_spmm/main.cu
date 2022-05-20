#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cuda_gemm.cuh"
#include "cuda_gemm_shared.cuh"
#include "cuda_spmm1.cuh"
#include "cusparse_spmm.cuh"
#include "gemm.cuh"
#include "gemm_cublas.cuh"
#include "spmm.cuh"

void swap(int *a, int *b) {
    if (*a != *b)
        *a ^= *b ^= *a ^= *b;
}

int cmp(const void *a, const void *b) {
    return *((int *) a) - *((int *) b);
}


void GenerateCsr(int **RowPtr, int **ColIdx, int m) {
    srand(m);
    const int nnzRate = NNZ / 100.0 * m;
    *RowPtr = (int *) malloc(sizeof(int) * (m + 1));
    *ColIdx = (int *) malloc(sizeof(int) * (m * nnzRate));

    int *randCol = (int *) malloc(sizeof(int) * m * 2);
    for (int i = 0; i < m; ++i) {
        randCol[i] = i;
    }

    for (int i = 0; i < m; ++i) {
        swap(randCol + i, randCol + rand() % m);
    }
    memcpy(randCol + m, randCol, sizeof(int) * m);
    (*RowPtr)[0] = 0;

    for (int i = 1; i <= m; ++i) {
        int nnz = rand() % nnzRate + 1;
        if (nnz > m)nnz = m;
        (*RowPtr)[i] = (*RowPtr)[i - 1] + nnz;
        int buff = rand() % m;
        memcpy(*ColIdx + (*RowPtr)[i - 1], randCol + buff, nnz * sizeof(int));
        qsort(*ColIdx + (*RowPtr)[i - 1], nnz, sizeof(int), cmp);
    }

    *ColIdx = (int *) realloc(*ColIdx, sizeof(int) * (*RowPtr)[m]);
    free(randCol);
}

void GeMM(int m, int width,
          VALUE_TYPE *MatrixVal, VALUE_TYPE *denseRightMatrix,
          VALUE_TYPE *Res, double *time_val) {

    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(Res, 0, sizeof(VALUE_TYPE) * width * m);
        timeStart();
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < width; ++k) {
                    Res[i * width + k] += MatrixVal[i * m + j] * denseRightMatrix[j * width + k];
                }
            }
        }

        *time_val += timeCut();
    }
    *time_val /= BENCH_TIMES;
}


void compareUndPrint(const char *name, const double *C_Golden, const double *C_ref, int m, int n) {

    int count1 = 0;

    //   for (int i = 0; i < m * n; i++)
    //     printf("%d %d %f %f\n",i/n,i%n,C_ref[i],C_Golden[i]);
    for (int i = 0; i < m * n; i++)
        if (C_Golden[i] != C_ref[i]) {
            //printf("%d %d %f %f\n",i/n,i%n,C_ref[i],C_Golden[i]);
            count1++;
        }
    if (count1 == 0)
        printf("(%s)(row-col, A and B are in row-major) PASS!\n\n", name);
    else
        printf("(%s)(row-col, A and B are in row-major) NOT PASS!\n\n", name);
}


int calc(int m,int width){
    printf("%d,%d,",m,width);
    int *RowPtr, *ColIdx;
    GenerateCsr(&RowPtr, &ColIdx, m);
    VALUE_TYPE *CsrVal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (RowPtr[m]));
    for (int i = 0; i < RowPtr[m]; ++i) {
        CsrVal[i] = 1;//(rand() % 8 + 1) / 8.0;
    }

    VALUE_TYPE *DenseMatrixVal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m * m);
    memset(DenseMatrixVal, 0, sizeof(VALUE_TYPE) * m * m);

    for (int i = 0; i < m; ++i) {
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; ++j) {
            DenseMatrixVal[i * m + ColIdx[j]] = CsrVal[j];
        }
    }

    VALUE_TYPE *RightThinMatrix = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    srand(width);
    for (int i = 0; i < width * m; ++i) {
        RightThinMatrix[i] = 1;//rand() % 32 * 0.125;
    }
    VALUE_TYPE *Res_Golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    VALUE_TYPE *Res = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * width * m);
    double time_value;


    double gflops_D = 2.0 * m * m * width / 1e9;
    double gflops_S = 2.0 * RowPtr[m] * width / 1e9;

    GeMM(m, width, DenseMatrixVal, RightThinMatrix, Res_Golden, &time_value);
    const char *Name = "GeMM";
    //printf("\n(%s)(row-col, A and B are in row-major)) used %4.5f ms, %.5f gflops\n",
    //       Name, time_value, gflops_D / time_value);

    gemm_ref(DenseMatrixVal, RightThinMatrix, Res,m,m, width, &time_value);
    Name = "(a) GEMM_OpenMP";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);


    gemm_OpenBlas(DenseMatrixVal, RightThinMatrix, Res,m,m, width, &time_value);
    Name = "(b) GEMM_OpenBLAS";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    gemm_cuda(DenseMatrixVal, RightThinMatrix, Res,m,m, width, &time_value);
    Name = "(c) GEMM_CUDA_global_memory";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    gemm_cuda_shared(DenseMatrixVal, RightThinMatrix, Res,m,m, width, &time_value);
    Name = "(d) GEMM_CUDA_shared_memory";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);


    gemm_cublas(DenseMatrixVal, RightThinMatrix, Res,m,m, width, &time_value);
    Name = "(e) GEMM_cuBLAS";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);


    csrSpMM_serial(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "(f) csrSpMM_serial";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    csrSpMM(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "(g) csrSpMM_OpenMP";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    spMM_cuda1_yours(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "(h) csrSpMM_CUDA_scalar";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    spMM_cuda16_yours(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "(i) csrSpMM_CUDA_vector";
    printf("%4.5f,%.5f,", time_value, gflops_S / time_value);

    spMM_cusparse(m, RowPtr, ColIdx, CsrVal, width, RightThinMatrix, Res, &time_value);
    Name = "(j) csrSpMM_cuSPARSE";
    printf("%4.5f,%.5f\n", time_value, gflops_S / time_value);

    return 0;
}
int main (){
    printf("m,width,(a) GEMM_OpenMP,(a) GEMM_OpenMP,(b) GEMM_OpenBLAS,(b) GEMM_OpenBLAS,(c) GEMM_CUDA_global_memory,(c) GEMM_CUDA_global_memory,(d) GEMM_CUDA_shared_memory,(d) GEMM_CUDA_shared_memory,(e) GEMM_cuBLAS,(e) GEMM_cuBLAS,(f) csrSpMM_serial,(f) csrSpMM_serial,(g) csrSpMM_OpenMP,(g) csrSpMM_OpenMP,(h) csrSpMM_CUDA_scalar,(h) csrSpMM_CUDA_scalar,(i) csrSpMM_CUDA_vector,(i) csrSpMM_CUDA_vector,(j) csrSpMM_cuSPARSE,(j) csrSpMM_cuSPARSE\n");
    for(int m=100;m<=2000;m+=100){
        calc(m,16);
        calc(m,32);
        calc(m,64);
    }
    return 0;
}