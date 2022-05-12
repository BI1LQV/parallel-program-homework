//
// Created by kouushou on 2021/5/19.
//

#ifndef MATRIXMULTISAMPLES_CUSPARSE_SPMM_CUH
#define MATRIXMULTISAMPLES_CUSPARSE_SPMM_CUH
#include <cusparse.h>

void toColIndx(int line, int ld, VALUE_TYPE *val) {
    VALUE_TYPE *temp = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * line * ld);

    for (int i = 0; i < ld; ++i) {
        for (int j = 0; j < line; ++j) {
            temp[i * line + j] = val[j * ld + i];
        }
    }
    memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
    free(temp);
}

void toRowIndx(int line, int ld, VALUE_TYPE *val) {
    VALUE_TYPE *temp = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * line * ld);

    for (int i = 0; i < line; ++i) {
        for (int j = 0; j < ld; ++j) {
            temp[i * ld + j] = val[j * line + i];
        }
    }
    memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
    free(temp);
}

void spMM_cusparse(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
                   int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_value) {

    int *d_RowPtr, *d_ColIdx;


    size_t size = (m + 1) * sizeof(int);
    //toColIndx(width,m,denseRightMatrix);
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

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseOperation_t A = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t B = CUSPARSE_OPERATION_NON_TRANSPOSE;

    VALUE_TYPE al = 1, be = 0;
    cusparseSpMatDescr_t csrMtxA;
    cusparseCreateCsr(&csrMtxA, (int64_t) m, (int64_t) m,
                      (int64_t) RowPtr[m], d_RowPtr, d_ColIdx, d_CsrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_64F
    );

    cusparseDnMatDescr_t dnsMtx;
    cusparseCreateDnMat(&dnsMtx, (int64_t) m, (int64_t) width,
                        (int64_t) m, d_denseRightMatrix, CUDA_R_64F, CUSPARSE_ORDER_COL);


    cusparseDnMatDescr_t ResDnsMtx;
    cusparseCreateDnMat(&ResDnsMtx, (int64_t) m, (int64_t) width,
                        (int64_t) m, d_Res, CUDA_R_64F, CUSPARSE_ORDER_COL);

    for (int i = 0; i < WARMUP_TIMES; ++i) {

        ///// edit your warmup code here
        cusparseSpMM(handle, A, B, &al, csrMtxA, dnsMtx, &be, ResDnsMtx, CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, NULL);
        ////
    }

    cudaDeviceSynchronize();
    *time_value = 0;
    for (int i = 0; i < BENCH_TIMES; ++i) {
        // cublasSgemm('N', 'N', m, n, k, 1.0f, d_A, m, d_B, k, 0, d_C, m);
        timeStart();
        ///// edit your code here

        cusparseSpMM(handle, A, B, &al, csrMtxA, dnsMtx, &be, ResDnsMtx,
                     CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, NULL);

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

    cusparseDestroyDnMat(ResDnsMtx);
    cusparseDestroyDnMat(dnsMtx);
    cusparseDestroySpMat(csrMtxA);
    //toRowIndx(m,width,denseRightMatrix);
    toRowIndx(m, width, Res);
}

#endif //MATRIXMULTISAMPLES_CUSPARSE_SPMM_CUH
