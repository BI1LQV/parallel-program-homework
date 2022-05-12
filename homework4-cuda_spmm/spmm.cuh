//
// Created by kouushou on 2021/5/19.
//
#include "defines.h"
#ifndef MATRIXMULTISAMPLES_SPMM_CUH
#define MATRIXMULTISAMPLES_SPMM_CUH

void csrSpMM(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
             int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_val) {


    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(Res, 0, sizeof(VALUE_TYPE) * width * m);
        timeStart();
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = RowPtr[i]; j < RowPtr[i + 1]; ++j) {
                for (int k = 0; k < width; ++k) {
                    Res[i * width + k] += CsrVal[j] * denseRightMatrix[ColIdx[j] * width + k];
                }
            }
        }

        *time_val += timeCut();
    }
    *time_val /= BENCH_TIMES;
}


void csrSpMM_serial(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *CsrVal,
             int width, VALUE_TYPE *denseRightMatrix, VALUE_TYPE *Res, double *time_val) {


    *time_val = 0;
    for (int _ = 0; _ < BENCH_TIMES; ++_) {
        memset(Res, 0, sizeof(VALUE_TYPE) * width * m);
        timeStart();

        for (int i = 0; i < m; ++i) {
            for (int j = RowPtr[i]; j < RowPtr[i + 1]; ++j) {
                for (int k = 0; k < width; ++k) {
                    Res[i * width + k] += CsrVal[j] * denseRightMatrix[ColIdx[j] * width + k];
                }
            }
        }

        *time_val += timeCut();
    }
    *time_val /= BENCH_TIMES;
}
#endif //MATRIXMULTISAMPLES_SPMM_CUH
