// Parallel Programming (English), Spring 2021
// Weifeng Liu, China University of Petroleum-Beijing
// Homework 1. *Use ``#pragma omp parallel for''
//             for accelerating for loops of GEMM in
//             the function gemm_yours().
//             *Try to put the OpenMP directive on
//             top of the three for loops and see
//             the performance.
//             *Also, try to explore better storage
//             methods and faster ways for optimize GEMM.

#ifndef HOMEWORK3_CUDA_GEMM_GEMM_H
#define HOMEWORK3_CUDA_GEMM_GEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define BLOCK_SIZE 16
#define BENCH_TIMES 10
#define WARMUP_TIMES 5


void gemm_ref(float *A, float *B, float *C, int m, int k, int n) ;

void gemm_yours(float *A, float *B, float *C, int m, int k, int n) ;

void gemm_OpenBlas(float *A, float *B, float *C, int m, int k, int n) ;

#if defined(__cplusplus)
extern "C"{
#endif

void gemm_cuda(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cuda_shared(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cublas(float *A, float *B, float *C, int m, int k, int n,double *time_value);

void gemm_cuda_yours(float *A, float *B, float *C, int m, int k, int n,double *time_value);

#if defined(__cplusplus)
}
#endif

#endif //HOMEWORK3_CUDA_GEMM_GEMM_H
