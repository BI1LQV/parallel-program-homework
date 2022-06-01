#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include <cublas.h>
#include "1.cuh"
int main()
{
    cublasHandle_t s222;
    cublasCreate_v2(&s222);
    float al = 1, ve = 0;
    float a[5 * 10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float b[5 * 5] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc(&d_a, 50 * sizeof(float));
    cudaMalloc(&d_b, 25 * sizeof(float));
    cudaMalloc(&d_c, 50 * sizeof(float));
    cudaMemcpy(d_a, a, 50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 25 * sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemm_v2(s222,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   10, 5, 5,
                   &al,
                   d_a, 10,
                   d_b, 5,
                   &ve,
                   d_c, 10);
    cudaDeviceSynchronize();

    float p[50];
    cudaMemcpy(p, d_c, 50 * sizeof(float), cudaMemcpyDeviceToHost);

    double time;
    // gemm_cublas(a, b, p, 10, 5, 5, &time);
    for (int sd = 0; sd < 50; sd++)
    {
        printf("%f ", p[sd]);
    }
    return 0;
}
