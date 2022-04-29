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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <openblas_config.h>
#include <generated/cblas.h>

#include <omp.h>

// the reference code of GEMM
void gemm_ref(double *A, double *B, double *C, int m, int k, int n)
{
    for (int mi = 0; mi < m; mi++)
    {
        for (int ni = 0; ni < n; ni++)
        {
            for (int ki = 0; ki < k; ki++)
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
        }
    }
}

void oppp(double *A, double *B, double *C, int m, int k, int n)
{
    for (int mi = 0; mi < m; mi += 4)
    {
        for (int ni = 0; ni < n; ni += 4)
        {
            register double c00 = 0, c01 = 0, c02 = 0, c03 = 0,
                            c10 = 0, c11 = 0, c12 = 0, c13 = 0,
                            c20 = 0, c21 = 0, c22 = 0, c23 = 0,
                            c30 = 0, c31 = 0, c32 = 0, c33 = 0;
            register double bi0, bi1, bi2, bi3;
            register double a0i, a1i, a2i, a3i;
            double *a0i_p, *a1i_p, *a2i_p, *a3i_p;
            a0i_p = A + mi * k;
            a1i_p = A + (mi + 1) * k;
            a2i_p = A + (mi + 2) * k;
            a3i_p = A + (mi + 3) * k;
            for (int ki = 0; ki < k; ki++)
            {
                bi0 = B[ni + 0 + ki * n];
                bi1 = B[ni + 1 + ki * n];
                bi2 = B[ni + 2 + ki * n];
                bi3 = B[ni + 3 + ki * n];
                a0i = *a0i_p++;
                a1i = *a1i_p++;
                a2i = *a2i_p++;
                a3i = *a3i_p++;
                c00 += a0i * bi0;
                c01 += a0i * bi1;
                c02 += a0i * bi2;
                c03 += a0i * bi3;

                c10 += a1i * bi0;
                c11 += a1i * bi1;
                c12 += a1i * bi2;
                c13 += a1i * bi3;

                c20 += a2i * bi0;
                c21 += a2i * bi1;
                c22 += a2i * bi2;
                c23 += a2i * bi3;

                c30 += a3i * bi0;
                c31 += a3i * bi1;
                c32 += a3i * bi2;
                c33 += a3i * bi3;
            }
            C[ni + 0 + (mi + 0) * n] += c00;
            C[ni + 1 + (mi + 0) * n] += c01;
            C[ni + 2 + (mi + 0) * n] += c02;
            C[ni + 3 + (mi + 0) * n] += c03;

            C[ni + 0 + (mi + 1) * n] += c10;
            C[ni + 1 + (mi + 1) * n] += c11;
            C[ni + 2 + (mi + 1) * n] += c12;
            C[ni + 3 + (mi + 1) * n] += c13;

            C[ni + 0 + (mi + 2) * n] += c20;
            C[ni + 1 + (mi + 2) * n] += c21;
            C[ni + 2 + (mi + 2) * n] += c22;
            C[ni + 3 + (mi + 2) * n] += c23;

            C[ni + 0 + (mi + 3) * n] += c30;
            C[ni + 1 + (mi + 3) * n] += c31;
            C[ni + 2 + (mi + 3) * n] += c32;
            C[ni + 3 + (mi + 3) * n] += c33;
        }
    }
}

double *mul(double *A, double *B, int m, int k, int n)
{
    double *C = (double *)malloc(sizeof(double) * m * n);
    for (int mi = 0; mi < m; mi++)
    {
        for (int ni = 0; ni < n; ni++)
        {
            for (int ki = 0; ki < k; ki++)
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
        }
    }
    return C;
}

double *optm(double *A, double *B, int len)
{
    if (len < 800)
    {
        return mul(A, B, len, len, len);
    }
    int newSize = len * len / 4;
    double *a1 = (double *)malloc(sizeof(double) * newSize);
    printf("%ld", sizeof(double) * newSize);
    double *a2 = (double *)malloc(sizeof(double) * newSize);
    double *a3 = (double *)malloc(sizeof(double) * newSize);
    double *a4 = (double *)malloc(sizeof(double) * newSize);
    double *b1 = (double *)malloc(sizeof(double) * newSize);
    double *b2 = (double *)malloc(sizeof(double) * newSize);
    double *b3 = (double *)malloc(sizeof(double) * newSize);
    double *b4 = (double *)malloc(sizeof(double) * newSize);
    for (int i = 0; i < len / 2; i++)
    {

        int offset = i * len / 2 * sizeof(double);
        printf("%d %d\n", offset, i * len);
        memcpy(a1 + offset, A + i * len, len / 2 - 1);
        memcpy(a2 + offset, A + i * len + len / 2, len / 2 - 1);
        memcpy(a3 + offset, A + len / 2 * len + i * len, len / 2 - 1);
        memcpy(a4 + offset, A + len / 2 * len + i * len + len / 2, len / 2 - 1);
        memcpy(b1 + offset, B + i * len, len / 2 - 1);
        memcpy(b2 + offset, B + i * len + len / 2, len / 2 - 1);
        memcpy(b3 + offset, B + len / 2 * len + i * len, len / 2 - 1);
        memcpy(b4 + offset, B + len / 2 * len + i * len + len / 2, len / 2 - 1);
    }

    double *c1a = optm(a1, b1, len / 2);
    double *c1b = optm(a2, b3, len / 2);
    double *c2a = optm(a1, b2, len / 2);
    double *c2b = optm(a2, b4, len / 2);
    double *c3a = optm(a3, b1, len / 2);
    double *c3b = optm(a4, b3, len / 2);
    double *c4a = optm(a3, b2, len / 2);
    double *c4b = optm(a4, b4, len / 2);
    for (int i = 0; i < newSize; i++)
    {
        c1a[i] += c1b[i];
        c2a[i] += c2b[i];
        c3a[i] += c3b[i];
        c4a[i] += c4b[i];
    }
    double *c = (double *)malloc(sizeof(double) * len * len);
    memcpy(c, c1a, newSize);
    memcpy(c + newSize, c2a, newSize);
    memcpy(c + newSize * 2, c3a, newSize);
    memcpy(c + newSize * 3, c4a, newSize);
    return c;
}

// insert your code in this function and run
void kmn(double *A, double *B, double *C, int m, int k, int n)
{
    for (int ki = 0; ki < k; ++ki)
    {
        for (int mi = 0; mi < m; ++mi)
        {
            for (int ni = 0; ni < n; ++ni)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}

void nmk(double *A, double *B, double *C, int m, int k, int n)
{
    for (int ni = 0; ni < n; ++ni)
    {
        for (int mi = 0; mi < m; ++mi)
        {
            for (int ki = 0; ki < k; ++ki)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}

void nkm(double *A, double *B, double *C, int m, int k, int n)
{
    for (int ni = 0; ni < n; ++ni)
    {
        for (int ki = 0; ki < k; ++ki)
        {
            for (int mi = 0; mi < m; ++mi)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}

void mnk(double *A, double *B, double *C, int m, int k, int n)
{
    for (int mi = 0; mi < m; ++mi)
    {
        for (int ni = 0; ni < n; ++ni)
        {
            for (int ki = 0; ki < k; ++ki)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}
void mkn(double *A, double *B, double *C, int m, int k, int n)
{
    for (int mi = 0; mi < m; ++mi)
    {
        for (int ki = 0; ki < k; ++ki)
        {
            for (int ni = 0; ni < n; ++ni)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}

void knm(double *A, double *B, double *C, int m, int k, int n)
{

    for (int ki = 0; ki < k; ++ki)
    {
        for (int ni = 0; ni < n; ++ni)
        {
            for (int mi = 0; mi < m; ++mi)
            {
                C[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
        }
    }
}

void gemm_OpenBlas(double *A, double *B, double *C, int m, int k, int n)
{
    enum CBLAS_ORDER order = CblasRowMajor;
    enum CBLAS_TRANSPOSE transposeA = CblasNoTrans;
    enum CBLAS_TRANSPOSE transposeB = CblasNoTrans;
    double alpha = 1;
    double beta = 1;
    cblas_dgemm(order, transposeA, transposeB, m, n, k, alpha, A, k, B, n, beta, C, n);
}

void calc(int n)
{
    struct timeval t1, t2;
    int m, k;
    m = k = n;
    double gflop = 2.0 * (double)m * (double)n * (double)k / 1000000000.0;

    // malloc A, B and Cs
    double *A = (double *)malloc(sizeof(double) * m * k);
    double *B = (double *)malloc(sizeof(double) * k * n);
    double *C_golden = (double *)malloc(sizeof(double) * m * n);
    // double *C_ref = (double *)malloc(sizeof(double) * m * n);
    double *C_yours = (double *)malloc(sizeof(double) * m * n);

    // randomly give values to elements in A and B
    srand(0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i * k + j] = rand() % m;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = rand() % k;

    // compute C_golden for validation
    // memset(C_golden, 0, sizeof(double) * m * n);
    // for (int mi = 0; mi < n; ++mi)
    // {
    //     for (int ki = 0; ki < n; ++ki)
    //     {
    //         for (int ni = 0; ni < n; ++ni)
    //         {
    //             C_golden[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
    //         }
    //     }
    // }
    // for (int mi = 0; mi < m; mi++)
    // {
    //     for (int ni = 0; ni < n; ni++)
    //     {
    //         for (int ki = 0; ki < k; ki++)
    //             C_golden[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
    //     }
    // }

    // the reference row-col method for GEMM, A in row-major, B in row-major
    memset(C_golden, 0, sizeof(double) * m * n);
    gettimeofday(&t1, NULL);
    gemm_ref(A, B, C_golden, m, k, n);
    gettimeofday(&t2, NULL);
    double time_rowrow1 = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("\n%d,%4.5f, %4.2f\n", n, time_rowrow1, gflop / time_rowrow1);

    // check results
    // int count1 = 0;
    // for (int i = 0; i < m * n; i++)
    //     if (C_golden[i] != C_ref[i])
    //         count1++;
    // if (count1 == 0)
    //     printf("\n\n");
    // else
    //     printf("GEMM (row-col, A and B are in row-major) NOT PASS!\n\n");

    // the your method for GEMM
    memset(C_yours, 0, sizeof(double) * m * n);
    gettimeofday(&t1, NULL);
    oppp(A, B, C_yours, m, k, n);
    gettimeofday(&t2, NULL);
    double time_yours = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("\n%d,%4.5f, %4.2f\n", n, time_yours, gflop / time_yours);

    // // check results
    int count2 = 0;
    for (int i = 0; i < m * n; i++)
        if (C_golden[i] != C_yours[i])
            count2++;
    if (count2 == 0)
        printf("\n\n");
    else
        printf("GEMM (your method) NOT PASS!\n\n");

    // memset(C_yours, 0, sizeof(double) * m * n);
    // gettimeofday(&t1, NULL);
    // C_yours = optm(A, B, 2000);
    // gettimeofday(&t2, NULL);
    // time_yours = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
    // printf("\n%d, %4.5f , %4.2f\n", n, time_yours, gflop / time_yours);

    // check results
    // int count2 = 0;
    // for (int i = 0; i < m * n; i++)
    //     if (C_golden[i] != C_yours[i])
    //         count2++;
    // if (count2 == 0)
    //     printf("pass\n");
    // else
    //     printf("GEMM (OpenBLAS) NOT PASS!%d\n\n",count2);

    // free memory
    free(A);
    free(B);
    // free(C_golden);
    // free(C_ref);
    free(C_yours);
}

int main(int argc, char **argv)
{

    for (int i = 100; i <= 2000; i += 100)
    {
        calc(i);
    }
}
