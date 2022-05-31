#include "gemm.h"

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

int main(int argc, char ** argv)
{

    // set matrix size
    if(argc!=4){
        printf("please input right parameter a,b,c\n");
        exit(0);
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
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
    {
#pragma omp parallel for
        for (int mi = 0; mi < m; mi++) {
            for (int ki = 0; ki < k; ki++) {
                for (int ni = 0; ni < n; ni++)
                    C_golden[mi * n + ni] += A[mi * k + ki] * B[ki * n + ni];
            }
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