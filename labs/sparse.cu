#include <cusparse.h>
#include <stdio.h>
int main()
{
    float a[] = {1, 0, -1, 4, 3, 2, 1, 0};
    // float cscval[] = {1, -1, -2};
    // int cscidx[] = {0, 0, 1};
    // int colptr[] = {0, 1, 3};
    float bvals[] = {1, -1, 0, -2};
    float *d_a, *d_cscval;
    int *d_cscidx, *d_colptr;
    float *d_res;
    cudaMalloc(&d_a, sizeof(float) * 8);
    cudaMalloc(&d_cscval, sizeof(float) * 3);
    cudaMalloc(&d_cscidx, sizeof(int) * 3);
    cudaMalloc(&d_colptr, sizeof(int) * 3);
    cudaMalloc(&d_res, sizeof(float) * 8);
    cudaMemcpy(d_a, a, sizeof(float) * 8, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cscval, cscval, sizeof(float) * 3, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cscidx, cscidx, sizeof(int) * 3, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_colptr, colptr, sizeof(int) * 3, cudaMemcpyHostToDevice);

    int dataNumInCol = 0;
    int B_csc_value_idx = 0;
    int B_csc_rowIdx_idx = 0;
    int B_csc_colPtr_idx = 0;
    float *B_csc_value_tmp = (float *)malloc((3) * sizeof(float));
    int *B_csc_rowIdx_tmp = (int *)malloc((3) * sizeof(int));
    int *B_csc_colPtr_tmp = (int *)malloc((2 + 1) * sizeof(int));
    B_csc_colPtr_tmp[0] = 0;
    for (int colIdx = 0; colIdx < 2; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < 2; rowIdx++)
        {
            if (bvals[rowIdx * 2 + colIdx])
            {
                B_csc_value_tmp[B_csc_value_idx++] = bvals[rowIdx * 2 + colIdx];
                dataNumInCol++;
                B_csc_rowIdx_tmp[B_csc_rowIdx_idx++] = rowIdx;
            }
        }
        B_csc_colPtr_tmp[B_csc_colPtr_idx + 1] = B_csc_colPtr_tmp[B_csc_colPtr_idx] + dataNumInCol;
        B_csc_colPtr_idx++;
        dataNumInCol = 0;
    }
    for (int i = 0; i < 3; i++)
    {
        printf("%f,%d,%d\n", B_csc_value_tmp[i], B_csc_rowIdx_tmp[i], B_csc_colPtr_tmp[i]);
    }
    cudaMemcpy(d_cscval, B_csc_value_tmp, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscidx, B_csc_rowIdx_tmp, sizeof(int) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colptr, B_csc_colPtr_tmp, sizeof(int) * 3, cudaMemcpyHostToDevice);
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    float al = 1;
    float bl = 0;
    cusparseSgemmi(handle,
                   4,
                   2,
                   2,
                   3,
                   &al,
                   d_a,
                   4,
                   d_cscval,
                   d_colptr,
                   d_cscidx,
                   &bl,
                   d_res,
                   4);
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_res, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++)
    {
        printf("%f\n", a[i]);
    }
    return 0;
}
