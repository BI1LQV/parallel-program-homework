#include <stdio.h>
__global__ void relu(float *d_C0_value, int mC, int nC)
{
    int i = (blockIdx.x * 2 + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * 4 + threadIdx.y;
    // float tmp = -0.3;
    // tmp += d_C0_value[i];
    // if (tmp <= 0)
    // {
    //     tmp = 0;
    // }
    // else if (tmp >= 32)
    // {
    //     tmp = 32;
    // }
    // d_C0_value[i] = tmp;
    printf("%d,", i);
}
int main()
{
    float s[25] = {0, -1, 2, 1, 39, 0, -1, 2, 1, 33, 0, -1, 2, 1, 33, 0, -1, 2, 1, 33, 0, -1, 2, 1, 33};
    float *d_s;
    cudaMalloc(&d_s, sizeof(float) * 25);
    cudaMemcpy(d_s, s, sizeof(float) * 25, cudaMemcpyHostToDevice);
    dim3 dimGrid(6 / 2, 2);
    dim3 dimBlock(8 / 4, 4);
    relu<<<dimGrid, dimBlock>>>(d_s, 1, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(s, d_s, sizeof(float) * 25, cudaMemcpyDeviceToHost);
    // for (int p = 0; p < 25; p++)
    // {
    //     printf("&=%f\n", s[p]);
    // }
    return 0;
}