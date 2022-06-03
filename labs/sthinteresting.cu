#define cycleTime 120
#define SPLIT_BLOCK 100
#define SPLIT_THREAD 256
#define CPU_SPLIT 59000
#define BIAS -0.3
#define VALUE_TYPE float
// __global__ void relu(VALUE_TYPE *d_C0_value, int mC, int nC)
// {
//     int i = (blockIdx.x * SPLIT_BLOCK + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * SPLIT_THREAD + threadIdx.y;
//     VALUE_TYPE tmp = BIAS;
//     tmp += d_C0_value[i];
//     if (tmp <= 0)
//     {
//         tmp = 0;
//     }
//     else if (tmp >= 32)
//     {
//         tmp = 32;
//     }
//     d_C0_value[i] = tmp;
// }
int numLogicalCpus(int i)
{
    return i + 22;
}
