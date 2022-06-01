#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <cublas.h>
#define cycleTime 120
typedef struct
{
	VALUE_TYPE *value;
	int *columnindex;
	int *rowpointer;

} SMatrix;
__global__ void relu(VALUE_TYPE *d_C0_value, int mC, int nC)
{
	int i = blockIdx.x * threadIdx.x;
	d_C0_value[i] += -0.3f;
	if (d_C0_value[i] <= 0)
	{
		d_C0_value[i] = 0;
	}
	else if (d_C0_value[i] >= 32)
	{
		d_C0_value[i] = 32;
	}
}
int main(int argc, char **argv)
{
	struct timeval t1, t2, t3, t4;
	int size1 = 0;
	int size2 = 0;
	int *tc1;
	int *tc2;
	VALUE_TYPE bias = -0.3000;

	int mA;
	int nA;
	int nnzA;
	int isSymmetricA;
	SMatrix A;

	int mB;
	int nB;
	int nnzB;
	int isSymmetricB;
	SMatrix B[120];

	int mC, nC;
	int nnzC_golden = 0;

	// load A data from file
	gettimeofday(&t3, NULL);
	char filename1[] = "sparse-images-1024.tsv";
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
	A.value = (VALUE_TYPE *)malloc((nnzA) * sizeof(VALUE_TYPE));
	A.columnindex = (int *)malloc((nnzA) * sizeof(int));
	A.rowpointer = (int *)malloc((mA + 1) * sizeof(int));
	mmio_data(A.rowpointer, A.columnindex, A.value, filename1);
	printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
	VALUE_TYPE *A0_dense_value = (VALUE_TYPE *)malloc(mA * nA * sizeof(VALUE_TYPE));
	VALUE_TYPE *d_A0_dense_value;
	cudaMalloc(&d_A0_dense_value, mA * nA * sizeof(VALUE_TYPE));
	memset(A0_dense_value, 0, sizeof(VALUE_TYPE) * mA * nA);
	for (int i = 0; i < mA; i++)
	{
		for (int j = A.rowpointer[i]; j < A.rowpointer[i + 1]; j++)
		{
			A0_dense_value[i * nA + A.columnindex[j]] = A.value[j];
		}
	}
	VALUE_TYPE *A0_dense_value_T = (VALUE_TYPE *)malloc(mA * nA * sizeof(VALUE_TYPE));
	memset(A0_dense_value_T, 0, sizeof(VALUE_TYPE) * mA * nA);
	for (int x = 0; x < mA; x++)
	{
		for (int y = 0; y < nA; y++)
		{
			A0_dense_value_T[y * mA + x] = A0_dense_value[x + y * nA];
		}
	}
	// for (int adf = 0; adf < 1000; adf++)
	// {
	// 	if (A0_dense_value_T[adf] > 0)
	// 	{
	// 		printf("%f ", A0_dense_value_T[adf]);
	// 	}
	// }
	cudaMemcpy(d_A0_dense_value, A0_dense_value_T, mA * nA * sizeof(VALUE_TYPE),
			   cudaMemcpyHostToDevice);
	// float ssss[1000] = {1};
	// cudaMemcpy(ssss, d_A0_dense_value, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
	// for (int adf = 0; adf < 1000; adf++)
	// {
	// 	if (ssss[adf] > 0)
	// 	{
	// 		printf("a%f ", ssss[adf]);
	// 	}
	// }
	char neuronfile1[] = "neuron1024/n1024-l";
	char neuronfile2[] = ".tsv";
	char filename3[60];

	VALUE_TYPE *d_B_value[120];
	VALUE_TYPE *B_value[120];
	for (int k = 0; k < cycleTime; k++)
	{
		char filenum[5];
		int k1 = k + 1;
		snprintf(filenum, sizeof(filenum), "%d", k1);

		strcpy(filename3, neuronfile1);
		strcat(filename3, filenum);
		strcat(filename3, neuronfile2);

		mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
		B[k].value = (VALUE_TYPE *)malloc((nnzB) * sizeof(VALUE_TYPE));
		B[k].columnindex = (int *)malloc((nnzB) * sizeof(int));
		B[k].rowpointer = (int *)malloc((mB + 1) * sizeof(int));
		mmio_data(B[k].rowpointer, B[k].columnindex, B[k].value, filename3);

		B_value[k] = (VALUE_TYPE *)malloc(mB * nB * sizeof(VALUE_TYPE));
		memset(B_value[k], 0, sizeof(VALUE_TYPE) * mB * nB);
		for (int i = 0; i < mB; i++)
		{
			for (int j = B[k].rowpointer[i]; j < B[k].rowpointer[i + 1]; j++)
			{
				B_value[k][i * nB + B[k].columnindex[j]] = B[k].value[j];
			}
		}
		for (int x = 0; x < mB; x++)
		{
			for (int y = 0; y < x; y++)
			{
				VALUE_TYPE tmp;
				tmp = B_value[k][y * mB + x];
				B_value[k][y * mB + x] = B_value[k][x * mB + y];
				B_value[k][x * mB + y] = tmp;
			}
		}

		cudaMalloc(&d_B_value[k], sizeof(VALUE_TYPE) * mB * nB);
		cudaMemcpy(d_B_value[k], B_value[k], sizeof(VALUE_TYPE) * mB * nB,
				   cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
	}
	gettimeofday(&t4, NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load time: %f ms \n", time_load);

	mC = mA;
	nC = nB;

	VALUE_TYPE *d_C0_value;
	cudaMalloc(&d_C0_value, (mA * nA) * sizeof(VALUE_TYPE));

	gettimeofday(&t3, NULL);
	for (int k = 0; k < cycleTime; k++)
	{
		gettimeofday(&t1, NULL);
		cudaMemset(d_C0_value, 100, sizeof(VALUE_TYPE) * mC * nC);

		// TODO: calc c=a*b
		cublasHandle_t s;
		cublasCreate_v2(&s);
		VALUE_TYPE al = 1, ve = 0;

		// printf("sss%d\n", sdss);
		cublasSgemm_v2(s,
					   CUBLAS_OP_N, CUBLAS_OP_N,
					   mA, 1024, 1024,
					   &al,
					   d_A0_dense_value, mA,
					   d_B_value[k], mB,
					   &ve,
					   d_C0_value, mA);
		cudaDeviceSynchronize();
		// float sss[1024 * 1024] = {1.0};
		// cudaMemcpy(sss, d_C0_value, 1024 * 1024 * sizeof(float), cudaMemcpyDeviceToHost);

		// int sdss = 0;
		// for (int adf = 0; adf < 1024 * 1024; adf++)
		// {
		// 	if (sss[adf] > 0)
		// 	{
		// 		printf("x%fx\n", sss[adf]);
		// 	}
		// }
		gettimeofday(&t2, NULL);
		double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

		gettimeofday(&t1, NULL);
		dim3 dimBlock(nC, 1);
		dim3 dimGrid(mC, 1);
		relu<<<dimGrid, dimBlock>>>(d_C0_value, mC, nC);
		cudaDeviceSynchronize();
		gettimeofday(&t2, NULL);
		double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms\n",
			   k + 1, time_gemm, time_biasrelu);

		cudaMemcpy(d_A0_dense_value, d_C0_value, (mC * nC) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);
	}

	gettimeofday(&t4, NULL);
	double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n", time_inference);

	VALUE_TYPE *A0 = (VALUE_TYPE *)malloc(60000 * 1024 * sizeof(VALUE_TYPE));
	cudaMemcpy(A0, d_A0_dense_value, 60000 * 1024 * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
	// TODO: 转置
	//  check results
	printf("test\n");
	FILE *fs;
	fs = fopen("sparse-images-1024-1.tsv", "w+");
	for (int i = 0; i < mA; i++)
	{
		int sum = 0;
		for (int j = (i * nA); j < ((i + 1) * nA); j++)
		{
			sum += A0[j];
		}
		// printf("s%d\n", sum);
		if (sum != 0)
		{
			fprintf(fs, "%d\n", i + 1);
		}
	}
	fclose(fs);
	FILE *fp2 = NULL;

	fp2 = fopen("sparse-images-1024-1.tsv", "rb");
	if (fp2 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp2, 0, SEEK_END);
	size2 = ftell(fp2);
	rewind(fp2);

	tc2 = (int *)malloc(sizeof(int) * size2 / 4);

	int readnum2 = fread(tc2, 4, size2 / 4, fp2);

	fclose(fp2);

	FILE *fp1;

	fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
	if (fp1 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp1, 0, SEEK_END);
	size1 = ftell(fp1);
	rewind(fp1);

	tc1 = (int *)malloc(sizeof(int) * size1 / 4);

	int readnum1 = fread(tc1, 4, size1 / 4, fp1);

	fclose(fp1);
	int judge = 0;
	for (int i = 0; i < size1 / 4; i++)
	{
		if (tc1[i] - tc2[i] != 0)
		{
			judge++;
		}
	}
	printf("judge:%d\n", judge);
	if (judge == 0)
	{
		printf("CHALLENGE PASSED\n");
	}
	else
	{
		printf("CHALLENGE FAILED\n");
	}

	free(A0);

	return 0;
}
