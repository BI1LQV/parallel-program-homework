#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <cublas.h>
#include <openblas_config.h>
#include <generated/cblas.h>
#include <omp.h>

#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cusparse.h>
#define cycleTime 120
#define SPLIT_BLOCK 100
#define SPLIT_THREAD 256
#define CPU_SPLIT 60000
#define BIAS -0.3
#define BATCH_SIZE 60000
typedef struct
{
	VALUE_TYPE *value;
	int *columnindex;
	int *rowpointer;

} SMatrix;
void toColIndx_(int line, int ld, VALUE_TYPE *val)
{
	VALUE_TYPE *temp = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * line * ld);

	for (int i = 0; i < ld; ++i)
	{
		for (int j = 0; j < line; ++j)
		{
			temp[i * line + j] = val[j * ld + i];
		}
	}
	memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
	free(temp);
}

void toRowIndx_(int line, int ld, VALUE_TYPE *val)
{
	VALUE_TYPE *temp = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * line * ld);

	for (int i = 0; i < line; ++i)
	{
		for (int j = 0; j < ld; ++j)
		{
			temp[i * ld + j] = val[j * line + i];
		}
	}
	memcpy(val, temp, sizeof(VALUE_TYPE) * line * ld);
	free(temp);
}
__global__ void relu(VALUE_TYPE *d_C0_value, int mC, int nC)
{
	int i = (blockIdx.x * SPLIT_BLOCK + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * SPLIT_THREAD + threadIdx.y;
	VALUE_TYPE tmp = BIAS;
	tmp += d_C0_value[i];
	if (tmp <= 0)
	{
		tmp = 0;
	}
	else if (tmp >= 32)
	{
		tmp = 32;
	}
	d_C0_value[i] = tmp;
}
void calc(timeval t1, timeval t2,
		  VALUE_TYPE *d_C0_value, int mC, int nC,
		  VALUE_TYPE *d_A0_dense_value, VALUE_TYPE **d_B_value, int mB, int cycleTime_var,
		  VALUE_TYPE **B_csc_value, int **B_csc_rowIdx, int **B_csc_colPtr, int *b_csc_nnz)
{
	VALUE_TYPE al = 1, ve = 0;
	dim3 dimGrid(mC / SPLIT_BLOCK, SPLIT_BLOCK);
	dim3 dimBlock(nC / SPLIT_THREAD, SPLIT_THREAD);
	for (int k = 0; k < cycleTime_var; k++)
	{
		gettimeofday(&t1, NULL);
		// calc c=a*b
		// cublasHandle_t s;
		// cublasCreate_v2(&s);

		// cublasSgemm_v2(s,
		// 			   CUBLAS_OP_N, CUBLAS_OP_N,
		// 			   BATCH_SIZE, 1024, 1024,
		// 			   &al,
		// 			   d_A0_dense_value, BATCH_SIZE,
		// 			   d_B_value[k], mB,
		// 			   &ve,
		// 			   d_C0_value, BATCH_SIZE);
		cusparseHandle_t handle;
		cusparseCreate(&handle);
		float a = 1;
		float b = 0;

		cusparseSgemmi(handle,
					   BATCH_SIZE,
					   1024,
					   1024,
					   b_csc_nnz[k],
					   &a,
					   d_A0_dense_value,
					   BATCH_SIZE,
					   B_csc_value[k],
					   B_csc_colPtr[k],
					   B_csc_rowIdx[k],
					   &b,
					   d_C0_value,
					   BATCH_SIZE);
		cudaDeviceSynchronize();

		gettimeofday(&t2, NULL);
		double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

		gettimeofday(&t1, NULL);

		relu<<<dimGrid, dimBlock>>>(d_C0_value, mC, nC);
		cudaDeviceSynchronize();
		gettimeofday(&t2, NULL);
		double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms\n",
			   k + 1, time_gemm, time_biasrelu);

		cudaMemcpy(d_A0_dense_value, d_C0_value, (BATCH_SIZE * nC) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice);
	}
}

int main(int argc, char **argv)
{
	struct timeval t1, t2, t3, t4;
	int size1 = 0;
	int size2 = 0;
	int *tc1;
	int *tc2;

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

	// int sock = socket(AF_INET, SOCK_STREAM, 0);
	// struct sockaddr_in serv_addr;
	// memset(&serv_addr, 0, sizeof(serv_addr));
	// serv_addr.sin_family = AF_INET;
	// serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	// serv_addr.sin_port = htons(1234);
	// connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

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

	memset(A0_dense_value, 0, sizeof(VALUE_TYPE) * mA * nA);
	for (int i = 0; i < mA; i++)
	{
		for (int j = A.rowpointer[i]; j < A.rowpointer[i + 1]; j++)
		{
			A0_dense_value[i * nA + A.columnindex[j]] = A.value[j];
		}
	}
	VALUE_TYPE *d_A0_dense_split_value[60000 / BATCH_SIZE];
	VALUE_TYPE *d_C0_split_value[60000 / BATCH_SIZE];
	VALUE_TYPE *tmp = (VALUE_TYPE *)malloc(BATCH_SIZE * nA * sizeof(VALUE_TYPE));
	for (int splice = (60000 / BATCH_SIZE) - 1; splice >= 0; splice--)
	{
		cudaMalloc(&d_A0_dense_split_value[splice], (BATCH_SIZE * nA) * sizeof(VALUE_TYPE));
		cudaMalloc(&d_C0_split_value[splice], (BATCH_SIZE * nA) * sizeof(VALUE_TYPE));
		cudaMemcpy(tmp, A0_dense_value + splice * (BATCH_SIZE * nA), (BATCH_SIZE * nA) * sizeof(VALUE_TYPE), cudaMemcpyHostToHost);
		toColIndx_(BATCH_SIZE, 1024, tmp);
		cudaMemcpy(d_A0_dense_split_value[splice], tmp, (BATCH_SIZE * nA) * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
	}

	char neuronfile1[] = "neuron1024/n1024-l";
	char neuronfile2[] = ".tsv";
	char filename3[60];

	VALUE_TYPE *d_B_value[120];
	VALUE_TYPE *B_value[120];
	VALUE_TYPE *B_csc_value[120];
	int *B_csc_rowIdx[120];
	int *B_csc_colPtr[120];
	int *B_csc_nnz = (int *)malloc(120 * sizeof(int));

	for (int k = 0; k < cycleTime; k++)
	{
		char filenum[5];
		int k1 = k + 1;
		snprintf(filenum, sizeof(filenum), "%d", k1);

		strcpy(filename3, neuronfile1);
		strcat(filename3, filenum);
		strcat(filename3, neuronfile2);

		mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
		B_csc_nnz[k] = nnzB;
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
		cudaMalloc(&d_B_value[k], sizeof(VALUE_TYPE) * mB * nB);
		cudaMemcpy(d_B_value[k], B_value[k], sizeof(VALUE_TYPE) * mB * nB,
				   cudaMemcpyHostToDevice);

		cudaMalloc(&B_csc_value[k], (nnzB) * sizeof(VALUE_TYPE));
		cudaMalloc(&B_csc_rowIdx[k], (nnzB) * sizeof(int));
		cudaMalloc(&B_csc_colPtr[k], (mB + 1) * sizeof(int));

		int dataNumInCol = 0;
		int B_csc_value_idx = 0;
		int B_csc_rowIdx_idx = 0;
		int B_csc_colPtr_idx = 0;
		float *B_csc_value_tmp = (VALUE_TYPE *)malloc((nnzB) * sizeof(VALUE_TYPE));
		int *B_csc_rowIdx_tmp = (int *)malloc((nnzB) * sizeof(int));
		int *B_csc_colPtr_tmp = (int *)malloc((mB + 1) * sizeof(int));
		B_csc_colPtr_tmp[0] = 0;
		for (int colIdx = 0; colIdx < 1024; colIdx++)
		{
			for (int rowIdx = 0; rowIdx < 1024; rowIdx++)
			{
				if (B_value[k][rowIdx * 1024 + colIdx])
				{
					B_csc_value_tmp[B_csc_value_idx++] = B_value[k][rowIdx * 1024 + colIdx];
					dataNumInCol++;
					B_csc_rowIdx_tmp[B_csc_rowIdx_idx++] = rowIdx;
				}
			}
			B_csc_colPtr_tmp[B_csc_colPtr_idx + 1] = B_csc_colPtr_tmp[B_csc_colPtr_idx] + dataNumInCol;
			B_csc_colPtr_idx++;
			dataNumInCol = 0;
		}
		cudaMemcpy(B_csc_value[k], B_csc_value_tmp, (nnzB) * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
		cudaMemcpy(B_csc_rowIdx[k], B_csc_rowIdx_tmp, (nnzB) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(B_csc_colPtr[k], B_csc_colPtr_tmp, (mB + 1) * sizeof(int), cudaMemcpyHostToDevice);

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
	}

	mC = BATCH_SIZE;
	nC = nB;
	// warm up
	printf("---------warm up------------\n");
	calc(t1, t2,
		 d_C0_split_value[0], mC, nC,
		 d_A0_dense_split_value[0], d_B_value, mB, 5,
		 B_csc_value, B_csc_rowIdx, B_csc_colPtr, B_csc_nnz);
	//清空d_a0
	cudaMemcpy(d_A0_dense_split_value[0], tmp, BATCH_SIZE * nA * sizeof(VALUE_TYPE),
			   cudaMemcpyHostToDevice);
	printf("---------warm up end------------\n");
	gettimeofday(&t4, NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load and warm up time: %f ms \n", time_load);

	gettimeofday(&t3, NULL);
	for (int st = 0; st < 60000 / BATCH_SIZE; st++)
	{
		calc(t1, t2,
			 d_C0_split_value[st], mC, nC,
			 d_A0_dense_split_value[st], d_B_value, mB, cycleTime,
			 B_csc_value, B_csc_rowIdx, B_csc_colPtr, B_csc_nnz);
	}

	gettimeofday(&t4, NULL);
	double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n", time_inference);

	VALUE_TYPE *A0 = (VALUE_TYPE *)malloc(CPU_SPLIT * 1024 * sizeof(VALUE_TYPE));
	VALUE_TYPE *tmp2 = (VALUE_TYPE *)malloc(BATCH_SIZE * 1024 * sizeof(VALUE_TYPE));
	for (int splice = 0; splice < 60000 / BATCH_SIZE; splice++)
	{
		cudaMemcpy(tmp2, d_A0_dense_split_value[splice], BATCH_SIZE * 1024 * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
		toRowIndx_(BATCH_SIZE, 1024, tmp2);
		cudaMemcpy(A0 + splice * BATCH_SIZE * 1024, tmp2, BATCH_SIZE * 1024 * sizeof(VALUE_TYPE), cudaMemcpyHostToHost);
	}

	//  check results
	// TODO: upload
	printf("test\n");
	FILE *fs;
	fs = fopen("sparse-images-1024-1.tsv", "w+");
	int i = 0;
	for (; i < CPU_SPLIT; i++)
	{
		int sum = 0;
		for (int j = (i * nA); j < ((i + 1) * nA); j++)
		{
			sum += A0[j];
		}

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
