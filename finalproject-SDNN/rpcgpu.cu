#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <cublas.h>
#include <omp.h>

#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define cycleTime 120
#define SPLIT_BLOCK 100
#define SPLIT_THREAD 256
#define BIAS -0.3
#define BATCH_SIZE 60000
enum REQ_MSG_TYPE
{
	CONNECT = -1,
	REQUEST_TASK = -2,
	REPORT_RES = -3,
};

enum MSG_SYMBOL
{
	MSG_END = -4,
	END_CALC = -5,
};
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
__global__ void relu(VALUE_TYPE *d_C0_value)
{
	int i = (blockIdx.x * SPLIT_BLOCK + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * SPLIT_THREAD + threadIdx.y;
	VALUE_TYPE tmp = BIAS + d_C0_value[i];
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
		  VALUE_TYPE **d_C0_value, int mC, int nC,
		  VALUE_TYPE **d_A0_dense_value, VALUE_TYPE **d_B_value, int mB, int cycleTime_var, int *requestTask, int *taskId, timeval *t5)
{
	VALUE_TYPE al = 1, ve = 0;
	dim3 dimGrid(mC / SPLIT_BLOCK, SPLIT_BLOCK);
	dim3 dimBlock(nC / SPLIT_THREAD, SPLIT_THREAD);
	int nowTaskId;
	int firstCalc = 1;
	while (*taskId != END_CALC)
	{
		nowTaskId = *taskId;
		if (nowTaskId == -1)
		{
			continue;
		}
		if (firstCalc && nowTaskId >= 0)
		{
			gettimeofday(t5, NULL);
			firstCalc = 0;
		}
		cublasHandle_t s;
		cublasCreate_v2(&s);
		float *from;
		float *to;
		for (int k = 0; k < cycleTime_var; k++)
		{
			gettimeofday(&t1, NULL);
			// calc c=a*b

			if (k % 2 == 0)
			{
				from = d_A0_dense_value[nowTaskId];
				to = d_C0_value[nowTaskId];
			}
			else
			{
				from = d_C0_value[nowTaskId];
				to = d_A0_dense_value[nowTaskId];
			}
			cublasSgemm_v2(s,
						   CUBLAS_OP_N, CUBLAS_OP_N,
						   BATCH_SIZE, 1024, 1024,
						   &al,
						   from, BATCH_SIZE,
						   d_B_value[k], mB,
						   &ve,
						   to, BATCH_SIZE);
			cudaDeviceSynchronize();

			gettimeofday(&t2, NULL);
			double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

			gettimeofday(&t1, NULL);
			relu<<<dimGrid, dimBlock>>>(to);
			cudaDeviceSynchronize();
			gettimeofday(&t2, NULL);
			double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
			printf("Taskid = %d,k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms\n", nowTaskId,
				   k + 1, time_gemm, time_biasrelu);
			if (k == 115)
			{
				*requestTask = 1;
			}
		}
		cublasDestroy_v2(s);
		if (cycleTime_var != 120)
		{
			return;
		}
	}
}

void comm(int *requestTask, int sock, int *taskId, int *taskList)
{
	int req[] = {REQUEST_TASK};
	int taskListPtr = 0;
	while (*taskId != END_CALC)
	{
		if (*requestTask == 1)
		{
			send(sock, req, sizeof(req), 0);
			read(sock, taskId, sizeof(*taskId));
			taskList[taskListPtr++] = *taskId;
			*requestTask = 0;
		}
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

	int sock = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in serv_addr;
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = inet_addr("192.168.1.253");
	serv_addr.sin_port = htons(1234);
	connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

	int clientId;
	int msgs[] = {CONNECT};

	send(sock, msgs, sizeof(msgs), 0);
	read(sock, &clientId, sizeof(clientId));
	printf("this clientid %d\n", clientId);
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
	}
	mC = BATCH_SIZE;
	nC = nB;
	// warm up
	printf("---------warm up------------\n");
	int fakeReq = 0;
	struct timeval faket5;
	calc(t1, t2,
		 d_C0_split_value, mC, nC,
		 d_A0_dense_split_value, d_B_value, mB, 10, &fakeReq, &fakeReq, &faket5);
	//清空d_a0
	cudaMemcpy(d_A0_dense_split_value[0], tmp, BATCH_SIZE * nA * sizeof(VALUE_TYPE),
			   cudaMemcpyHostToDevice);
	printf("---------warm up end------------\n");
	gettimeofday(&t4, NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load and warm up time: %f ms \n", time_load);

	int taskId = -1;
	int requestTask = 1;
	int *taskIdList = (int *)malloc(60000 / BATCH_SIZE * 4);
	struct timeval t5;
#pragma omp parallel num_threads(2) shared(requestTask, taskId, t1, t2, d_C0_split_value, d_A0_dense_split_value, mC, nC, d_B_value, mB)
	{
#pragma omp single
		{
#pragma omp task
			comm(&requestTask, sock, &taskId, taskIdList);
#pragma omp task
			calc(t1, t2,
				 d_C0_split_value, mC, nC,
				 d_A0_dense_split_value, d_B_value, mB, cycleTime, &requestTask, &taskId, &t5);
		}
	}

	gettimeofday(&t4, NULL);
	double time_inference = (t4.tv_sec - t5.tv_sec) * 1000.0 + (t4.tv_usec - t5.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n", time_inference);

	// TODO: upload
	int currentTaskIdPtr = 0;
	VALUE_TYPE *tmp2 = (VALUE_TYPE *)malloc(BATCH_SIZE * 1024 * sizeof(VALUE_TYPE));
	int *tmp1 = (int *)malloc((BATCH_SIZE * 1024 + 2) * sizeof(VALUE_TYPE));

	while (1)
	{
		int currentTaskId = taskIdList[currentTaskIdPtr++];
		if (currentTaskId == END_CALC)
		{
			break;
		}
		int resLength = 2;
		memset(tmp1, 0, (BATCH_SIZE * 1024 + 2) * sizeof(VALUE_TYPE));
		cudaMemcpy(tmp2, d_A0_dense_split_value[currentTaskId], BATCH_SIZE * 1024 * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
		toRowIndx_(BATCH_SIZE, 1024, tmp2);

		for (int p = 0; p < BATCH_SIZE; p++)
		{
			int rowSum = 0;
			for (int q = 0; q < 1024; q++)
			{
				rowSum += tmp2[p * 1024 + q];
			}
			if (rowSum != 0)
			{
				tmp1[resLength++] = p + 1 + currentTaskId * BATCH_SIZE;
			}
		}

		tmp1[0] = REPORT_RES;
		tmp1[1] = currentTaskId;
		tmp1[resLength] = MSG_END;

		printf("sending,%d,%d\n", tmp1[resLength], resLength);
		send(sock, tmp1, (resLength + 1) * sizeof(VALUE_TYPE), 0);
		printf("sent\n");
		usleep(100 * 1000);
	}
	return 0;
}
