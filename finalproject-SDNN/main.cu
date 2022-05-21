#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
typedef struct 
{
	VALUE_TYPE *value;
	int *columnindex;
	int *rowpointer;

}SMatrix;

int main(int argc, char ** argv)
{
	struct timeval t1, t2, t3, t4;
	int size1 = 0;
	int size2 = 0;
	int *tc1;
	int *tc2;
	double bias = -0.3000;

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
	
	int mC,nC;
	int nnzC_golden = 0;

    // load A data from file
	gettimeofday(&t3, NULL);
	char filename1[]="sparse-images-1024.tsv";
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
	A.value=(VALUE_TYPE*)malloc((nnzA)*sizeof(VALUE_TYPE));
	A.columnindex=(int*)malloc((nnzA)*sizeof(int));
	A.rowpointer=(int*)malloc((mA+1)*sizeof(int));
	mmio_data(A.rowpointer, A.columnindex, A.value, filename1);
	printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
	
	int *d_A_rowpointer, *d_A_columnindex;

    cudaMalloc(&d_A_rowpointer, (mA+1)*sizeof(int));
    cudaMemcpy(d_A_rowpointer, A.rowpointer, (mA+1)*sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_A_columnindex, (nnzA)*sizeof(int));
    cudaMemcpy(d_A_columnindex, A.columnindex, (nnzA)*sizeof(int),
               cudaMemcpyHostToDevice);

	float *d_A_value;
	cudaMalloc(&d_A_value, (nnzA)*sizeof(VALUE_TYPE));
	cudaMemcpy(d_A_value, A.value, (nnzA)*sizeof(VALUE_TYPE),
			   cudaMemcpyHostToDevice);


    cusparseSpMatDescr_t d_csr_A;
    cusparseCreateCsr(&d_csr_A, (int64_t) mA, (int64_t) nA,
                      nnzA, d_A_rowpointer, d_A_columnindex, d_A_value,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F
    );

	char neuronfile1[] = "neuron1024/n1024-l";
	char neuronfile2[] = ".tsv";
	char filename3[60];
	
	cusparseSpMatDescr_t B0[120];
	for (int k = 0; k < 120; k++) 
	{	
		char filenum[5];
		int k1=k+1;
		snprintf(filenum,sizeof(filenum),"%d",k1);
		
		strcpy(filename3, neuronfile1);
		strcat(filename3, filenum);
		strcat(filename3, neuronfile2);

		mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
		B[k].value=(VALUE_TYPE*)malloc((nnzB)*sizeof(VALUE_TYPE));
		B[k].columnindex=(int*)malloc((nnzB)*sizeof(int));
		B[k].rowpointer=(int*)malloc((mB+1)*sizeof(int));
		mmio_data(B[k].rowpointer, B[k].columnindex, B[k].value, filename3);
		
		int *d_Bk_rowpointer, *d_Bk_columnindex;

		cudaMalloc(&d_Bk_rowpointer, (mB+1)*sizeof(int));
		cudaMemcpy(d_Bk_rowpointer, B[k].rowpointer, (mB+1)*sizeof(int),
				cudaMemcpyHostToDevice);

		cudaMalloc(&d_Bk_columnindex, (nnzB)*sizeof(int));
		cudaMemcpy(d_Bk_columnindex, B[k].columnindex, (nnzB)*sizeof(int),
				cudaMemcpyHostToDevice);

		VALUE_TYPE *d_Bk_value;
		cudaMalloc(&d_Bk_value, (nnzB)*sizeof(VALUE_TYPE));
		cudaMemcpy(d_Bk_value, B[k].value, (nnzB)*sizeof(VALUE_TYPE),
				cudaMemcpyHostToDevice);


		cusparseCreateCsr(&B0[k], (int64_t) mB, (int64_t) nB,
						nnzB, d_Bk_rowpointer, d_Bk_columnindex, d_Bk_value,
						CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
						CUSPARSE_INDEX_BASE_ZERO,
						CUDA_R_32F
		);

	}
	gettimeofday(&t4,NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load time: %f ms \n",time_load);

    mC = mA;
	nC = nB;

	int *d_C0_rowpointer, *d_C0_columnindex;

    cudaMalloc(&d_C0_rowpointer, (mC+1)*sizeof(int));

    cudaMalloc(&d_C0_columnindex, (60000*1024)*sizeof(int));

	float *d_C0_value;
	cudaMalloc(&d_C0_value, (60000*1024)*sizeof(VALUE_TYPE));


    cusparseSpMatDescr_t d_csr_C0;
    cusparseCreateCsr(&d_csr_C0, (int64_t) mC, (int64_t) nC,
                      60000*1024, d_C0_rowpointer, d_C0_columnindex, d_C0_value,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F
    );

    gettimeofday(&t3, NULL);
	for (int k = 0; k < 1; k++) 
	{
		int k1=k+1;
        cudaMemset(d_C0_value, 0, sizeof(VALUE_TYPE)*mC*nC);

		gettimeofday(&t1, NULL);
		
		cusparseHandle_t handle;
		cusparseCreate(&handle);
		cusparseOperation_t Ap = CUSPARSE_OPERATION_NON_TRANSPOSE;
		cusparseOperation_t Bp = CUSPARSE_OPERATION_NON_TRANSPOSE;
		VALUE_TYPE al = 1, be = 0;
		
        cusparseSpMM(handle, Ap, Bp, &al, d_csr_A, &B0[k], &be, d_csr_C0,
                     CUDA_R_64F, CUSPARSE_MM_ALG_DEFAULT, NULL);
        cudaDeviceSynchronize();

		gettimeofday(&t2,NULL);
        double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		
		gettimeofday(&t1, NULL);
		
		// for (int i = 0; i < mC*nC; i++) 
		// {
		// 	// C0[i] += bias;
			
		// 	if (C0[i] <= 0)
		// 	{   
		// 		C0[i] = 0;
		// 	}
		// 	else if (C0[i] >= 32) 
		// 	{
		// 		C0[i] = 32;
		// 	}
		// }
		gettimeofday(&t2,NULL);
        double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms\n", 
		       k+1, time_gemm, time_biasrelu);

		
		//cudaMemcpy(A0, C0, (mC*nC)*sizeof(VALUE_TYPE));
	}
	
	gettimeofday(&t4,NULL);
	double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n",time_inference);
	
	// free(C0);
	
    // // check results
	// printf("test\n");
	// FILE* fs;
	// fs=fopen("sparse-images-1024-1.tsv","w+");
	// for (int i = 0; i <mA; i++)
	// {	
	// 	int sum =0;
	// 	for (int j = (i*nA); j < ((i+1)*nA); j++)
	// 	{  
	// 		sum+=A0[j];
		
	// 	}
	// 	if(sum!=0)
	// 	{
	// 		fprintf(fs,"%d\n", i+1);
	// 	}
	// }
	// fclose(fs);
	// FILE* fp2=NULL;

	// fp2 = fopen("sparse-images-1024-1.tsv", "rb");
	// if (fp2 == NULL)
	// {
	// 	printf("Error:Open file fail!\n");
	// }

	// fseek(fp2, 0, SEEK_END);
	// size2 = ftell(fp2);
	// rewind(fp2);

	// tc2 = (int*)malloc(sizeof(int) * size2/4);

	// int readnum2 = fread(tc2, 4, size2/4, fp2);

	// fclose(fp2);
	
	// FILE* fp1;

	// fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
	// if (fp1 == NULL)
	// {
	// 	printf("Error:Open file fail!\n");
	// }

	// fseek(fp1, 0, SEEK_END);
	// size1 = ftell(fp1);
	// rewind(fp1);

	// tc1 = (int*)malloc(sizeof(int) * size1/4);

	// int readnum1 = fread(tc1, 4, size1/4, fp1);

	// fclose(fp1);
	// int judge=0;
	// for(int i=0;i<size1/4;i++)
	// {
	// 	if(tc1[i]-tc2[i] != 0)
	// 	{
	// 		judge++;
	// 	}
	// }
	// printf("judge:%d\n",judge);
	// if (judge == 0) {
	// 	printf("CHALLENGE PASSED\n");
	// }
	// else
	// {
	// 	printf("CHALLENGE FAILED\n");
	// }

	// free(A0);

	return 0;
}

		
