// Parallel Programming (English), Spring 2021
// SUMMA GEMV
// Wrote by Yingfeng Chen 2012
// Update-by Yuxiang Gao 2021 Spring, China University of Petroleum-Beijing
// Update-by Wenhao Dai 2022 Spring, China University of Petroleum-Beijing
//
// Homework 5.	This version has a low performance
//		and a strict rule for parameters
//		try to optimize by using MPI 
// ENV suggest：mpicc for mpich version 3.3.2
//             gcc version 9.4.0
// 


/**
summa:
	mpicc -g -Wall -O3 -fopenmp -std=c11 main.c -o summa -lm
run:
	mpirun -np 16 ./summa 16 256 256 256
    
    eg: mpirun -np num1 ./summa num2 num3 num4 num5
    (Attention the num1 & num2 must be same !!!)

To use Makefile:
    make       : clean the old summa file
    make summa : compile main.c and generate summa
    make run   : run file'summa' with input parameter(more information,read Makefile)
*/



#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
 
 
void PrintMatrixForVector(int * matrix,int high,int len)
{
    int i;
    for(i=0;i<high*len;i++)
    {
        printf("%6d  ",matrix[i]);
        if(i%len==len-1&&i!=0)
			printf("\n");
    }
}
 

void MatrixMultiply(int * A,int *B,int *C,unsigned m,unsigned n,unsigned p)
{   int i,j,k;
    /*printf("A: \n");
	PrintMatrixForVector(A,m,n);
	printf("B: \n");
	PrintMatrixForVector(B,n,p);*/
    for(i=0;i<m;i++)
       for(j=0;j<p;j++)    
	   {
		   int result=0; 
		   for(k=0;k<n;k++)
            {
               result=A[i*n+k]*B[k*p+j]+result;  
            }
            C[i*p+j]=result;
	   }
}
 

void MatrixAdd(int * A,int *B,unsigned m,unsigned n) 
{  int i,j;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
        {
            A[i*n+j]=A[i*n+j]+B[i*n+j];
        }
}
 
void PrintMatrix(int ** matrix,int high,int len)
{
    int i,j;
    for(i=0;i<high;i++)
    {
        for(j=0;j<len;j++)
        {
            printf("%6d  ",matrix[i][j]);
        }
        printf("\n");
    }
}
 
/****rand the  data to compute****/
void RandomMatrix(int *matrix,int len)
{
   struct timeval tpstart;  
   gettimeofday(&tpstart,NULL);
   srand(tpstart.tv_usec);
   int i=0;
   for(i=0;i<len;i++)
   matrix[i]=rand()%8;
}


int main(int argc,char **argv)
{
	int rank;
    MPI_Status status;
    MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
	int nodeNum;   //num of num : should be like 4 (can be sqrt)
	
	int matrixHighA;
	int matrixLenA; 
	
	int matrixHighB;
    int matrixLenB;

	if(argc!=5&&rank==0)
	{
		 printf("The para is wrong!using default para\n");
 
		 nodeNum=4;
		 matrixHighA=6;
		 matrixLenA=8;
		 matrixHighB=8;
		 matrixLenB=10;
	}
	else
	{
		nodeNum=atoi(argv[1]);
		matrixHighA=atoi(argv[2]);
		matrixLenA=atoi(argv[3]);
		matrixHighB=atoi(argv[3]);
        matrixLenB=atoi(argv[4]);
	}
    int p=sqrt(nodeNum);    
 
    
    int localHighA=matrixHighA/p;
    int localLenA=matrixLenA/p;

    int localHighB=matrixHighB/p;
    int localLenB=matrixLenB/p;
      
    int i;
    int j;
    int k;
    int l;
 
    int * A=(int *)malloc(localLenA*localHighA*sizeof(int));
    RandomMatrix(A,localHighA*localLenA);

    int * B=(int *)malloc(localLenB*localHighB*sizeof(int));
    RandomMatrix(B,localHighB*localLenB);
 
    int * C=(int *)malloc(localHighA*localLenB*sizeof(int));
    for(i=0;i<localHighA*localLenB;i++)C[i]=0;
 
   /* printf("%d local  A:\n",rank);
    PrintMatrixForVector(A,localHighA,localLenA);
    printf("%d local  B:\n",rank);
    PrintMatrixForVector(B,localHighB,localLenB);*/

	int myRow=rank/p;
	int myCol=rank%p;

    //send data to rank(0) :collect and display 
    if(rank!=0){
        MPI_Send(A,localHighA*localLenA,MPI_INT,0,rank+100,MPI_COMM_WORLD);
        MPI_Send(B,localHighB*localLenB,MPI_INT,0,rank+200,MPI_COMM_WORLD);
    }
    
 
	if(rank==0)
	{   
        //printf("When rank = 0\n");

		int **matrixA=(int **)malloc(matrixHighA*sizeof(int *));
		for (i=0;i<matrixHighA;i++)
			matrixA[i]=(int *)malloc(matrixLenA*sizeof(int));
 
		int **matrixB=(int **)malloc(matrixHighB*sizeof(int *));
		for (i=0;i<matrixHighB;i++)
			matrixB[i]=(int *)malloc(matrixLenB*sizeof(int));
        

        for(i=0;i<nodeNum;i++)  //except the rank == 0;
        {
            int *receiveATemp=(int *)malloc(localLenA*localHighA*sizeof(int));
            int *receiveBTemp=(int *)malloc(localLenB*localHighB*sizeof(int));

            if(i!=0)
            {
                MPI_Recv(receiveATemp,localHighA*localLenA,MPI_INT,i,i+100,MPI_COMM_WORLD,&status);
                MPI_Recv(receiveBTemp,localHighB*localLenB,MPI_INT,i,i+200,MPI_COMM_WORLD,&status);
            }
            
            else  //i == 0
            {
                for(int ii=0;ii<localHighA*localLenA;ii++){
                    receiveATemp[ii]=A[ii];
                }
                for(int ii=0;ii<localHighB*localLenB;ii++){
                    receiveBTemp[ii]=B[ii];
                }
            }

            l=0;
            for(j=0;j<localHighA;j++)
                for(k=0;k<localLenA;k++){
                    matrixA[j+(int)(i/p)*localHighA][k+(int)(i%p)*localLenA]=receiveATemp[l++];
                } 
            l=0;
            for(j=0;j<localHighB;j++)
                for(k=0;k<localLenB;k++){
                    matrixB[j+(int)(i/p)*localHighB][k+(int)(i%p)*localLenB]=receiveBTemp[l++];
                }
 
            free(receiveATemp);
            free(receiveBTemp);
        }
 
        printf("A:\n");
        PrintMatrix(matrixA,matrixHighA,matrixLenA);
        printf("B:\n");
        PrintMatrix(matrixB,matrixHighB,matrixLenB);
 
        for (i=0;i<matrixHighA;i++)
            free(matrixA[i]);
        for (i=0;i<matrixHighB;i++)
            free(matrixB[i]);
 
        free(matrixA);
        free(matrixB);
	}

    MPI_Request  request1;

    for(i=0;i<p;i++)//send part data 
	{
	    //if(myCol!=i)
		{
			MPI_Isend(A,localHighA*localLenA,MPI_INT,myRow*p+i,1,MPI_COMM_WORLD,&request1);
			MPI_Isend(B,localHighB*localLenB,MPI_INT,myRow*p+i,2,MPI_COMM_WORLD,&request1);
		}
	    //if(myRow!=i)
		{
		   MPI_Isend(A,localHighA*localLenA,MPI_INT,i*p+myCol,1,MPI_COMM_WORLD,&request1);
		   MPI_Isend(B,localHighB*localLenB,MPI_INT,i*p+myCol,2,MPI_COMM_WORLD,&request1);
		}
	}
	int *receiveA=(int *)malloc(localLenA*localHighA*sizeof(int));
	int *receiveB=(int *)malloc(localLenB*localHighB*sizeof(int));
    int *resultC= (int *)malloc(localHighA*localLenB*sizeof(int));
	for(i=0;i<localHighA*localLenB;i++)resultC[i]=0;

 
/*********************compute data of matrix *****************************/
	for(i=0;i<p;i++)
	{
        if(i!= 0)
        {
            MPI_Recv(receiveA,localHighA*localLenA,MPI_INT,myRow*p+i,1,MPI_COMM_WORLD,&status);
		    MPI_Recv(receiveB,localHighB*localLenB,MPI_INT,i*p+myCol,2,MPI_COMM_WORLD,&status);
        }

        else
        {
            for(int ii=0;ii<localHighA*localLenA;ii++){
                    receiveA[ii]=A[ii];
                }
                for(int ii=0;ii<localHighB*localLenB;ii++){
                    receiveB[ii]=B[ii];
                }
        }

        MatrixMultiply(receiveA,receiveB,resultC,localHighA,localLenA,localLenB);
	/*	if(rank==0)
		{
			printf("%d  A:\n",i);
			PrintMatrixForVector(receiveA,localHighA,localLenA);
			printf("%d  B:\n",i);
			PrintMatrixForVector(receiveB,localHighB,localLenB);
			printf("%d  C:\n",i);
			PrintMatrixForVector(resultC,localHighA,localLenB);
			
		}*/
		MatrixAdd(C,resultC,localHighA,localLenB);
	}
 

    
    if(rank!= 0)
        MPI_Send(C,localHighA*localLenB,MPI_INT,0,rank+400,MPI_COMM_WORLD);
    


    if(rank==0)
    {
        int **matrixC=(int **)malloc(matrixHighA*sizeof(int *));
        for (i=0;i<matrixHighA;i++)
            matrixC[i]=(int *)malloc(matrixLenB*sizeof(int));

        for(i=0;i<nodeNum;i++)
        {
            int *receiveCTemp=(int *)malloc(localLenB*localHighA*sizeof(int));

            if(i!=0)
                MPI_Recv(receiveCTemp,localHighA*localLenB,MPI_INT,i,i+400,MPI_COMM_WORLD,&status);
            else{
                for(int ii=0;ii<localHighA*localLenB;ii++){
                    receiveCTemp[ii]=C[ii];
                }
            }

            l=0;
            for(j=0;j<localHighA;j++)
                for(k=0;k<localLenB;k++)
                {
                    matrixC[j+(int)(i/p)*localHighA][k+(int)(i%p)*localLenB]=receiveCTemp[l++];
                }
            free(receiveCTemp);
        }
        printf("C:\n");
        PrintMatrix(matrixC,matrixHighA,matrixLenB);
     }

    MPI_Finalize();
    return 0; 
}
 


