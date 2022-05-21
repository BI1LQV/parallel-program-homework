cc=nvcc
NVCC_OPTIONS=-lcublas -lcudart -lcusparse -O3 -gencode=arch=compute_61,code=compute_61
ALL:TIME
	${cc} main.cu time.o -lopenblas -o a ${NVCC_OPTIONS} 
TIME:
	gcc time.c -c
