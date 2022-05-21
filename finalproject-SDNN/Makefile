cc=nvcc
NVCC_OPTIONS=-lcublas -lcudart -lcusparse -O3 -gencode=arch=compute_61,code=compute_61
ALL
	${cc} main.cu -lopenblas -o a ${NVCC_OPTIONS} 