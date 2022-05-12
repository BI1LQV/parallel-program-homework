cc=nvcc
NVCC_OPTIONS=-lcublas -lcudart -lcusparse -O3 -gencode=arch=compute_61,code=compute_61
OPENBLAS_INSTALL_PATH=~/github/OpenBLAS/build/
ALL:TIME
	${cc} main.cu time.o -o gemm_test ${NVCC_OPTIONS} -I ${OPENBLAS_INSTALL_PATH} ${OPENBLAS_INSTALL_PATH}lib/libopenblas.a
TIME:
	gcc time.c -c
