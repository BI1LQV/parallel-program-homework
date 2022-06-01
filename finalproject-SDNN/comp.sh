nvcc main.cu -Xcompiler -fopenmp -lopenblas -o a -lcublas -lcudart -lcusparse -O3 -gencode=arch=compute_61,code=compute_61 -I ~/oblas/build 
./a