clean:
	rm -rf summa
summa:clean
	mpicc -g -Wall -O3 -fopenmp -std=c11 main.c -o summa -lm
run:summa
	mpirun -np 4 ./summa 4 16 16 16


