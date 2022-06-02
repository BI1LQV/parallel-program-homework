clean:
	rm -rf summa
summa:clean
	mpicc -g -Wall -O3 -fopenmp -std=c11 main.c -o summa -lm
run:summa
	mpirun -np 1 ./summa 1 1000 1000 1000

