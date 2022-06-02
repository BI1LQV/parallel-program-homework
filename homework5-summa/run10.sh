# #!/bin/bash
rm -rf summa
mpicc -g -Wall -O3 -fopenmp -std=c11 main.c -o summa -lm
#100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
echo 'time,gflops' >> h5.csv
matrixNum=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)
procNum=(1 4 16)
ouput=""
for i in ${procNum[*]}
do
    for j in ${matrixNum[*]}
    do 
        # sleep 2
        # echo ${i} ${j}
        mpirun -np ${i} ./summa ${i} ${j} ${j} ${j} >> h5.csv
    done
done
#$1第一个参数 $2第二个参数
# mpirun -np $1 ./summa $1 $2 $2 $2
