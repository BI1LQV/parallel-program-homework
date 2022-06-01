//
// Created by kouushou on 2021/5/19.
//

#ifndef MATRIXMULTISAMPLES_DEFINES_H
#define MATRIXMULTISAMPLES_DEFINES_H

#if defined(__cplusplus)
extern "C"{
#endif
#define BLOCK_SIZE 16
#define NNZ 20
#define WARMUP_TIMES 5
#define VALUE_TYPE double
#define BENCH_TIMES 10

void timeStart();
double timeCut();

#if defined(__cplusplus)
}
#endif

#endif //MATRIXMULTISAMPLES_DEFINES_H
