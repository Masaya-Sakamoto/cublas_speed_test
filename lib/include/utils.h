#pragma once
#include <cstdint>
#include <vector>
#include <cstdlib>

typedef struct
{
    int16_t r;
    int16_t i;
} c16_t;

typedef struct
{
    float r;
    float i;
} cf_t;

int setArray(cf_t *arrayPtr, size_t array_size);

int setValue(cf_t *complex_value);

int setArrays(cf_t *A, cf_t *B, cf_t *C, cf_t *alpha, cf_t *beta, const int M, const int N, const int K);

double getMean(const std::vector<double> results);

double getStdev(const std::vector<double> results, int ddof=1);