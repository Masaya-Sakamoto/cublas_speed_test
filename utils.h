#include <cstdint>
#include <vector>
#include <cstdlib>

typedef struct c16_t
{
    int16_t r;
    int16_t i;
};

typedef struct cf_t
{
    float r;
    float i;
};

int setArray(cf_t *arrayPtr, size_t array_size);

int setValue(cf_t *complex_value);

int setArrays(cf_t *A, cf_t *B, cf_t *C, cf_t *alpha, cf_t *beta, const int M, const int N, const int K);

double getMean(const std::vector<double> results);

double getStdev(const std::vector<double> results, int ddof=1);