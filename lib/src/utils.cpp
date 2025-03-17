#include "utils.h"
#include <random>

int setArray(cf_t *arrayPtr, size_t array_size)
{
    // set noraml distribution random values
    // but the element type is complex float
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < array_size; ++i)
    {
        arrayPtr[i].r = static_cast<float>(dis(gen)); // fixme
        arrayPtr[i].i = static_cast<float>(dis(gen));
    }
    return 0;
}

int setValue(cf_t *complex_value)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    complex_value->r = static_cast<float>(dis(gen));
    complex_value->i = static_cast<float>(dis(gen));
    return 0;
}

int setArrays(cf_t *A, cf_t *B, cf_t *C, cf_t *alpha, cf_t *beta, const int M, const int N, const int K)
{
    int result = 0;
    result += setArray(A, M * K);
    result += setArray(B, K * N);
    result += setArray(C, M * N);
    result += setValue(alpha);
    result += setValue(beta);
    return result;
}

double getMean(const std::vector<double> results)
{
    double sum = 0;
    for (const auto result : results)
    {
        sum += result;
    }
    return sum / results.size();
}

double getStdev(const std::vector<double> results, int ddof=1)
{
    double sqdiff_sum = 0;
    double mean = getMean(results);
    for (const auto result : results)
    {
        sqdiff_sum += (result - mean) * (result - mean);
    }
    return sqrt(sqdiff_sum / (results.size() - ddof));
}