#include "utils.h"
#include <random>
#include <iostream>

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

double getStdev(const std::vector<double> results, int ddof)
{
    double sqdiff_sum = 0;
    double mean = getMean(results);
    for (const auto result : results)
    {
        sqdiff_sum += (result - mean) * (result - mean);
    }
    return sqrt(sqdiff_sum / (results.size() - ddof));
}

void printSimpleResults(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times)
{
    // Firstly, check if all three inputs have the same size
    if (calc_duration_times.size() != memcpyh2d_duration_times.size() ||
        calc_duration_times.size() != memcpyd2h_duration_times.size())
    {
        throw std::invalid_argument("All input vectors must have the same size.");
    }
    // 
    for (int i = 0; i < calc_duration_times.size(); i++)
    {
        std::cout << calc_duration_times[i] << ",";
        std::cout << memcpyh2d_duration_times[i] << ",";
        std::cout << memcpyd2h_duration_times[i] << std::endl;
    }
    return;
}

void printResultsSummary(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times)
{
    // Firstly, check if all three inputs have the same size
    if (calc_duration_times.size() != memcpyh2d_duration_times.size() ||
        calc_duration_times.size() != memcpyd2h_duration_times.size())
    {
        throw std::invalid_argument("All input vectors must have the same size.");
    }
    // 
    std::cout << getMean(calc_duration_times) << "," << getStdev(calc_duration_times);
    std::cout << getMean(memcpyh2d_duration_times) << "," << getStdev(memcpyh2d_duration_times);
    std::cout << getMean(memcpyd2h_duration_times) << "," << getStdev(memcpyd2h_duration_times);
    std::cout << std::endl;
}

void printResults(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times)
    {
        switch (print_option)
        {
        case PRINT_NONE:
            // 
            break;
        case PRINT_ALL:
            printSimpleResults(
                calc_duration_times,
                memcpyh2d_duration_times,
                memcpyd2h_duration_times
            );
            break;
        case PRINT_MEAN_STDEV:
            printResultsSummary(
                calc_duration_times,
                memcpyh2d_duration_times,
                memcpyd2h_duration_times
            );
            break;
        
        default:
            break;
        }
    }