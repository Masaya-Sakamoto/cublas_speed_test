#pragma once
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <map>
#include <string>

enum PRINT_OPTIONS
{
    PRINT_NONE,
    PRINT_ALL,
    PRINT_MEAN_STDEV
};

static int print_option = PRINT_ALL;

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

std::map<std::string, int> getTestArgs(int argc, char *argv[]);

int setArray(cf_t *arrayPtr, size_t array_size);

int setValue(cf_t *complex_value);

int setArrays(cf_t *A, cf_t *B, cf_t *C, cf_t *alpha, cf_t *beta, const int M, const int N, const int K);

double getMean(const std::vector<double> results);

double getStdev(const std::vector<double> results, int ddof = 1);

void printSimpleResults(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times);

void printResultsSummary(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times);

void printResults(
    const std::vector<double> calc_duration_times,
    const std::vector<double> memcpyh2d_duration_times,
    const std::vector<double> memcpyd2h_duration_times);