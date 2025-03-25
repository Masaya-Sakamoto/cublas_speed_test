#include "utils.h"
#include <cblas.h>
#include <chrono>
#include <iostream>
#include <vector>
#define ALIGN 32

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int iters = atoi(argv[4]);

    // initialize arrays
    cf_t *A = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * M * K);
    cf_t *B = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * K * N);
    cf_t *C = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * M * N);
    cf_t alpha = {1.0, 0.0};
    cf_t beta = {0.0, 0.0};

    // initialize results
    std::vector<double> ms_results;
    std::vector<double> memcpy_h2d_results(iters, 0.0);
    std::vector<double> memcpy_d2h_results(iters, 0.0);

    for (int i = 0; i < iters+1; i++) // iters + warmup run
    {
        setArrays(A, B, C, &alpha, &beta, M, N, K);
        auto start = std::chrono::high_resolution_clock::now();
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, K, B, N, &beta, C, N);
        auto cpu_duration = std::chrono::high_resolution_clock::now() - start;
        auto time_count = (double)std::chrono::duration_cast<std::chrono::microseconds>(cpu_duration).count() / 1000;
        if (i != 0)
        {
            ms_results.push_back(time_count);
        }
    }
    printResults(ms_results, memcpy_h2d_results, memcpy_d2h_results);

    free(A);
    free(B);
    free(C);
    return 0;
}