#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#define ALIGN 32

int main(int argc, char *argv[])
{
    auto args = getTestArgs(argc, argv);
    auto M = args.at("M");
    auto N = args.at("N");
    auto K = args.at("K");
    auto divisions = args.at("divisions");
    auto iters = args.at("iters");
    int n = ceil((double)N/divisions);

    // initialize host arrays
    cf_t *A = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * M * K);
    cf_t *B = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * K * N);
    cf_t *C = (cf_t *)aligned_alloc(ALIGN, sizeof(cf_t) * M * N);
    cuComplex *h_A, *h_B, *h_C;
    cudaHostAlloc((void **)&h_A, sizeof(cuComplex) * M * K, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, sizeof(cuComplex) * K * N, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, sizeof(cuComplex) * M * N, cudaHostAllocDefault);
    cf_t alpha, beta;
    cuComplex d_alpha, d_beta;

    // initialize device arrays
    cuComplex *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(cuComplex) * M * K);
    cudaMalloc((void **)&d_B, sizeof(cuComplex) * K * N);
    cudaMalloc((void **)&d_C, sizeof(cuComplex) * M * N);

    // initialize results
    std::vector<double> memcpy_d2h_results, memcpy_h2d_results;
    std::vector<double> ms_results{-1};
    cudaEvent_t start, stop;

    // initialize cudaEvent
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize cudaHandles and cudaStreams
    std::vector<cudaStreamHandle_t> cudaStreamHandles(divisions);
    // std::vector<cublasHandle_t> cudaHandles(divisions);
    // std::vector<cudaStream_t> cudaStreams(divisions);
    for (int k = 0; k < divisions; k++)
    {
        cublasCreate(&cudaStreamHandles[k].first);
        cudaStreamCreate(&cudaStreamHandles[k].second);
        cublasSetStream(cudaStreamHandles[k].first, cudaStreamHandles[k].second);
    }

    // 
    for (int i = 0; i < iters+1; i++)
    {
        setArrays((cf_t *)A, B, C, &alpha, &beta, M, N, K);
        memcpyPinned(h_A, h_B, h_C, &d_alpha, &d_beta, A, B, C, &alpha, &beta, M, N, K);
        auto mem_h2d_result = Arrays2DeviceWithStreams(d_A, d_B, d_C, h_A, h_B, h_C, M, N, K, divisions, cudaStreamHandles);
        cudaEventRecord(start);
        for (int k = 0; k < divisions; k++)
        {
            auto start_point = k*n;
            auto end_point = start_point + n;
            auto _n = end_point < N ? n : N-start_point;
            cublasCgemm(
                cudaStreamHandles[k].first,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                _n,
                M,
                K,
                &d_alpha,
                &d_B[start_point],
                _n,
                d_A,
                K,
                &d_beta,
                &d_C[start_point],
                _n
            );
        }
        cudaEventRecord(stop);

        auto mem_d2h_result = Array2Host(h_C, d_C, M, N, K);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // 記録
        if (i != 0)
        {
            ms_results.push_back(milliseconds);
            memcpy_h2d_results.push_back(mem_h2d_result.second);
            memcpy_d2h_results.push_back(mem_d2h_result.second);
        }
    }
    printResults(ms_results, memcpy_h2d_results, memcpy_d2h_results);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    free(A);
    free(B);
    free(C);
    for (int k = 0; k < divisions; k++)
    {
        cublasDestroy(cudaStreamHandles[k].first);
        cudaStreamDestroy(cudaStreamHandles[k].second);
    }
    cudaErrorHandle(cudaDeviceReset());
    return 0;
}