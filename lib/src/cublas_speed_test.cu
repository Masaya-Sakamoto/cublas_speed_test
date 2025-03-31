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
    auto iters = args.at("iters");

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
    std::vector<double> ms_results, memcpy_d2h_results, memcpy_h2d_results;
    cudaEvent_t start, stop;

    // 初期化
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize cuda, cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // warm-up run
    float warmup;
    setArrays((cf_t *)A, B, C, &alpha, &beta, M, N, K);
    memcpyPinned(h_A, h_B, h_C, &d_alpha, &d_beta, A, B, C, &alpha, &beta, M, N, K);
    Arrays2Device(d_A, d_B, d_C, h_A, h_B, h_C, M, N, K);
    cudaEventRecord(start);
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&warmup, start, stop);
    Array2Host(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);

    for (int i = 0; i < iters; i++)
    {
        setArrays((cf_t *)A, B, C, &alpha, &beta, M, N, K);
        memcpyPinned(h_A, h_B, h_C, &d_alpha, &d_beta, A, B, C, &alpha, &beta, M, N, K);
        auto mem_h2d_result = Arrays2Device(d_A, d_B, d_C, h_A, h_B, h_C, M, N, K);
        memcpy_h2d_results.push_back(mem_h2d_result.second);
        cudaEventRecord(start);
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        auto mem_d2h_result = Array2Host(h_A, h_B, h_C, d_A, d_B, d_C, M, N, K);
        memcpy_d2h_results.push_back(mem_d2h_result.second);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        ms_results.push_back(milliseconds);
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
    cublasDestroy(handle);
    return 0;
}