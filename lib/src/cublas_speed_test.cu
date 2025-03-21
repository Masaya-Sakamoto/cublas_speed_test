#include "utils.h"
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#define ALIGN 32

int cudaErrorHandle(cudaError_t result)
{
    if (result == cudaSuccess)
    {
        return 0;
    }
    else if (result == cudaErrorInvalidValue)
    {
        std::cout << "Error: Invalid value\n";
        return 1;
    }
    else if (result == cudaErrorInvalidMemcpyDirection)
    {
        std::cout << "Error: Invalid memory copy direction\n";
        return 1;
    }
    return 2;
}

int memcpyPinned(cuComplex *h_A, cuComplex *h_B, cuComplex *h_C, cuComplex *h_alpha, cuComplex *h_beta, const cf_t *A,
                 const cf_t *B, const cf_t *C, const cf_t *alpha, const cf_t *beta, const int M, const int N,
                 const int K)
{
    if (sizeof(cf_t) != sizeof(cuComplex))
    {
        return 1;
    }
    memcpy(h_A, A, sizeof(cf_t) * M * K);
    memcpy(h_B, B, sizeof(cf_t) * K * N);
    memcpy(h_C, C, sizeof(cf_t) * M * N);
    h_alpha->x = alpha->r;
    h_alpha->y = alpha->i;
    h_beta->x = beta->r;
    h_beta->y = beta->i;
    return 0;
}

std::pair<int, float> Arrays2Device(cuComplex *d_A, cuComplex *d_B, cuComplex *d_C, cuComplex *h_A, cuComplex *h_B,
                                    cuComplex *h_C, int M, int N, int K)
{
    int check = 0;
    cudaError_t result;

    // 初期化
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    result = cudaMemcpy(d_A, h_A, sizeof(cuComplex) * M * K, cudaMemcpyHostToDevice);
    check += cudaErrorHandle(result);
    cudaEventRecord(start);
    result = cudaMemcpy(d_B, h_B, sizeof(cuComplex) * K * N, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    check += cudaErrorHandle(result);
    result = cudaMemcpy(d_C, h_C, sizeof(cuComplex) * M * N, cudaMemcpyHostToDevice);
    check += cudaErrorHandle(result);
    return std::make_pair(check, milliseconds);
}

std::pair<int, float> Array2Host(cuComplex *h_A, cuComplex *h_B, cuComplex *h_C, cuComplex *d_A, cuComplex *d_B,
                                 cuComplex *d_C, int M, int N, int K)
{
    int check = 0;
    cudaError_t result;

    // 初期化
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    result = cudaMemcpy(h_A, d_A, sizeof(cuComplex) * M * K, cudaMemcpyDeviceToHost);
    check += cudaErrorHandle(result);
    result = cudaMemcpy(h_B, d_B, sizeof(cuComplex) * K * N, cudaMemcpyDeviceToHost);
    check += cudaErrorHandle(result);
    cudaEventRecord(start);
    result = cudaMemcpy(h_C, d_C, sizeof(cuComplex) * M * N, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    check += cudaErrorHandle(result);
    return std::make_pair(check, milliseconds);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
        return 1;
    }
    int N = atoi(argv[0]);
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int iters = atoi(argv[3]);

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