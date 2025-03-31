#include <cublas_v2.h>
#include <iostream>
#include "cuda_utils.cuh"

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