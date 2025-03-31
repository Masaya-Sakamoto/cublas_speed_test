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

    // initialize cudaEvent
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize cudaHandles and cudaStreams
    std::vector<cublasHandle_t> cudaHandles(divisions);
    std::vector<cudaStream_t> cudaStreams(divisions);
    for (int k = 0; k < divisions; k++)
    {
        cublasCreate(&cudaHandles[k]);
        cudaStreamCreate(&cudaStreams[k]);
        cublasSetStream(cudaHandles[k], cudaStreams[k]); // Associate cuBLAS with stream
    }

    // 
    for (int i = 0; i < iters; i++)
    {
        setArrays((cf_t *)A, B, C, &alpha, &beta, M, N, K);
        memcpyPinned(h_A, h_B, h_C, &d_alpha, &d_beta, A, B, C, &alpha, &beta, M, N, K);
        
    }
}