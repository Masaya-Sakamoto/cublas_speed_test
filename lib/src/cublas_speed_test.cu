#include "cuda_utils.cuh"
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#define ALIGN 32

int main_part_single_stream(int M, int N, int K, int iters)
{
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
    Array2Host(h_C, d_C, M, N, K);

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
        auto mem_d2h_result = Array2Host(h_C, d_C, M, N, K);
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
    cudaErrorHandle(cudaDeviceReset());
    return 0;
}

int main_part_multiple_stream(int M, int N, int K, int divisions, int iters)
{
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
    std::vector<double> ms_results, memcpy_d2h_results, memcpy_h2d_results;
    cudaEvent_t start, stop;

    // initialize cudaEvent
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize cudaHandles and cudaStreams
    cublasHandle_t cublasHandle;
    std::vector<cudaStream_t> streams(divisions);
    // std::vector<cublasHandle_t> cudaHandles(divisions);
    // std::vector<cudaStream_t> cudaStreams(divisions);
    for (int k = 0; k < divisions; k++)
    {
        cudaStreamCreate(&streams[k]);
    }

    // 
    for (int i = 0; i < iters+1; i++)
    {
        setArrays((cf_t *)A, B, C, &alpha, &beta, M, N, K);
        memcpyPinned(h_A, h_B, h_C, &d_alpha, &d_beta, A, B, C, &alpha, &beta, M, N, K);
        cudaMemcpy(d_A, h_A, sizeof(cuComplex) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, sizeof(cuComplex) * M * N, cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        for (int k = 0; k < divisions; k++)
        {
            auto start_point = k*n;
            auto end_point = start_point + n;
            auto _n = end_point < N ? n : N-start_point;
            cublasSetStream(cublasHandle, streams[k]);
            cudaMemcpyAsync(&d_B[start_point], &h_B[start_point], sizeof(cuComplex) * K * _n, cudaMemcpyHostToDevice, streams[k];
            cublasCgemm(
                cublasHandle,
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
                &d_C[start_point], // TODO: fixme: wrong start_point
                _n
            );
            cudaMemcpyAsync(&h_C[start_point], &d_C[start_point], sizeof(cuComplex) * K * _n, cudaMemcpyDeviceToHost, streams[k]);
        }
        // memcpy sync
        for (int k = 0; k < divisions; k++)
        {
            cudaStreamSynchronize(streams[k]);
        }
        cudaEventRecord(stop);

        // auto mem_d2h_result = Arrays2HostWithStreams(h_C, d_C, M, N, K, divisions, cudaStreamHandles);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // 記録
        if (i != 0)
        {
            ms_results.push_back(milliseconds);
            memcpy_h2d_results.push_back(0);
            memcpy_d2h_results.push_back(0);
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
    cublasDestroy(cublasHandle);
    for (int k = 0; k < divisions; k++)
    {
        cudaStreamDestroy(streams[k]);
    }
    cudaErrorHandle(cudaDeviceReset());
    return 0;
}

int main(int argc, char *argv[])
{
    auto args = getTestArgs(argc, argv);
    auto M = args.at("M");
    auto N = args.at("N");
    auto K = args.at("K");
    auto divisions = args.at("divisions");
    auto iters = args.at("iters");
    
    if (divisions > 0)
    {
        return main_part_multiple_stream(M, N, K, divisions, iters);
    }
    else
    {
        return main_part_single_stream(M, N, K, iters);
    }
}