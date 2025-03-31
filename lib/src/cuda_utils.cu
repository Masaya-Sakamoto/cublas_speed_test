#include <cublas_v2.h>
#include <iostream>
#include "cuda_utils.cuh"

typedef std::pair<cublasHandle_t, cudaStream_t> cudaStreamHandle_t;

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

std::pair<int, float> Arrays2DeviceWithStreams(
    cuComplex *d_A, cuComplex *d_B, cuComplex *d_C,
    cuComplex *h_A, cuComplex *h_B, cuComplex *h_C,
    int M, int N, int K,
    int divisions, std::vector<cudaStreamHandle_t> &streamHandles)
{
    int check = 0;
    cudaError_t result;

    // 初期化
    std::vector<cudaEvent_t> starts(divisions), stops(divisions);
    for (int k = 0; k < divisions; k++)
    {
        cudaEventCreate(&starts[k]);
        cudaEventCreate(&stops[k]);
    }

    // パラメタ設定
    int n = ceil((double)N/divisions);
    int _N = N;

    // 時間測定結果保管用
    float milliseconds = 0;

    // 非測定部分の行列を先に転送
    result = cudaMemcpy(d_A, h_A, sizeof(cuComplex) * M * K, cudaMemcpyHostToDevice);
    check += cudaErrorHandle(result);
    result = cudaMemcpy(d_C, h_C, sizeof(cuComplex) * M * N, cudaMemcpyHostToDevice);
    check += cudaErrorHandle(result);

    
    for (int k = 0; k < divisions; k++)
    {
        // update _n: transfering size
        auto start_point = k*n;
        auto end_point = (k+1)*n;
        auto _n = end_point < N ? n : N-start_point;

        cudaEventRecord(starts[k]);
        result = cudaMemcpyAsync(
            &d_B[start_point],
            &h_B[start_point],
            sizeof(cuComplex) * K * _n,
            cudaMemcpyHostToDevice,
            streamHandles[k].second);
        check += cudaErrorHandle(result);
    }
    for (int k = 0; k < divisions; k++)
    {
        float tmp_milliseconds = 0;
        cudaStreamSynchronize(streamHandles[k].second);
        cudaEventRecord(stops[k]);
        cudaEventSynchronize(stops[k]);
        cudaEventElapsedTime(&tmp_milliseconds, starts[k], stops[k]);
        milliseconds += tmp_milliseconds;
    }
    
    
    return std::make_pair(check, milliseconds/divisions);
}

std::pair<int, float> Array2Host(
    cuComplex *h_C, cuComplex *d_C,
    int M, int N, int K)
{
    int check = 0;
    cudaError_t result;

    // 初期化
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    result = cudaMemcpy(h_C, d_C, sizeof(cuComplex) * M * N, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    check += cudaErrorHandle(result);
    return std::make_pair(check, milliseconds);
}

std::pair<int, float> Array2HostWithStreams(
    cuComplex *d_C, cuComplex *h_C,
    int M, int N, int K,
    int divisions, std::vector<cudaStreamHandle_t> &streamHandles)
{
    int check = 0;
    cudaError_t result;

    // 初期化
    std::vector<cudaEvent_t> starts(divisions), stops(divisions);
    for (int k = 0; k < divisions; k++)
    {
        cudaEventCreate(&starts[k]);
        cudaEventCreate(&stops[k]);
    }

    // パラメタ設定
    int n = ceil((double)N/divisions);
    int _N = N;

    // 時間測定結果保管用
    float milliseconds = 0;

    for (int k = 0; k < divisions; k++)
    {
        // update _n: transfering size
        auto start_point = k*n;
        auto end_point = (k+1)*n;
        auto _n = end_point < N ? n : N-start_point;
        
        cudaEventRecord(starts[k]);
        result = cudaMemcpyAsync(
            &h_C[start_point],
            &d_C[start_point],
            sizeof(cuComplex) * K * _n,
            cudaMemcpyDeviceToHost,
            streamHandles[k].second);
        check += cudaErrorHandle(result);
    }
    for (int k = 0; k < divisions; k++)
    {
        float tmp_milliseconds = 0;
        cudaStreamSynchronize(streamHandles[k].second);
        cudaEventRecord(stops[k]);
        cudaEventSynchronize(stops[k]);
        cudaEventElapsedTime(&tmp_milliseconds, starts[k], stops[k]);
        milliseconds += tmp_milliseconds;
    }
    
    return std::make_pair(check, milliseconds/divisions);
}