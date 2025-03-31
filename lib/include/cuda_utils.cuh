#include <cublas_v2.h>
#include "utils.h"

typedef std::pair<cublasHandle_t, cudaStream_t> cudaStreamHandle_t;

int cudaErrorHandle(cudaError_t result);

int memcpyPinned(cuComplex *h_A, cuComplex *h_B, cuComplex *h_C, cuComplex *h_alpha, cuComplex *h_beta, const cf_t *A,
                 const cf_t *B, const cf_t *C, const cf_t *alpha, const cf_t *beta, const int M, const int N,
                 const int K);

std::pair<int, float> Arrays2Device(cuComplex *d_A, cuComplex *d_B, cuComplex *d_C, cuComplex *h_A, cuComplex *h_B,
                                    cuComplex *h_C, int M, int N, int K);

std::pair<int, float> Array2Host(cuComplex *h_C, cuComplex *d_C, int M, int N, int K);

std::pair<int, float> Arrays2DeviceWithStreams(
    cuComplex *d_A, cuComplex *d_B, cuComplex *d_C,
    cuComplex *h_A, cuComplex *h_B, cuComplex *h_C,
    int M, int N, int K,
    int divisions, std::vector<cudaStreamHandle_t> &streamHandles);

std::pair<int, float> Arrays2HostWithStreams(
    cuComplex *d_C, cuComplex *h_C,
    int M, int N, int K,
    int divisions, std::vector<cudaStreamHandle_t> &streamHandles);