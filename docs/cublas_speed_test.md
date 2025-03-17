# cublas_speed_test

## Purpose
Measures the performance of the `cublasCgemm` function using CUDA.

## Usage
```sh
./cublas_speed_test <M> <N> <K> <iterations>
```

## Description
1. Initializes host and device arrays `A`, `B`, and `C` with dimensions based on input parameters.
2. Performs matrix multiplication using `cublasCgemm`.
3. Measures and outputs the mean and standard deviation of the execution time, as well as memory transfer times between host and device, over the specified number of iterations.
