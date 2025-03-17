# cblas_row_speed_test

## Purpose
Measures the performance of the `cblas_cgemm` function using row-major order.

## Usage
```sh
./cblas_row_speed_test <M> <N> <K> <iterations>
```

## Description
1. Initializes arrays `A`, `B`, and `C` with dimensions based on input parameters.
2. Performs matrix multiplication using `cblas_cgemm` in row-major order.
3. Measures and outputs the mean and standard deviation of the execution time over the specified number of iterations.
