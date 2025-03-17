# cblas_col_speed_test

## Purpose
Measures the performance of the `cblas_cgemm` function using column-major order.

## Usage
```sh
./cblas_col_speed_test <M> <N> <K> <iterations>
```

## Description
1. Initializes arrays `A`, `B`, and `C` with dimensions based on input parameters.
2. Performs matrix multiplication using `cblas_cgemm` in column-major order.
3. Measures and outputs the mean and standard deviation of the execution time over the specified number of iterations.
