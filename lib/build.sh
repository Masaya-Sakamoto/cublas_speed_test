#!/bin/bash

#init
source /opt/AMD/setenv_AOCC.sh
CXX_FLAGS=" -L/usr/local/lib -lblis-mt -I/usr/local/include/blis"

#clean up
rm -rf build
mkdir -p build/modules

# compile utils
g++ -O3 -g -Iinclude -c src/utils.cpp -o build/modules/utils.o

# compile CXX files
g++ -O3 -g -Iinclude -c src/cblas_col_speed_test.cpp -o build/modules/cblas_col_speed_test.o -std=c++20
g++ -O3 -g -Iinclude -c src/cblas_row_speed_test.cpp -o build/modules/cblas_row_speed_test.o -std=c++20

# link cxx objects
g++ build/modules/cblas_col_speed_test.o build/modules/utils.o -o build/cblas_col_speed_test $CXX_FLAGS
g++ build/modules/cblas_row_speed_test.o build/modules/utils.o -o build/cblas_row_speed_test $CXX_FLAGS

# compile CUDA utils
nvcc -O3 -g -Iinclude -c src/cuda_utils.cu -o build/modules/cuda_utils.o

# compile CUDA file
nvcc -O3 -g -Iinclude -c src/cublas_speed_test.cu -o build/modules/cublas_speed_test.o

# link cuda and cxx objects
nvcc build/modules/cublas_speed_test.o build/modules/utils.o build/modules/cuda_utils.o  -o build/cublas_speed_test $CXX_FLAGS -lcublas