#!/bin/bash

#init
source /opt/AMD/setenv_AOCC.sh
CXX_FLAGS=" -L/usr/local/lib -lblis-mt -I/usr/local/include/blis"

#clean up
rm -rf build
mkdir build

# compile utils
g++ -Iinclude -c src/utils.cpp -o build/utils.o

# compile CXX files
g++ -Iinclude -c src/cblas_col_speed_test.cpp -o build/cblas_col_speed_test.o -std=c++20
g++ -Iinclude -c src/cblas_row_speed_test.cpp -o build/cblas_row_speed_test.o -std=c++20

# link cxx objects
g++ build/cblas_col_speed_test.o build/utils.o -o build/cblas_col_speed_test $CXX_FLAGS
g++ build/cblas_row_speed_test.o build/utils.o -o build/cblas_row_speed_test $CXX_FLAGS

# compile CUDA file
nvcc -Iinclude -c src/cublas_speed_test.cu -o build/cublas_speed_test.o $CXX_FLAGS
# link cuda and cxx objects
nvcc build/cublas_speed_test.o build/utils.o -o build/cublas_speed_test $CXX_FLAGS -lcublas