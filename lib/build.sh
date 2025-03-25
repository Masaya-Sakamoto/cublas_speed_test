#!/bin/bash

#init
source /opt/AMD/setenv_AOCC.sh
CXX_FLAGS=" -L/usr/local/lib -lblis-mt -I/usr/local/include/blis"

#clean up
rm -rf build
mkdir -p build/modules

# compile utils
g++ -g -O3 -Iinclude -c src/utils.cpp -o build/modules/utils.o

# compile CXX files
g++ -g -O0 -Iinclude -c src/cblas_col_speed_test.cpp -o build/modules/cblas_col_speed_test.o -std=c++20
g++ -g -O0 -Iinclude -c src/cblas_row_speed_test.cpp -o build/modules/cblas_row_speed_test.o -std=c++20

# link cxx objects
g++ -g build/modules/cblas_col_speed_test.o build/modules/utils.o -o build/cblas_col_speed_test $CXX_FLAGS
g++ -g build/modules/cblas_row_speed_test.o build/modules/utils.o -o build/cblas_row_speed_test $CXX_FLAGS

# compile CUDA file
nvcc -g -O0 -Iinclude -c src/cublas_speed_test.cu -o build/modules/cublas_speed_test.o
# link cuda and cxx objects
nvcc -g build/modules/cublas_speed_test.o build/modules/utils.o -o build/cublas_speed_test $CXX_FLAGS -lcublas