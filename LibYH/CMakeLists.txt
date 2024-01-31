cmake_minimum_required(VERSION 3.27)
project(LibYH CUDA)

set(CMAKE_CUDA_STANDARD 17)
add_subdirectory(cuda-common)

add_library(LibYH INTERFACE cuda-common)

add_executable(main main.cpp)
