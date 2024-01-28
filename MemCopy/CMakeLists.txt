cmake_minimum_required(VERSION 3.27)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(mem_copy CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(mem_copy main.cu)
