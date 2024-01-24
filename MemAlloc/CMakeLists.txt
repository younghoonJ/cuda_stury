cmake_minimum_required(VERSION 3.27)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(mem_alloc CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(mem_alloc main.cu)

set_target_properties(mem_alloc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
