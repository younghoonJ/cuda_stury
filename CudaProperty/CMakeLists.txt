cmake_minimum_required(VERSION 3.27)

set(project_name CudaProperty)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(${project_name} CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(${project_name} main.cu)

set_target_properties(${project_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
