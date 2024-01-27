cmake_minimum_required(VERSION 3.27)


set(project_name VectorAddition)

set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

project(${VectorAddition}CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(%{VectorAddigion} main.cu)

set_target_properties(%{VectorAddigion} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
