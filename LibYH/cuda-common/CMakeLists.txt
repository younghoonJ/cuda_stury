set(LIB_NAME cuda-common)

add_library(${LIB_NAME} STATIC
        src/common.cu
        include/libyh/cuda-common/include/common.cuh
)
set_target_properties(${LIB_NAME} PROPERTIES LANGUAGE CUDA)
target_include_directories(${LIB_NAME} PUBLIC include)


set_target_properties(${LIBNAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
