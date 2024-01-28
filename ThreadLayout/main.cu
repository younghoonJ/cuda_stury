#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cstddef>
#include <iostream>

__global__ void
checkIndex() {
    printf(
        "ThreadIdx=(%d, %d, %d), blockIdx=(%d, %d, %d), blockDim=(%d, %d, %d), "
        "gridDim=(%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
        blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y,
        gridDim.z);
}

int
main() {

    dim3 dimBlock(4, 2, 3);
    dim3 dimGrid(2, 1, 1);

    const auto num_thread_total = dimBlock.x * dimBlock.y * dimBlock.z *
                                  dimGrid.x * dimGrid.y * dimGrid.z;

    printf("dimGrid=(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("dimBlock=(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);

    // checkIndex<<<dimGrid, dimBlock>>>();
    checkIndex<<<1, 1024>>>();
    cudaDeviceSynchronize();

    printf("num total thread=%u", num_thread_total);
    return 0;
}
