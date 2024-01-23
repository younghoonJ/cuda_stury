#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void helloCuda() {
    printf("hello GPU!\n");
}

int main() {
    printf("Hello CPU!\n");
    helloCuda<<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}
