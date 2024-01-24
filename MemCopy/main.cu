#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

__global__ void printDeviceData(const int *d_dataPtr) {
    printf("%d", d_dataPtr[threadIdx.x]);
}

__global__ void setData(int *d_dataPtr) {
    d_dataPtr[threadIdx.x] = 2;
}

int main() {
    int data[10];
    for (auto i = 0; i < 10; ++i) data[i] = 7;

    int *d_dataPtr;

    cudaDeviceSynchronize();
    auto err = cudaMalloc(&d_dataPtr, sizeof(int) * 10);
    err = cudaMemset(d_dataPtr, 0, sizeof(int) * 10);
    printf("device data: ");
    printDeviceData<<<1, 10>>>(d_dataPtr);
    cudaDeviceSynchronize();
    printf("\n");

    err = cudaMemcpy(d_dataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice);
    printf("host to device: ");
    printDeviceData<<<1, 10>>>(d_dataPtr);
    cudaDeviceSynchronize();
    printf("\n");

    setData<<<1, 10>>>(d_dataPtr);
    printf("device set data: ");
    printDeviceData<<<1, 10>>>(d_dataPtr);
    cudaDeviceSynchronize();
    printf("\n");

    err = cudaMemcpy(data, d_dataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost);
    printf("device to host: ");
    for (auto i = 0; i < 10; ++i) printf("%d", data[i]);

    cudaFree(d_dataPtr);
    return 0;
}
