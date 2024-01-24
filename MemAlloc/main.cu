#include <iostream>
// I don't know why i dont need to include cuda_runtime.h

void deviceMemInfo(const char *header = nullptr) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    if (header != nullptr) {
        printf("[%s] Device Mem: %lld/%lld bytes\n", header, free, total);
    } else {
        printf("Device Mem: %lld/%lld bytes\n", free, total);
    }
}

int main() {
    int *d_dataPtr;

    deviceMemInfo("start");

    auto err = cudaMalloc(&d_dataPtr, sizeof(int) * 1024 * 1024);
    printf("cudaMalloc: %s\n", cudaGetErrorName(err));
    deviceMemInfo("after cudaMalloc");

    err = cudaMemset(d_dataPtr, 2, sizeof(int) * 1024 * 1024);
    printf("cudaMemset: %s\n", cudaGetErrorName(err));
    deviceMemInfo("after cudaMemset");

    err = cudaFree(d_dataPtr);
    printf("cudaFree: %s\n", cudaGetErrorName(err));
    deviceMemInfo();

    return 0;
}
