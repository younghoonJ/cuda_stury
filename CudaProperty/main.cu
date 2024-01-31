#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

int
getGpuCount() {
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    return ngpus;
}

cudaDeviceProp
getDeviceProp(int idx) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, idx);
    return prop;
}

inline int
_ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
        {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
        {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
        {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int
main() {
    auto ngpus = getGpuCount();

    for (int i = 0; i < ngpus; ++i) {
        auto prop = getDeviceProp(i);
        printf("Device %d: %s\n", i, prop.name);
        printf("Compute capability: %d, %d\n", prop.major, prop.minor);
        printf("#SM: %d\n", prop.multiProcessorCount);
        printf("#Cuda core: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) *
                                       prop.multiProcessorCount);
        printf("Global mem size: %.2f MB\n",
               float(prop.totalGlobalMem) / (1024 * 1024));
    }
    return 0;
}
