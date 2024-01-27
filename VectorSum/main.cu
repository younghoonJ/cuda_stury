#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <memory>

#define NUM_DATA 1024
#define MEMSIZE_(type_, n_data_) sizeof(type_) * n_data_;

int
main() {
    int *a, *b, *c;

    int memSize = MEMSIZE_(int, NUM_DATA);


    printf("%lld", memSize);


    return 0;
}
