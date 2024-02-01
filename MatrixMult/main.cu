#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

namespace yh {
// control
constexpr bool PRINT_ERR_LOG = false;

template<typename T>
size_t
getMemSize(size_t numData) {
    return sizeof(T) * numData;
}

template<typename T>
using DevicePtr = T *;

template<typename T>
DevicePtr<T>
deviceMallocInit(size_t num_data) {
    T *ptr_;
    auto err = cudaMalloc(&ptr_, num_data * sizeof(T));
    if (PRINT_ERR_LOG) {
        printf("cudaMalloc::cudaMalloc: %s\n", cudaGetErrorName(err));
    }
    err = cudaMemset(ptr_, 0, num_data * sizeof(T));
    if (PRINT_ERR_LOG) {
        printf("cudaMalloc::cudaMemset: %s\n", cudaGetErrorName(err));
    }
    return ptr_;
}

template<typename T>
void
deviceMemFree(DevicePtr<T> ptr) {
    auto err = cudaFree(ptr);
    if (PRINT_ERR_LOG) {
        printf("cudaFree: %s\n", cudaGetErrorName(err));
    }
}

template<typename T>
using HostPtr = T *;

template<typename T>
HostPtr<T>
hostMallocInit(size_t num_data) {
    T *arr = new T[num_data];
    memset(arr, 0, sizeof(T) * num_data);
    return arr;
}

template<typename T>
void
push(DevicePtr<T> dst, const HostPtr<T> src, size_t count) {
    const auto err =
        cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
    if (PRINT_ERR_LOG) {
        // ReSharper disable once CppDFAUnreachableCode
        printf("push:cudaMemcpy: %s\n", cudaGetErrorName(err));
    }
}

template<typename T>
inline void
pull(HostPtr<T> dst, const DevicePtr<T> src, size_t count) {
    const auto err =
        cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
    if (PRINT_ERR_LOG) {
        // ReSharper disable once CppDFAUnreachableCode
        printf("pull:cudaMemcpy: %s\n", cudaGetErrorName(err));
    }
}

class NTimer {
    enum class TimerStatus {
        ON  = 0,
        OFF = 1,
    };
    using clock_ = std::chrono::high_resolution_clock;

    class Timer_ {
        TimerStatus status = TimerStatus::OFF;
        std::chrono::time_point<clock_> t_measure_start;
        std::chrono::duration<double> t_acc;
        const char *name_;


    public:
        explicit Timer_(const char *name) : name_(name) { reset(); }

        void reset() {
            status = TimerStatus::OFF;
            t_acc  = std::chrono::duration<double>::zero();
        }

        void tick() {
            if (status == TimerStatus::ON) return;

            status          = TimerStatus::ON;
            t_measure_start = clock_::now();
        }

        void tock() {
            if (status == TimerStatus::OFF) return;

            status = TimerStatus::OFF;
            t_acc += (clock_::now() - t_measure_start);
        }

        friend std::ostream &operator<<(std::ostream &os, const Timer_ &obj) {
            const auto t_mills =
                std::chrono::duration_cast<std::chrono::nanoseconds>(obj.t_acc)
                    .count() /
                1000000.0;
            return os << obj.name_ << ": " << t_mills << " ms";
        }
    };

    bool isOn_ = true;
    std::vector<Timer_> timers;

public:
    NTimer() = default;

    void reset() {
        for (auto &timer : timers)
            timer.reset();
    }

    /*
    Returns the index of the timer just added.
    */
    size_t addTimter(const char *timer_name) {
        timers.emplace_back(timer_name);
        return timers.size() - 1;
    }

    void tick(size_t timer_id) { timers.at(timer_id).tick(); }

    void tock(size_t timer_id) { timers.at(timer_id).tock(); }

    friend std::ostream &operator<<(std::ostream &os, const NTimer &obj) {
        os << "Timer Report[num_timers=" << obj.timers.size() << "]\n";
        for (auto i = 0; i < obj.timers.size(); ++i)
            os << "  " << i << ". " << obj.timers[i] << '\n';
        return os;
    };
};
}  // namespace yh

/*
Kernel when the matrix C is less then Max block size(1024)
A: m x k
B: k x n
C: m x n
*/
template<typename T>
__global__ void
matMul0(yh::DevicePtr<T> A, yh::DevicePtr<T> B, yh::DevicePtr<T> C, int m,
        int n, int k) {
    auto row = threadIdx.x;
    auto col = threadIdx.y;
    auto idx = row * n + col;

    C[idx] = 0;
    for (int i = 0; i < k; ++i) {
        C[idx] += (A[row * k + i] * B[i * n + col]);
    }
}

template<typename T>
__global__ void
matMul1(yh::DevicePtr<T> A, yh::DevicePtr<T> B, yh::DevicePtr<T> C, int m,
        int n, int k) {
    auto row = blockDim.x * blockIdx.x + threadIdx.x;
    auto col = blockDim.y * blockIdx.y + threadIdx.y;
    auto idx = row * n + col;

    if (row < m and col < n) {
        C[idx] = 0;
        for (int i = 0; i < k; ++i)
            C[idx] += A[row * k + i] * B[i * n + col];
    }
}

int
main() {
    // clang-format off
    yh::NTimer timer;
    const auto timer_kernel_host = timer.addTimter("Kernel Execution(Host)");
    const auto timer_kernel_device = timer.addTimter("Kernel Execution(Device)");
    const auto timer_memcp_device_host = timer.addTimter("MemCpy host to device ");
    const auto timer_memcp_host_device = timer.addTimter("MemCpy device to host ");

    // clang-format on

    using DType_ = int;

    int m = 1024, n = 2048, k = 1024;

    auto sizeA = yh::getMemSize<DType_>(m * k);
    auto sizeB = yh::getMemSize<DType_>(k * n);
    auto sizeC = yh::getMemSize<DType_>(m * n);

    auto A          = yh::hostMallocInit<DType_>(sizeA);
    auto B          = yh::hostMallocInit<DType_>(sizeB);
    auto host_ans   = yh::hostMallocInit<DType_>(sizeC);
    auto device_ans = yh::hostMallocInit<DType_>(sizeC);

    auto dA = yh::deviceMallocInit<DType_>(sizeA);
    auto dB = yh::deviceMallocInit<DType_>(sizeB);
    auto dC = yh::deviceMallocInit<DType_>(sizeC);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_int_distribution<int> dis(0, 99);

    for (int i = 0; i < sizeA; i++)
        A[i] = dis(rng);
    for (int i = 0; i < sizeB; i++)
        B[i] = dis(rng);

    timer.tick(timer_kernel_host);
    //    device compute
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            host_ans[row * n + col] = 0;
            for (int y = 0; y < k; ++y) {
                host_ans[row * n + col] += A[row * k + y] * B[y * n + col];
            }
        }
    }
    timer.tock(timer_kernel_host);

    timer.tick(timer_memcp_host_device);
    yh::push(dA, A, sizeA);
    yh::push(dB, B, sizeB);
    timer.tock(timer_memcp_host_device);

    dim3 grid, block;

    if (m * n <= 1024) {
        block = dim3(m, n);
        grid  = dim3(1, 1);

        timer.tick(timer_kernel_device);
        matMul0<<<grid, block>>>(dA, dB, dC, m, n, k);
        cudaDeviceSynchronize();
        timer.tock(timer_kernel_device);

    } else {
        block = dim3(1, 512);
        grid  = dim3(ceil(float(m) / block.x), ceil(float(n) / block.y));

        timer.tick(timer_kernel_device);
        matMul1<<<grid, block>>>(dA, dB, dC, m, n, k);
        cudaDeviceSynchronize();
        timer.tock(timer_kernel_device);
    }


    timer.tick(timer_memcp_device_host);
    yh::pull(device_ans, dC, sizeC);
    timer.tock(timer_memcp_device_host);

    yh::deviceMemFree(dA);
    yh::deviceMemFree(dB);
    yh::deviceMemFree(dC);

    size_t diff_count = 0;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            auto idx = row * n + col;
            if (host_ans[idx] != device_ans[idx]) {
                diff_count += 1;
                printf("diff: C[%d, %d], %d!=%d \n", row, col, host_ans[idx],
                       device_ans[idx]);
            }
        }
    }
    delete[] A;
    delete[] B;
    delete[] host_ans;
    delete[] device_ans;

    printf("A(%d,%d), B(%d,%d), C(%d,%d), grid:(%d, %d), block: (%d, %d)\n", m,
           k, k, n, m, n, grid.x, grid.y, block.x, block.y);


    std::cout << timer << std::endl;
    if (diff_count > 0) {
        printf("diff_count:%lld/%lld", diff_count, (m * n));
    }

    return 0;
}
