#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
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
T *
cudaMallocInit(size_t memSize) {
    T *ptr_;
    auto err = cudaMalloc(&ptr_, memSize);
    if (PRINT_ERR_LOG) {
        // ReSharper disable once CppDFAUnreachableCode
        printf("cudaMalloc::cudaMalloc: %s\n", cudaGetErrorName(err));
    }
    err = cudaMemset(ptr_, 0, memSize);
    if (PRINT_ERR_LOG) {
        // ReSharper disable once CppDFAUnreachableCode
        printf("cudaMalloc::cudaMemset: %s\n", cudaGetErrorName(err));
    }
    return ptr_;
}

inline void
push(void *dst, const void *src, size_t memSize) {
    const auto err = cudaMemcpy(dst, src, memSize, cudaMemcpyHostToDevice);
    if (PRINT_ERR_LOG) {
        // ReSharper disable once CppDFAUnreachableCode
        printf("push:cudaMemcpy: %s\n", cudaGetErrorName(err));
    }
}

inline void
pull(void *dst, const void *src, size_t memSize) {
    const auto err = cudaMemcpy(dst, src, memSize, cudaMemcpyDeviceToHost);
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

__global__ void
vecorAdd(const int *a, int const *b, int *c) {
    const auto tId = threadIdx.x;
    c[tId]         = a[tId] + b[tId];
}

int
main() {

    // clang-format off
    yh::NTimer timer;
    const auto timer_cuda_total  = timer.addTimter("CUDA total");
    const auto timer_kernel_exec = timer.addTimter("Kernel Execution(Device)");
    const auto timer_cp_host_to_device =timer.addTimter("MemCopy(Host->Device)");
    const auto timer_cp_device_to_host =timer.addTimter("MemCopy(Device->Host)");
    const auto timer_host_vec_add = timer.addTimter("Kernel Execution(Host)");
    // clang-format on

    const auto NUM_DATA = 1024 * 1;
    const auto memSize  = yh::getMemSize<int>(NUM_DATA);

    const auto a          = new int[memSize];
    const auto b          = new int[memSize];
    const auto ans_device = new int[memSize];
    const auto ans_host   = new int[memSize];

    memset(a, 0, memSize);
    memset(b, 0, memSize);
    memset(ans_device, 0, memSize);
    memset(ans_host, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }


    timer.tick(timer_host_vec_add);
    for (int i = 0; i < NUM_DATA; ++i) {
        ans_host[i] = a[i] + b[i];
    }
    timer.tock(timer_host_vec_add);

    const auto da = yh::cudaMallocInit<int>(memSize);
    const auto db = yh::cudaMallocInit<int>(memSize);
    const auto dc = yh::cudaMallocInit<int>(memSize);

    timer.tick(timer_cuda_total);

    timer.tick(timer_cp_host_to_device);
    yh::push(da, a, memSize);
    yh::push(db, b, memSize);
    timer.tock(timer_cp_host_to_device);

    timer.tick(timer_kernel_exec);
    vecorAdd<<<1, NUM_DATA>>>(da, db, dc);
    cudaDeviceSynchronize();
    timer.tock(timer_kernel_exec);

    timer.tick(timer_cp_device_to_host);
    yh::pull(ans_device, dc, memSize);
    timer.tock(timer_cp_device_to_host);

    timer.tock(timer_cuda_total);

    cudaFree(da), cudaFree(db), cudaFree(dc);

    int num_diff = 0;
    for (auto i = 0; i < NUM_DATA; ++i) {
        if (ans_host[i] != ans_device[i]) {
            num_diff += 1;
        }
    }
    const auto err = static_cast<double>(num_diff) / NUM_DATA * 100;
    printf("num_diff/total = %d/%d, %.6f%%\n", num_diff, NUM_DATA, err);

    delete[] a;
    delete[] b;
    delete[] ans_device;
    delete[] ans_host;


    std::cout << timer << std::endl;

    return 0;
}
