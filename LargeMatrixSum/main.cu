#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
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

template<typename T>
T *
hostMallocInit(size_t memSize) {
    T *arr = new T[memSize];
    memset(arr, 0, sizeof(T) * memSize);
    return arr;
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

dim3
blockSize2d(size_t x, size_t y) {
    return {static_cast<unsigned int>(x), static_cast<unsigned int>(y)};
}

dim3
blockSize1d(size_t x) {
    return {static_cast<unsigned int>(x)};
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
largeMatrixAdd_G2B2(const float *matA, const float *matB, float *matC,
                    int row_size, int col_size) {
    auto idx_row = (blockIdx.y * blockDim.y + threadIdx.y);
    auto idx_col = (blockIdx.x * blockDim.x + threadIdx.x);
    auto idx     = idx_row * col_size + idx_col;

    if (idx_col < col_size and idx_row < row_size) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

__global__ void
largeMatrixAdd_G1B1(const float *matA, const float *matB, float *matC,
                    int row_size, int col_size) {
    auto idx_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_col < col_size) {
        for (int row = 0; row < row_size; ++row) {
            auto idx  = row * col_size + idx_col;
            matC[idx] = matA[idx] + matB[idx];
        }
    }
}

__global__ void
largeMatrixAdd_G2B1(const float *matA, const float *matB, float *matC,
                    int row_size, int col_size) {
    auto idx_col = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx_row = blockIdx.y;
    auto idx     = idx_row * col_size + idx_col;
    if (idx_col < col_size && idx_row < row_size) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

int
main() {

    // clang-format off
    yh::NTimer timer;
    const auto timer_kernel_g2b2 = timer.addTimter("Kernel Execution(Device G2B2)");
    const auto timer_kernel_g1b1 = timer.addTimter("Kernel Execution(Device G1B1)");
    const auto timer_kernel_g2b1 = timer.addTimter("Kernel Execution(Device G2B1)");
    const auto timer_cp_host_to_device =timer.addTimter("MemCopy(Host->Device)");
    const auto timer_cp_device_to_host =timer.addTimter("MemCopy(Device->Host)");
    const auto timer_host_vec_add = timer.addTimter("Kernel Execution(Host)");
    // clang-format on

    const auto mat_rows = 8192;
    const auto mat_cols = 8192;
    const auto num_data = mat_rows * mat_cols;
    const auto memSize  = yh::getMemSize<float>(num_data);


    const auto matA = yh::hostMallocInit<float>(memSize);
    const auto matB = yh::hostMallocInit<float>(memSize);
    auto ans_host   = yh::hostMallocInit<float>(memSize);
    auto ans_g2b2   = yh::hostMallocInit<float>(memSize);
    auto ans_g1b1   = yh::hostMallocInit<float>(memSize);
    auto ans_g2b1   = yh::hostMallocInit<float>(memSize);


    for (int i = 0; i < memSize; i++) {
        matA[i] = rand() % 100;
        matB[i] = rand() % 50;
    }


    timer.tick(timer_host_vec_add);
    for (int i = 0; i < memSize; ++i) {
        ans_host[i] = matA[i] + matB[i];
    }
    timer.tock(timer_host_vec_add);

    const auto da = yh::cudaMallocInit<float>(memSize);
    const auto db = yh::cudaMallocInit<float>(memSize);
    const auto dc = yh::cudaMallocInit<float>(memSize);


    timer.tick(timer_cp_host_to_device);
    yh::push(da, matA, memSize);
    yh::push(db, matB, memSize);
    timer.tock(timer_cp_host_to_device);

    auto blockSize = dim3(32, 32);
    auto gridSize =
        dim3{(unsigned int) (std::ceil(float(mat_cols) / blockSize.x)),
             (unsigned int) (std::ceil(float(mat_rows) / blockSize.y))};
    timer.tick(timer_kernel_g2b2);
    largeMatrixAdd_G2B2<<<gridSize, blockSize>>>(da, db, dc, mat_rows,
                                                 mat_cols);
    cudaDeviceSynchronize();
    timer.tock(timer_kernel_g2b2);

    timer.tick(timer_cp_device_to_host);
    yh::pull(ans_g2b2, dc, memSize);
    timer.tock(timer_cp_device_to_host);

    blockSize = dim3(32);
    gridSize  = {(unsigned int) (std::ceil(float(mat_cols) / blockSize.x))};
    timer.tick(timer_kernel_g1b1);
    largeMatrixAdd_G1B1<<<gridSize, blockSize>>>(da, db, dc, mat_rows,
                                                 mat_cols);
    cudaDeviceSynchronize();
    timer.tock(timer_kernel_g1b1);
    yh::pull(ans_g1b1, dc, memSize);


    blockSize = dim3(32);
    gridSize  = dim3((std::ceil(float(mat_cols) / blockSize.x)), mat_rows);
    timer.tick(timer_kernel_g2b1);
    largeMatrixAdd_G2B1<<<gridSize, blockSize>>>(da, db, dc, mat_rows,
                                                 mat_cols);
    cudaDeviceSynchronize();
    timer.tock(timer_kernel_g2b1);
    yh::pull(ans_g2b1, dc, memSize);

    //
    cudaFree(da), cudaFree(db), cudaFree(dc);

    int num_diff_g2b2 = 0;
    int num_diff_g1b1 = 0;
    int num_diff_g2b1 = 0;
    for (auto i = 0; i < num_data; ++i) {
        if (ans_host[i] != ans_g2b2[i]) num_diff_g2b2 += 1;
        if (ans_host[i] != ans_g1b1[i]) num_diff_g1b1 += 1;
        if (ans_host[i] != ans_g2b1[i]) num_diff_g2b1 += 1;
    }
    const auto err_g2b2 = static_cast<double>(num_diff_g2b2) / num_data * 100;
    printf("num_diff/total(g2b2) = %d/%d, %.6f%%\n", num_diff_g2b2, num_data,
           err_g2b2);
    const auto err_g1b1 = static_cast<double>(num_diff_g1b1) / num_data * 100;
    printf("num_diff/total(g1b1) = %d/%d, %.6f%%\n", num_diff_g1b1, num_data,
           err_g1b1);
    const auto err_g2b1 = static_cast<double>(num_diff_g2b1) / num_data * 100;
    printf("num_diff/total(g2b1) = %d/%d, %.6f%%\n", num_diff_g2b1, num_data,
           err_g2b1);

    delete[] matA;
    delete[] matB;
    delete[] ans_g2b2;
    delete[] ans_g1b1;
    delete[] ans_host;


    std::cout << timer << std::endl;

    return 0;
}
