//
// Created by younghoon on 24. 1. 29.
//

#ifndef LIBYH_CUDA_COMMON_INCLUDE_LIBYH_INCLUDE_CUDA_COMMON_COMMON_CUH
#define LIBYH_CUDA_COMMON_INCLUDE_LIBYH_INCLUDE_CUDA_COMMON_COMMON_CUH

#include <chrono>
#include <iostream>
#include <vector>

namespace yh::common {
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
            printf("cudaMalloc::cudaMalloc: %s\n", cudaGetErrorName(err));
        }
        err = cudaMemset(ptr_, 0, memSize);
        if (PRINT_ERR_LOG) {
            printf("cudaMalloc::cudaMemset: %s\n", cudaGetErrorName(err));
        }
        return ptr_;
    }

    inline void
    push(void *dst, const void *src, size_t memSize) {
        const auto err = cudaMemcpy(dst, src, memSize, cudaMemcpyHostToDevice);
        if (PRINT_ERR_LOG) {
            printf("push:cudaMemcpy: %s\n", cudaGetErrorName(err));
        }
    }

    inline void
    pull(void *dst, const void *src, size_t memSize) {
        const auto err = cudaMemcpy(dst, src, memSize, cudaMemcpyDeviceToHost);
        if (PRINT_ERR_LOG) {
            printf("pull:cudaMemcpy: %s\n", cudaGetErrorName(err));
        }
    }

    class NTimer {
        enum class TimerStatus {
            ON = 0,
            OFF = 1,
        };

        class Timer_ {
            TimerStatus status = TimerStatus::OFF;
            std::chrono::time_point <std::chrono::high_resolution_clock>
                    t_measure_start;
            std::chrono::duration<double> t_acc{};
            const char *name_;

        public:
            explicit Timer_(const char *name);

            void reset();

            void tick();

            void tock();

            friend std::ostream &operator<<(std::ostream &os, const Timer_ &obj) {
                const auto t_mills =
                        static_cast<double>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        obj.t_acc)
                                        .count()) /
                        1000000.0;
                return os << obj.name_ << ": " << t_mills << " ms";
            }
        };

        std::vector <Timer_> timers;

    public:
        NTimer() = default;

        void reset();

        /* Returns the index of the timer just added. */
        size_t addTimter(const char *timer_name);

        void tick(size_t timer_id);

        void tock(size_t timer_id);

        friend std::ostream &operator<<(std::ostream &os, const NTimer &obj) {
            os << "Timer Report[num_timers=" << obj.timers.size() << "]\n";
            for (auto i = 0; i < obj.timers.size(); ++i)
                os << "  " << i << ". " << obj.timers[i] << '\n';
            return os;
        };
    };
}  // namespace yh::cuda-common


#endif //LIBYH_CUDA_COMMON_INCLUDE_LIBYH_INCLUDE_CUDA_COMMON_COMMON_CUH
