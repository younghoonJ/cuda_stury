#include "libyh/cuda-common/include/common.cuh"

namespace yh::common {

    void
    NTimer::Timer_::reset() {
        status = TimerStatus::OFF;
        t_acc = std::chrono::duration<double>::zero();
    }

    void
    NTimer::Timer_::tick() {
        if (status == TimerStatus::ON) return;

        status = TimerStatus::ON;
        t_measure_start = std::chrono::high_resolution_clock::now();
    }

    NTimer::Timer_::Timer_(const char *name) : name_(name) {
        reset();
    }

    void
    NTimer::Timer_::tock() {
        if (status == TimerStatus::OFF) return;

        status = TimerStatus::OFF;
        t_acc += (std::chrono::high_resolution_clock::now() - t_measure_start);
    }

    void
    NTimer::reset() {
        for (auto &timer: timers)
            timer.reset();
    }

    size_t
    NTimer::addTimter(const char *timer_name) {
        timers.emplace_back(timer_name);
        return timers.size() - 1;
    }

    void
    NTimer::tick(size_t timer_id) {
        timers.at(timer_id).tick();
    }

    void
    NTimer::tock(size_t timer_id) {
        timers.at(timer_id).tock();
    }
}  // namespace yh::cuda-common