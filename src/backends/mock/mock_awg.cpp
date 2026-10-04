#include "qhal/backends/mock_awg.hpp"
#include <chrono>
#include <thread>
#include <stdexcept>

namespace qhal {

using clock_t_ = std::chrono::steady_clock;
static double ms_between(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

MockAWG::MockAWG() : cfg_() {}
MockAWG::MockAWG(Config cfg) : cfg_(cfg) {}

bool MockAWG::connect() {
    connected_ = true;
    return true;
}

void MockAWG::disconnect() {
    connected_ = false;
    armed_ = false;
    uploaded_ = Waveform{};
}

bool MockAWG::upload(const Waveform& wf) {
    if (!connected_) return false;

    auto t0 = clock_t_::now();

    // Simulate upload bandwidth
    size_t bytes = wf.sample_count() * wf.channels.size() * sizeof(Complex);
    double kb = bytes / 1024.0;
    if (cfg_.simulate_latency) {
        double sleep_ms = kb * cfg_.upload_ms_per_kb;
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<long>(sleep_ms * 1000)));
        }
    }

    uploaded_ = wf;
    auto t1 = clock_t_::now();

    stats_.upload_ms = ms_between(t0, t1);
    stats_.samples_uploaded = wf.sample_count() * wf.channels.size();
    return true;
}

bool MockAWG::trigger() {
    if (!connected_) return false;
    if (uploaded_.empty()) return false;

    auto t0 = clock_t_::now();
    if (cfg_.simulate_latency) {
        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<long>(cfg_.trigger_ms * 1000)));
    }
    armed_ = true;
    auto t1 = clock_t_::now();
    stats_.trigger_ms = ms_between(t0, t1);
    return true;
}

std::vector<int32_t> MockAWG::acquire(int32_t shot_count) {
    std::vector<int32_t> results;
    if (!connected_ || !armed_) return results;

    auto t0 = clock_t_::now();
    if (cfg_.simulate_latency) {
        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<long>(cfg_.acquire_ms * 1000)));
    }

    // Run the logical circuit through the statevector simulator.
    // For multiple shots, aggregate the results as a histogram.
    std::vector<int32_t> last_shot;
    for (int32_t s = 0; s < shot_count; ++s) {
        Simulator sim(cfg_.num_qubits,
                      cfg_.seed + static_cast<uint64_t>(s));
        last_shot = sim.run(gates_);
    }
    results = last_shot;

    auto t1 = clock_t_::now();
    stats_.acquisition_ms = ms_between(t0, t1);
    stats_.total_ms = stats_.upload_ms + stats_.trigger_ms +
                      stats_.acquisition_ms;
    stats_.shots = shot_count;
    stats_.pulses_uploaded = static_cast<int32_t>(uploaded_.channels.size());

    armed_ = false;
    return results;
}

// Default full-cycle implementation
std::vector<int32_t> Backend::execute(const Waveform& wf, int32_t shot_count) {
    if (!upload(wf))  return {};
    if (!trigger())   return {};
    return acquire(shot_count);
}

} // namespace qhal