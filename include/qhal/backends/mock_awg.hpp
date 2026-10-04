#pragma once
#include "qhal/backend.hpp"
#include "qhal/simulator.hpp"
#include <random>

namespace qhal {

// Simulated AWG. Accepts a rendered Waveform, returns measurement results
// by running the logical circuit through the built-in statevector
// simulator. Configurable latency to mimic real hardware.
//
// The point is not to simulate the I/Q envelope physics — that's what
// QuTiP is for. The point is to close the HAL's execution loop without
// physical hardware, so the full pipeline is testable.
class MockAWG : public Backend {
public:
    struct Config {
        int32_t  num_qubits      = 4;
        uint64_t seed            = 42;
        double   upload_ms_per_kb = 0.5;   // upload throughput
        double   trigger_ms      = 0.1;    // trigger latency
        double   acquire_ms      = 1.0;    // acquisition latency
        bool     simulate_latency = true;  // if false, no sleeps
    };

    MockAWG();
    explicit MockAWG(Config cfg);

    // Attach the original circuit so the simulator knows what to run.
    // Real backends derive this from the uploaded sequence file.
    void set_circuit(const std::vector<Gate>& gates) { gates_ = gates; }

    std::string name() const override { return "MockAWG"; }

    bool connect() override;
    void disconnect() override;
    bool is_connected() const override { return connected_; }

    bool upload(const Waveform& wf) override;
    bool trigger() override;
    std::vector<int32_t> acquire(int32_t shot_count) override;

    ExecutionStats last_stats() const override { return stats_; }

private:
    Config          cfg_;
    bool            connected_ = false;
    bool            armed_     = false;
    Waveform        uploaded_;
    std::vector<Gate> gates_;
    ExecutionStats  stats_;
};

} // namespace qhal
