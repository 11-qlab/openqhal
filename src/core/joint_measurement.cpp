#include "qhal/joint_measurement.hpp"
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <thread>

namespace qhal {

JointMeasurement::JointMeasurement(Backend* alice, Backend* bob)
    : alice_(alice), bob_(bob) {
    if (!alice_ || !bob_)
        throw std::invalid_argument("JointMeasurement: null backend");
}

JointResult JointMeasurement::acquire(int32_t shots) {
    if (!alice_->is_connected() || !bob_->is_connected())
        throw std::runtime_error("JointMeasurement: backend not connected");

    JointResult r;
    r.shots = shots;

    // 1. Alice acquires
    r.alice_bits = alice_->acquire(shots);

    // 2. Optional classical-channel delay between endpoints
    if (latency_s_ > 0.0) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(latency_s_));
    }

    // 3. Bob acquires
    r.bob_bits = bob_->acquire(shots);

    // 4. Joint parity: XOR of all bits from both ends.
    //    For a Bell measurement, this is the classical correction index.
    int32_t p = 0;
    for (auto b : r.alice_bits) p ^= (b & 1);
    for (auto b : r.bob_bits)   p ^= (b & 1);
    r.parity = p;

    // 5. Correlation tag — a string the correction layer can route on.
    r.correlation_tag = "bell_" + std::to_string(p);

    return r;
}

} // namespace qhal
