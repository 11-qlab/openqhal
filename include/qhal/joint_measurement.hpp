#pragma once
#include "qhal/backend.hpp"
#include <vector>
#include <memory>
#include <string>

namespace qhal {

// Result of a joint measurement across two backends.
struct JointResult {
    // Bits observed at each endpoint, in the order the endpoints were given.
    std::vector<int32_t> alice_bits;
    std::vector<int32_t> bob_bits;

    // Joint parity: XOR of all bits. For a Bell measurement, this is the
    // key output — it tells the correction layer what to apply.
    int32_t parity = 0;

    // Optional: an identifier the correction layer can route on.
    std::string correlation_tag;

    // Number of shots this result represents.
    int32_t shots = 1;
};

// A JointMeasurement composes two Backend objects and returns their
// correlated results. It does NOT require the two backends to share
// physical entanglement — that's a property of the hardware, not the
// software. This class only defines the contract.
//
// The Bell measurement, in practice, is:
//   1. Alice's backend acquires her half.
//   2. Bob's backend acquires his half.
//   3. The results are XORed.
//   4. The parity is dispatched to a correction layer.
class JointMeasurement {
public:
    // `alice` and `bob` are borrowed; caller retains ownership.
    JointMeasurement(Backend* alice, Backend* bob);

    // Run one joint measurement cycle. Both backends must already be
    // connected, uploaded, and triggered.
    JointResult acquire(int32_t shots);

    // Inject a simulated latency (seconds) between Alice's and Bob's
    // acquisitions. Models the classical channel delay.
    void set_classical_latency_s(double latency_s) { latency_s_ = latency_s; }
    double classical_latency_s() const { return latency_s_; }

private:
    Backend* alice_;
    Backend* bob_;
    double   latency_s_ = 0.0;
};

} // namespace qhal
