#pragma once
#include "types.hpp"
#include <vector>
#include <cstdint>

namespace qhal {

struct SampleRate {
    double samples_per_ns = 1.0;
    int32_t to_count(double duration_ns) const {
        return static_cast<int32_t>(duration_ns * samples_per_ns);
    }
};

std::vector<Complex> envelope_shape(
    Envelope       envelope,
    double         duration_ns,
    const SampleRate& rate,
    double         sigma_frac = 0.2,
    double         beta = 0.0,
    double         anharm_hz = 0.0,
    int32_t        hermite_order = 0
);

double pulse_area(const std::vector<Complex>& samples, const SampleRate& rate);

void normalize_for_angle(std::vector<Complex>& samples,
                         const SampleRate& rate,
                         double target_angle_rad);

} // namespace qhal
