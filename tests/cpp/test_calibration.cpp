#include "qhal/calibration.hpp"
#include <cstdio>
#include <cassert>

using namespace qhal;

int main() {
    // Load from file
    auto cal = load_calibration_json("calibration/nv_template.json");
    printf("device: %s\n", cal.device_name.c_str());
    printf("qubits: %zu\n", cal.qubits.size());

    const auto& q0 = cal.get(0);
    printf("q0: rabi=%.3e Hz  drive=%.3e Hz  t1=%.3e ns\n",
           q0.rabi_hz, q0.drive_freq_hz, q0.t1_ns);
    assert(std::abs(q0.rabi_hz - 10e6) < 1e-6);
    assert(std::abs(q0.drive_freq_hz - 2.87e9) < 1e-6);

    const auto& q1 = cal.get(1);
    printf("q1: rabi=%.3e Hz  drive=%.3e Hz\n",
           q1.rabi_hz, q1.drive_freq_hz);
    assert(std::abs(q1.rabi_hz - 9.5e6) < 1e-6);

    // Round-trip
    save_calibration_json(cal, "/tmp/roundtrip.json");
    auto cal2 = load_calibration_json("/tmp/roundtrip.json");
    assert(cal2.device_name == cal.device_name);
    assert(std::abs(cal2.get(0).rabi_hz - q0.rabi_hz) < 1e-6);

    printf("\nall calibration tests passed\n");
    return 0;
}
