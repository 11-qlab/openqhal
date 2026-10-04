#pragma once
#include "types.hpp"
#include <string>
#include <unordered_map>

namespace qhal {

// Per-qubit calibration data. All physical parameters, no pulse shapes.
struct QubitCalibration {
    int32_t qubit_id              = 0;
    double  rabi_hz               = 10e6;    // Rabi frequency at amp=1
    double  anharm_hz             = 1e9;     // anharmonicity (NV: ~2.87 GHz)
    double  drive_freq_hz         = 2.87e9;  // carrier (NV zero-field)
    double  t1_ns                 = 1e6;     // relaxation time
    double  t2_ns                 = 1e5;     // dephasing time
    double  drag_beta             = 0.5;     // DRAG coefficient
    double  sigma_frac            = 0.2;     // Gaussian width fraction
    double  readout_duration_ns   = 300.0;   // laser pulse width
    double  virtual_z             = true;    // Z is frame rotation (free)
};

// Full device calibration: qubit parameters + two-qubit couplings
struct Calibration {
    std::string device_name;
    std::unordered_map<int32_t, QubitCalibration> qubits;

    // Two-qubit gate durations (CNOT, CZ, iSWAP)
    double cnot_duration_ns = 200.0;
    double cz_duration_ns   = 150.0;

    // Readout
    double measure_duration_ns = 300.0;

    const QubitCalibration& get(int32_t q) const;
    bool has(int32_t q) const { return qubits.count(q) > 0; }
};

// Load from JSON. Throws std::runtime_error on parse failure.
Calibration load_calibration_json(const std::string& path);
Calibration load_calibration_string(const std::string& json);

// Save to JSON
void save_calibration_json(const Calibration& cal, const std::string& path);

} // namespace qhal
