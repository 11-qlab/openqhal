#pragma once
#include "qhal.h"
#include <string>
#include <vector>
#include <complex>
#include <cstdint>

namespace qhal {

using Complex = std::complex<double>;

// Envelope shapes (matches qhal_envelope_t)
enum class Envelope : int32_t {
    Square   = QHAL_ENV_SQUARE,
    Gaussian = QHAL_ENV_GAUSSIAN,
    DRAG     = QHAL_ENV_DRAG,
    Sech     = QHAL_ENV_SECH,
    Hermite  = QHAL_ENV_HERMITE,
    Custom   = QHAL_ENV_CUSTOM
};

// Channels (matches qhal_channel_t)
enum class Channel : int32_t {
    I       = QHAL_CH_I,
    Q       = QHAL_CH_Q,
    Laser   = QHAL_CH_LASER,
    Readout = QHAL_CH_READOUT,
    Flux    = QHAL_CH_FLUX
};

// Status (matches qhal_status_t)
enum class Status : int32_t {
    OK             = QHAL_OK,
    Generic        = QHAL_ERR_GENERIC,
    InvalidArg     = QHAL_ERR_INVALID_ARG,
    NoMemory       = QHAL_ERR_NO_MEMORY,
    UnknownGate    = QHAL_ERR_UNKNOWN_GATE,
    Calibration    = QHAL_ERR_CALIBRATION,
    Scheduling     = QHAL_ERR_SCHEDULING,
    Backend        = QHAL_ERR_BACKEND,
    IO             = QHAL_ERR_IO,
    Version        = QHAL_ERR_VERSION
};

struct Gate {
    std::string              name;
    std::vector<int32_t>     qubits;
    std::vector<double>      params;

    Gate() = default;
    Gate(std::string n, std::vector<int32_t> q,
         std::vector<double> p = {})
        : name(std::move(n)), qubits(std::move(q)), params(std::move(p)) {}
};

struct Pulse {
    Channel  channel       = Channel::I;
    double   start_ns      = 0.0;
    double   duration_ns   = 0.0;
    double   frequency_hz  = 0.0;
    double   amplitude     = 1.0;
    double   phase_rad     = 0.0;
    Envelope envelope      = Envelope::Square;
    double   drag_beta     = 0.0;
    double   envelope_param = 0.2;   // σ fraction for Gaussian

    // C++-only metadata (not part of the C ABI):
    int32_t  qubit   = -1;   // which qubit this pulse acts on
    int32_t  gate_id = -1;   // which logical gate produced it
};

struct Program {
    std::vector<Pulse> pulses;
    double             total_duration_ns = 0.0;
    int32_t            shot_count        = 1;
};

struct Device {
    std::string device_name;
    int32_t     qubit_count      = 0;
    std::string calibration_json;
};

// Conversion: C++ → C ABI
inline qhal_gate_t to_c(const Gate& g) {
    qhal_gate_t out{};
    out.name = g.name.c_str();
    out.qubit_count = static_cast<int32_t>(g.qubits.size());
    for (size_t i = 0; i < g.qubits.size() && i < QHAL_MAX_QUBITS_PER_GATE; ++i)
        out.qubits[i] = g.qubits[i];
    for (size_t i = 0; i < g.params.size() && i < QHAL_MAX_GATE_PARAMS; ++i)
        out.params[i] = g.params[i];
    return out;
}

inline qhal_pulse_t to_c(const Pulse& p) {
    qhal_pulse_t out{};
    out.channel        = static_cast<int32_t>(p.channel);
    out.start_ns       = p.start_ns;
    out.duration_ns    = p.duration_ns;
    out.frequency_hz   = p.frequency_hz;
    out.amplitude      = p.amplitude;
    out.phase_rad      = p.phase_rad;
    out.envelope       = static_cast<qhal_envelope_t>(p.envelope);
    out.drag_beta      = p.drag_beta;
    out.envelope_param = p.envelope_param;
    return out;
}

inline qhal_device_t to_c(const Device& d) {
    qhal_device_t out{};
    out.device_name      = d.device_name.c_str();
    out.qubit_count      = d.qubit_count;
    out.calibration_json = d.calibration_json.c_str();
    return out;
}

} // namespace qhal
