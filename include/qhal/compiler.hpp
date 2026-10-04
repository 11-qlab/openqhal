#pragma once
#include "types.hpp"
#include "calibration.hpp"
#include <vector>
#include <string>

namespace qhal {

// The compiler translates a circuit (list of Gate) into a program (list of Pulse).
class Compiler {
public:
    explicit Compiler(Calibration cal) : cal_(std::move(cal)) {}

    // Main entry point. Throws std::runtime_error on failure.
    Program compile(const std::vector<Gate>& gates);

    // Access to calibration (for CLI inspection)
    const Calibration& calibration() const { return cal_; }

private:
    Calibration cal_;

    // Single-gate translation. Appends pulses starting at t_start.
    // Returns the duration consumed (for sequencing).
    double compile_gate(const Gate& g, double t_start, Program& out);

    // Specific gates
    double compile_rotation(const Gate& g, double t_start, Program& out,
                            double angle_rad, char axis);
    double compile_virtual_z(const Gate& g, double t_start, Program& out,
                             double angle_rad);
    double compile_measure(const Gate& g, double t_start, Program& out);
    double compile_cnot(const Gate& g, double t_start, Program& out);

    // Helpers
    double rabi_period_ns(int32_t qubit) const;
    double rotation_duration_ns(int32_t qubit, double angle_rad) const;
};

} // namespace qhal
