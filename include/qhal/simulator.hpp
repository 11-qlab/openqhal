#pragma once
#include "types.hpp"
#include <vector>
#include <complex>
#include <random>
#include <cstdint>

namespace qhal {

using Complex = std::complex<double>;
using StateVector = std::vector<Complex>;

// A minimal statevector simulator for up to ~20 qubits.
// Handles the standard gate set: I, X, Y, Z, H, S, T, RX, RY, RZ,
// CNOT, CZ, SWAP, MEASURE.
class Simulator {
public:
    explicit Simulator(int32_t num_qubits, uint64_t seed = 42);

    // Apply a gate to the state vector.
    void apply_gate(const Gate& g);

    // Measure the listed qubits, collapsing the state.
    // Returns a bit for each qubit in the list.
    std::vector<int32_t> measure(const std::vector<int32_t>& qubits);

    // Run a full circuit. Returns the measurement outcomes (one per
    // MEASURE gate encountered), in order.
    std::vector<int32_t> run(const std::vector<Gate>& gates);

    // Run a program by re-deriving logical gates from the pulse sequence.
    // This is how the HAL-level simulator works: it reads the compiled
    // pulses, infers the gate angles, and executes them.
    std::vector<int32_t> run_program(const Program& prog);

    // Access the current state (for debugging/tests).
    const StateVector& state() const { return state_; }

    // Probability of measuring |0...0> (all zeros) in the current state.
    double prob_zero() const;

private:
    int32_t num_qubits_;
    StateVector state_;
    std::mt19937_64 rng_;

    void apply_single_qubit(int32_t qubit, const Complex m[2][2]);
    void apply_controlled(int32_t ctrl, int32_t tgt, const Complex m[2][2]);
    void apply_cnot(int32_t ctrl, int32_t tgt);
    void apply_cz(int32_t ctrl, int32_t tgt);
    void apply_swap(int32_t a, int32_t b);

    int32_t sample_qubit(int32_t qubit);
    void collapse(int32_t qubit, int32_t outcome);
};

} // namespace qhal
