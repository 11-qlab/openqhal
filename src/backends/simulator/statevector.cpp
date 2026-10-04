#include <map>
#include <set>
#include "qhal/simulator.hpp"
#include <cmath>
#include <stdexcept>

namespace qhal {

static constexpr double PI = 3.14159265358979323846;
static const Complex I(0.0, 1.0);

// ---------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------
Simulator::Simulator(int32_t num_qubits, uint64_t seed)
    : num_qubits_(num_qubits), rng_(seed) {
    if (num_qubits <= 0 || num_qubits > 24)
        throw std::invalid_argument("Simulator: qubit count must be 1..24");
    size_t dim = size_t(1) << num_qubits;
    state_.assign(dim, Complex(0.0, 0.0));
    state_[0] = Complex(1.0, 0.0);   // |0...0>
}

// ---------------------------------------------------------------------
// Single-qubit gate application
// ---------------------------------------------------------------------
void Simulator::apply_single_qubit(int32_t q, const Complex m[2][2]) {
    size_t dim = size_t(1) << num_qubits_;
    size_t mask = size_t(1) << q;
    for (size_t i = 0; i < dim; ++i) {
        if (i & mask) continue;
        size_t j = i | mask;
        Complex a = state_[i];
        Complex b = state_[j];
        state_[i] = m[0][0] * a + m[0][1] * b;
        state_[j] = m[1][0] * a + m[1][1] * b;
    }
}

// ---------------------------------------------------------------------
// Controlled-U
// ---------------------------------------------------------------------
void Simulator::apply_controlled(int32_t ctrl, int32_t tgt,
                                 const Complex m[2][2]) {
    size_t dim = size_t(1) << num_qubits_;
    size_t cmask = size_t(1) << ctrl;
    size_t tmask = size_t(1) << tgt;
    for (size_t i = 0; i < dim; ++i) {
        if (!(i & cmask)) continue;   // control is 0 → skip
        if (i & tmask) continue;      // only touch |..0..> states
        size_t j = i | tmask;
        Complex a = state_[i];
        Complex b = state_[j];
        state_[i] = m[0][0] * a + m[0][1] * b;
        state_[j] = m[1][0] * a + m[1][1] * b;
    }
}

void Simulator::apply_cnot(int32_t ctrl, int32_t tgt) {
    size_t dim = size_t(1) << num_qubits_;
    size_t cmask = size_t(1) << ctrl;
    size_t tmask = size_t(1) << tgt;
    for (size_t i = 0; i < dim; ++i) {
        if (!(i & cmask)) continue;
        if (i & tmask) continue;
        size_t j = i | tmask;
        std::swap(state_[i], state_[j]);
    }
}

void Simulator::apply_cz(int32_t ctrl, int32_t tgt) {
    size_t dim = size_t(1) << num_qubits_;
    size_t cmask = size_t(1) << ctrl;
    size_t tmask = size_t(1) << tgt;
    for (size_t i = 0; i < dim; ++i) {
        if ((i & cmask) && (i & tmask)) state_[i] = -state_[i];
    }
}

void Simulator::apply_swap(int32_t a, int32_t b) {
    size_t dim = size_t(1) << num_qubits_;
    size_t amask = size_t(1) << a;
    size_t bmask = size_t(1) << b;
    for (size_t i = 0; i < dim; ++i) {
        bool abit = i & amask;
        bool bbit = i & bmask;
        if (!abit && bbit) {
            size_t j = (i | amask) & ~bmask;
            std::swap(state_[i], state_[j]);
        }
    }
}

// ---------------------------------------------------------------------
// Gate dispatch
// ---------------------------------------------------------------------
void Simulator::apply_gate(const Gate& g) {
    const std::string& n = g.name;

    if (n == "I") return;

    // Pauli and Clifford gates
    if (n == "X") {
        Complex m[2][2] = {{0,0},{1,0}};
        m[0][0] = 0; m[0][1] = 1;
        m[1][0] = 1; m[1][1] = 0;
        apply_single_qubit(g.qubits[0], m);
        return;
    }
    if (n == "Y") {
        Complex m[2][2];
        m[0][0] = 0; m[0][1] = -I;
        m[1][0] = I; m[1][1] = 0;
        apply_single_qubit(g.qubits[0], m);
        return;
    }
    if (n == "Z") {
        Complex m[2][2];
        m[0][0] = 1; m[0][1] = 0;
        m[1][0] = 0; m[1][1] = -1;
        apply_single_qubit(g.qubits[0], m);
        return;
    }
    if (n == "H") {
        double s = 1.0 / std::sqrt(2.0);
        Complex m[2][2];
        m[0][0] = s; m[0][1] = s;
        m[1][0] = s; m[1][1] = -s;
        apply_single_qubit(g.qubits[0], m);
        return;
    }
    if (n == "S") {
        Complex m[2][2];
        m[0][0] = 1; m[0][1] = 0;
        m[1][0] = 0; m[1][1] = I;
        apply_single_qubit(g.qubits[0], m);
        return;
    }
    if (n == "T") {
        Complex m[2][2];
        m[0][0] = 1; m[0][1] = 0;
        m[1][0] = 0; m[1][1] = std::exp(I * PI / 4.0);
        apply_single_qubit(g.qubits[0], m);
        return;
    }

    // Rotations
    if (n == "RX" || n == "RY" || n == "RZ") {
        double theta = g.params.at(0);
        double c = std::cos(theta / 2.0);
        double s = std::sin(theta / 2.0);
        Complex m[2][2];
        if (n == "RX") {
            m[0][0] = c;        m[0][1] = -I * s;
            m[1][0] = -I * s;   m[1][1] = c;
        } else if (n == "RY") {
            m[0][0] = c;        m[0][1] = -s;
            m[1][0] = s;        m[1][1] = c;
        } else {
            m[0][0] = std::exp(-I * theta / 2.0); m[0][1] = 0;
            m[1][0] = 0; m[1][1] = std::exp(I * theta / 2.0);
        }
        apply_single_qubit(g.qubits[0], m);
        return;
    }

    // Two-qubit
    if (n == "CNOT" || n == "CX") {
        apply_cnot(g.qubits[0], g.qubits[1]);
        return;
    }
    if (n == "CZ") {
        apply_cz(g.qubits[0], g.qubits[1]);
        return;
    }
    if (n == "SWAP") {
        apply_swap(g.qubits[0], g.qubits[1]);
        return;
    }

    if (n == "MEASURE" || n == "M") return;   // handled by run()

    throw std::runtime_error("Simulator: unsupported gate '" + n + "'");
}

// ---------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------
int32_t Simulator::sample_qubit(int32_t q) {
    size_t dim = size_t(1) << num_qubits_;
    size_t mask = size_t(1) << q;
    double p1 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        if (i & mask) p1 += std::norm(state_[i]);
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return (u(rng_) < p1) ? 1 : 0;
}

void Simulator::collapse(int32_t q, int32_t outcome) {
    size_t dim = size_t(1) << num_qubits_;
    size_t mask = size_t(1) << q;
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        bool bit = (i & mask) ? 1 : 0;
        if (bit != outcome) state_[i] = 0.0;
        else norm += std::norm(state_[i]);
    }
    if (norm > 0.0) {
        double inv = 1.0 / std::sqrt(norm);
        for (auto& a : state_) a *= inv;
    }
}

std::vector<int32_t> Simulator::measure(const std::vector<int32_t>& qubits) {
    std::vector<int32_t> out;
    out.reserve(qubits.size());
    for (int32_t q : qubits) {
        int32_t bit = sample_qubit(q);
        collapse(q, bit);
        out.push_back(bit);
    }
    return out;
}

// ---------------------------------------------------------------------
// Circuit execution
// ---------------------------------------------------------------------
std::vector<int32_t> Simulator::run(const std::vector<Gate>& gates) {
    std::vector<int32_t> results;
    for (const auto& g : gates) {
        if (g.name == "MEASURE" || g.name == "M") {
            auto bits = measure(g.qubits);
            results.insert(results.end(), bits.begin(), bits.end());
        } else {
            apply_gate(g);
        }
    }
    return results;
}

std::vector<int32_t> Simulator::run_program(const Program& prog) {
    // Reconstruct logical gates from pulse sequence.
    // Group pulses by gate_id and infer the operation.
    std::vector<int32_t> results;

    // Map gate_id -> list of pulses
    std::map<int32_t, std::vector<const Pulse*>> gate_map;
    for (const auto& p : prog.pulses) {
        if (p.gate_id < 0) continue;
        gate_map[p.gate_id].push_back(&p);
    }

    for (auto& [gid, pulses] : gate_map) {
        // Readout pulse → MEASURE
        bool has_readout = false;
        for (const auto* p : pulses) {
            if (p->channel == Channel::Readout) has_readout = true;
        }
        if (has_readout) {
            std::set<int32_t> qs;
            for (const auto* p : pulses) {
                if (p->channel == Channel::Readout && p->qubit >= 0) {
                    qs.insert(p->qubit);
                }
            }
            std::vector<int32_t> qubits(qs.begin(), qs.end());
            auto bits = measure(qubits);
            results.insert(results.end(), bits.begin(), bits.end());
            continue;
        }

        // Drive pulses → infer angle and axis
        for (const auto* p : pulses) {
            if (p->channel != Channel::I) continue;
            if (p->duration_ns == 0.0) continue;   // virtual Z, skip
            if (p->qubit < 0) continue;
            // angle = 2π · freq · duration (in seconds)
            double t_sec = p->duration_ns * 1e-9;
            double angle = 2.0 * PI * p->frequency_hz * t_sec;
            // Round to nearest known gate
            double a_pi = angle / PI;   // in units of π
            Gate g;
            if (std::abs(a_pi - 1.0) < 0.01)      g.name = "X";
            else if (std::abs(a_pi - 0.5) < 0.01) g.name = "H";
            else continue;
            g.qubits = {p->qubit};
            apply_gate(g);
        }
    }

    return results;
}

double Simulator::prob_zero() const {
    return std::norm(state_[0]);
}

} // namespace qhal
