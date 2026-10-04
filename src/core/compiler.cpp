#include "qhal/compiler.hpp"
#include "qhal/envelope.hpp"
#include <cmath>
#include <stdexcept>

namespace qhal {

static constexpr double PI = 3.14159265358979323846;

double Compiler::rabi_period_ns(int32_t qubit) const {
    const auto& qc = cal_.get(qubit);
    return 1e9 / qc.rabi_hz;
}

double Compiler::rotation_duration_ns(int32_t qubit, double angle_rad) const {
    const auto& qc = cal_.get(qubit);
    return (angle_rad / (2.0 * PI * qc.rabi_hz)) * 1e9;
}

double Compiler::compile_rotation(const Gate& g, double t_start,
                                  Program& out,
                                  double angle_rad, char axis) {
    const auto& qc = cal_.get(g.qubits[0]);
    double tau_ns = rotation_duration_ns(g.qubits[0], angle_rad);

    Pulse p;
    // Each qubit gets its own I/Q pair. Qubit q → channels 2q, 2q+1.
    p.channel        = static_cast<Channel>(2 * g.qubits[0]);
    p.start_ns       = t_start;
    p.duration_ns    = tau_ns;
    p.frequency_hz   = qc.drive_freq_hz;
    p.amplitude      = 1.0;
    p.phase_rad      = (axis == 'Y') ? PI / 2.0 : 0.0;
    p.envelope       = Envelope::DRAG;
    p.drag_beta      = qc.drag_beta;
    p.envelope_param = qc.sigma_frac;
    p.qubit          = g.qubits[0];   // <- tag qubit here
    // gate_id is assigned by compile() after the gate is fully emitted

    out.pulses.push_back(p);
    return tau_ns;
}

double Compiler::compile_virtual_z(const Gate& g, double t_start,
                                   Program& out, double angle_rad) {
    const auto& qc = cal_.get(g.qubits[0]);
    if (qc.virtual_z) {
        Pulse p;
        p.channel      = Channel::I;
        p.start_ns     = t_start;
        p.duration_ns  = 0.0;
        p.frequency_hz = 0.0;
        p.amplitude    = 0.0;
        p.phase_rad    = angle_rad;
        p.envelope     = Envelope::Square;
        p.qubit        = g.qubits[0];
        out.pulses.push_back(p);
        return 0.0;
    }
    double half = compile_rotation(g, t_start, out, PI / 2.0, 'X');
    double full = compile_rotation(g, t_start + half, out, angle_rad, 'Z');
    return half + full;
}

double Compiler::compile_measure(const Gate& g, double t_start,
                                 Program& out) {
    // Emit ONE global readout pulse for the whole measurement gate.
    // Real NV hardware uses a single laser shot that reads all NVs at once.
    double max_dur = 0.0;
    for (int32_t q : g.qubits) {
        max_dur = std::max(max_dur, cal_.get(q).readout_duration_ns);
    }
    Pulse p;
    p.channel      = Channel::Readout;
    p.start_ns     = t_start;
    p.duration_ns  = max_dur;
    p.frequency_hz = 0.0;
    p.amplitude    = 1.0;
    p.phase_rad    = 0.0;
    p.envelope     = Envelope::Square;
    p.qubit        = -1;   // global
    out.pulses.push_back(p);
    return max_dur;
}

double Compiler::compile_cnot(const Gate& g, double t_start,
                              Program& out) {
    if (g.qubits.size() != 2)
        throw std::runtime_error("CNOT requires exactly 2 qubits");

    int32_t ctrl = g.qubits[0];
    int32_t tgt  = g.qubits[1];

    // Placeholder CNOT model: π on control, then π on target.
    // Real hardware uses hyperfine-mediated sequences.
    Gate gc{"X", {ctrl}};
    double d1 = compile_rotation(gc, t_start, out, PI, 'X');

    Gate gt{"X", {tgt}};
    double d2 = compile_rotation(gt, t_start + d1, out, PI, 'X');

    return d1 + d2;
}

double Compiler::compile_gate(const Gate& g, double t_start, Program& out) {
    const std::string& name = g.name;

    if (name == "I") return 0.0;

    if (name == "X")  return compile_rotation(g, t_start, out, PI, 'X');
    if (name == "Y")  return compile_rotation(g, t_start, out, PI, 'Y');
    if (name == "Z")  return compile_virtual_z(g, t_start, out, PI);
    if (name == "H") {
        double d1 = compile_rotation(g, t_start, out, PI / 2.0, 'Y');
        double d2 = compile_virtual_z(g, t_start + d1, out, PI);
        return d1 + d2;
    }
    if (name == "S")  return compile_virtual_z(g, t_start, out, PI / 2.0);
    if (name == "T")  return compile_virtual_z(g, t_start, out, PI / 4.0);

    if (name == "RX") return compile_rotation(g, t_start, out, g.params.at(0), 'X');
    if (name == "RY") return compile_rotation(g, t_start, out, g.params.at(0), 'Y');
    if (name == "RZ") return compile_virtual_z(g, t_start, out, g.params.at(0));

    if (name == "CNOT" || name == "CX") return compile_cnot(g, t_start, out);

    if (name == "MEASURE" || name == "M") return compile_measure(g, t_start, out);

    throw std::runtime_error("Compiler: unknown gate '" + name + "'");
}

Program Compiler::compile(const std::vector<Gate>& gates) {
    Program prog;
    double t = 0.0;

    for (size_t i = 0; i < gates.size(); ++i) {
        size_t start_idx = prog.pulses.size();
        double dur = compile_gate(gates[i], t, prog);
        // Tag every pulse emitted by this gate with the gate index
        for (size_t k = start_idx; k < prog.pulses.size(); ++k) {
            prog.pulses[k].gate_id = static_cast<int32_t>(i);
        }
        t += dur;
    }

    prog.total_duration_ns = t;
    prog.shot_count = 1;
    return prog;
}

} // namespace qhal
