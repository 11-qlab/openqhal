#include "qhal/qhal.h"
#include "qhal/compiler.hpp"
#include "qhal/scheduler.hpp"
#include "qhal/calibration.hpp"
#include "qhal/simulator.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace qhal;

// ---------------------------------------------------------------------
// Minimal circuit parser.
// Accepts lines like:
//   H 0
//   CNOT 0 1
//   MEASURE 0 1
//   RX(1.5708) 0
// ---------------------------------------------------------------------
static bool parse_line(const std::string& line,
                       std::string& gate_name,
                       std::vector<int32_t>& qubits,
                       std::vector<double>& params) {
    if (line.empty() || line[0] == '#') return false;

    std::string s = line;
    auto lp = s.find('(');
    auto rp = s.find(')');
    if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
        std::string param_str = s.substr(lp + 1, rp - lp - 1);
        std::stringstream ps(param_str);
        std::string tok;
        while (std::getline(ps, tok, ',')) params.push_back(std::stod(tok));
        s = s.substr(0, lp) + s.substr(rp + 1);
    }

    std::stringstream ss(s);
    std::string tok;
    if (!(ss >> gate_name)) return false;

    int q;
    while (ss >> q) qubits.push_back(q);

    return !gate_name.empty();
}

int main(int argc, char** argv) {
    std::string calibration_path;
    std::string input_path;
    bool verbose    = false;
    bool do_schedule = false;
    bool do_simulate = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-c" && i + 1 < argc) calibration_path = argv[++i];
        else if (a == "-v") verbose = true;
        else if (a == "-s" || a == "--schedule") do_schedule = true;
        else if (a == "--simulate") do_simulate = true;
        else if (a == "-h" || a == "--help") {
            std::printf(
                "qhalc — quantum pulse compiler\n"
                "Usage: qhalc [-c calibration.json] [-s] [-v] [circuit_file]\n"
                "  -s, --schedule   pack pulses for parallel execution\n"
                "  -v               verbose output\n"
                "  --simulate       run the circuit on the built-in simulator\n"
                "If no file is given, reads from stdin.\n");
            return 0;
        }
        else input_path = a;
    }

    std::istream* in = &std::cin;
    std::ifstream fin;
    if (!input_path.empty()) {
        fin.open(input_path);
        if (!fin) { std::fprintf(stderr, "qhalc: cannot open %s\n", input_path.c_str()); return 1; }
        in = &fin;
    }

    // Parse the circuit
    std::vector<Gate> gates;
    std::string line;
    while (std::getline(*in, line)) {
        std::string gname;
        std::vector<int32_t> qubits;
        std::vector<double> params;
        if (!parse_line(line, gname, qubits, params)) continue;
        Gate g;
        g.name   = gname;
        g.qubits = qubits;
        g.params = params;
        gates.push_back(std::move(g));
    }

    // Build calibration
    Calibration cal;
    if (!calibration_path.empty()) {
        cal = load_calibration_json(calibration_path);
    } else {
        cal = load_calibration_string("");
    }

    // Compile
    Compiler compiler(cal);
    Program prog;
    try {
        prog = compiler.compile(gates);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qhalc: compile failed: %s\n", e.what());
        return 1;
    }

    std::printf("qhalc — compiled %zu gates → %zu pulses\n",
                gates.size(), prog.pulses.size());
    std::printf("Sequential duration: %.1f ns\n", prog.total_duration_ns);

    // Schedule
    std::vector<Pulse> display_pulses = prog.pulses;
    double display_duration = prog.total_duration_ns;

    if (do_schedule) {
        Scheduler sched;
        auto result = sched.schedule_program(prog);
        display_pulses   = std::move(result.pulses);
        display_duration = result.total_duration_ns;
        std::printf("Scheduled duration:  %.1f ns\n", display_duration);
        std::printf("Improvement:         %.1f ns\n",
                    result.makespan_improvement_ns);
    }

    if (verbose) {
        std::printf("\n%-4s %-10s %-10s %-10s %-8s %-6s %-6s\n",
                    "ch", "start_ns", "dur_ns", "freq_GHz", "amp", "env", "qbit");
        for (const auto& p : display_pulses) {
            std::printf("%-4d %-10.1f %-10.1f %-10.3f %-8.2f %-6d %-6d\n",
                        static_cast<int>(p.channel), p.start_ns, p.duration_ns,
                        p.frequency_hz / 1e9, p.amplitude,
                        static_cast<int>(p.envelope), p.qubit);
        }
    }

    if (do_simulate) {
        // Run on built-in simulator
        int32_t nq = 4;
        for (const auto& g : gates)
            for (int32_t q : g.qubits)
                if (q + 1 > nq) nq = q + 1;

        Simulator sim(nq, 42);
        auto bits = sim.run(gates);
        std::printf("\nSimulation results: [");
        for (size_t i = 0; i < bits.size(); ++i) {
            std::printf("%d%s", bits[i], i + 1 < bits.size() ? ", " : "");
        }
        std::printf("]\n");
    }

    return 0;
}
