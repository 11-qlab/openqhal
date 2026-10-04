#include "qhal/compiler.hpp"
#include <cstdio>
#include <cassert>

using namespace qhal;

int main() {
    // Default calibration: 4 qubits, 10 MHz Rabi, 2.87 GHz drive
    Calibration cal = load_calibration_string("");
    Compiler compiler(cal);

    // --- Test 1: single H ---
    {
        std::vector<Gate> gates = { {"H", {0}} };
        auto prog = compiler.compile(gates);
        assert(prog.pulses.size() == 2);   // one π/2 rotation + one virtual Z
        assert(prog.total_duration_ns > 0.0);
        printf("H(0)         → %zu pulses, %.1f ns\n",
               prog.pulses.size(), prog.total_duration_ns);
    }

    // --- Test 2: X ---
    {
        std::vector<Gate> gates = { {"X", {0}} };
        auto prog = compiler.compile(gates);
        assert(prog.pulses.size() == 1);
        // π pulse at 10 MHz Rabi → 50 ns
        double expected = 1e9 / (2.0 * 10e6);
        assert(std::abs(prog.total_duration_ns - expected) < 1e-6);
        printf("X(0)         → %zu pulses, %.1f ns (expect %.1f)\n",
               prog.pulses.size(), prog.total_duration_ns, expected);
    }

    // --- Test 3: Z is virtual (zero duration) ---
    {
        std::vector<Gate> gates = { {"Z", {0}} };
        auto prog = compiler.compile(gates);
        assert(prog.pulses.size() == 1);
        assert(prog.total_duration_ns == 0.0);
        printf("Z(0)         → %zu pulses, %.1f ns (virtual)\n",
               prog.pulses.size(), prog.total_duration_ns);
    }

    // --- Test 4: Bell state circuit ---
    {
        std::vector<Gate> gates = {
            {"H", {0}},
            {"CNOT", {0, 1}},
            {"MEASURE", {0, 1}}
        };
        auto prog = compiler.compile(gates);
        assert(prog.pulses.size() >= 5);
        printf("Bell circuit → %zu pulses, %.1f ns\n",
               prog.pulses.size(), prog.total_duration_ns);

        // Print the schedule
        for (const auto& p : prog.pulses) {
            printf("  ch=%d t=%7.1f dur=%6.1f f=%.3f GHz amp=%.2f env=%d\n",
                   static_cast<int>(p.channel), p.start_ns, p.duration_ns,
                   p.frequency_hz / 1e9, p.amplitude,
                   static_cast<int>(p.envelope));
        }
    }

    // --- Test 5: unknown gate throws ---
    {
        std::vector<Gate> gates = { {"FOO", {0}} };
        bool threw = false;
        try { compiler.compile(gates); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
        printf("FOO(0)       → correctly threw\n");
    }

    printf("\nall compiler tests passed\n");
    return 0;
}
