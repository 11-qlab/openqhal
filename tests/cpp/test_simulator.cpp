#include "qhal/simulator.hpp"
#include "qhal/compiler.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>

using namespace qhal;

int main() {
    // --- Test 1: Bell state ---
    {
        Simulator sim(2, 42);
        std::vector<Gate> gates = {
            {"H", {0}},
            {"CNOT", {0, 1}}
        };
        sim.run(gates);
        // After H+CNOT, state is (|00> + |11>)/√2
        // Probabilities: P(00) = 0.5, P(11) = 0.5, P(01) = P(10) = 0
        double p00 = std::norm(sim.state()[0]);
        double p11 = std::norm(sim.state()[3]);
        printf("Bell state: P(00)=%.4f  P(11)=%.4f\n", p00, p11);
        assert(std::abs(p00 - 0.5) < 1e-9);
        assert(std::abs(p11 - 0.5) < 1e-9);
    }

    // --- Test 2: Measurement of Bell state ---
    {
        Simulator sim(2, 42);
        std::vector<Gate> gates = {
            {"H", {0}},
            {"CNOT", {0, 1}},
            {"MEASURE", {0, 1}}
        };
        auto results = sim.run(gates);
        printf("Bell measurement: [%d, %d]\n", results[0], results[1]);
        // Results must be correlated: 00 or 11
        assert(results[0] == results[1]);
    }

    // --- Test 3: |0> measures 0 ---
    {
        Simulator sim(1, 42);
        std::vector<Gate> gates = { {"MEASURE", {0}} };
        auto results = sim.run(gates);
        assert(results[0] == 0);
        printf("|0> measurement: %d\n", results[0]);
    }

    // --- Test 4: X|0> measures 1 ---
    {
        Simulator sim(1, 42);
        std::vector<Gate> gates = {
            {"X", {0}},
            {"MEASURE", {0}}
        };
        auto results = sim.run(gates);
        assert(results[0] == 1);
        printf("X|0> measurement: %d\n", results[0]);
    }

    // --- Test 5: Statistical distribution over many shots ---
    {
        int zeros = 0, ones = 0;
        for (int trial = 0; trial < 1000; ++trial) {
            Simulator sim(1, 1000 + trial);
            std::vector<Gate> gates = {
                {"H", {0}},
                {"MEASURE", {0}}
            };
            auto r = sim.run(gates);
            if (r[0] == 0) ++zeros; else ++ones;
        }
        printf("H|0> over 1000 shots: zeros=%d ones=%d\n", zeros, ones);
        assert(zeros > 400 && ones > 400);
    }

    printf("\nall simulator tests passed\n");
    return 0;
}
