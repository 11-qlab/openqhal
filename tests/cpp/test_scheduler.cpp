#include "qhal/scheduler.hpp"
#include "qhal/compiler.hpp"
#include <cstdio>
#include <cassert>

using namespace qhal;

int main() {
    // Build a Bell circuit
    Calibration cal = load_calibration_string("");
    Compiler compiler(cal);
    std::vector<Gate> gates = {
        {"H", {0}},
        {"CNOT", {0, 1}},
        {"MEASURE", {0, 1}}
    };
    Program prog = compiler.compile(gates);
    printf("Sequential: %.1f ns\n", prog.total_duration_ns);

    // Schedule it
    Scheduler sched;
    auto result = sched.schedule_program(prog);
    printf("Scheduled:  %.1f ns\n", result.total_duration_ns);
    printf("Improvement: %.1f ns\n", result.makespan_improvement_ns);

    printf("\nSchedule:\n");
    printf("%-4s %-10s %-10s %-8s %-6s\n",
           "ch", "start_ns", "dur_ns", "freq", "env");
    for (const auto& p : result.pulses) {
        printf("%-4d %-10.1f %-10.1f %-8.3f %-6d\n",
               p.channel, p.start_ns, p.duration_ns,
               p.frequency_hz / 1e9, static_cast<int>(p.envelope));
    }

    // Verify no same-channel overlap
    for (size_t i = 0; i < result.pulses.size(); ++i) {
        for (size_t j = i + 1; j < result.pulses.size(); ++j) {
            const auto& a = result.pulses[i];
            const auto& b = result.pulses[j];
            if (a.channel == b.channel &&
                a.duration_ns > 0.0 && b.duration_ns > 0.0) {
                double a_end = a.start_ns + a.duration_ns;
                double b_end = b.start_ns + b.duration_ns;
                bool overlap = a.start_ns < b_end && b.start_ns < a_end;
                assert(!overlap);
            }
        }
    }
    printf("\nno same-channel overlaps: OK\n");

    printf("\nall scheduler tests passed\n");
    return 0;
}
