#include "qhal/scheduler.hpp"
#include <algorithm>
#include <stdexcept>

namespace qhal {

namespace {
inline bool intervals_overlap(double a_start, double a_end,
                              double b_start, double b_end) {
    return a_start < b_end && b_start < a_end;
}
} // namespace

bool Scheduler::conflicts(const TaggedPulse& a, const TaggedPulse& b) {
    if (a.pulse.channel == b.pulse.channel) {
        if (a.pulse.duration_ns == 0.0 || b.pulse.duration_ns == 0.0)
            return false;
        return intervals_overlap(a.pulse.start_ns,
                                 a.pulse.start_ns + a.pulse.duration_ns,
                                 b.pulse.start_ns,
                                 b.pulse.start_ns + b.pulse.duration_ns);
    }
    return false;
}

std::vector<std::vector<int>>
Scheduler::build_dependencies(const std::vector<TaggedPulse>& tagged) const {
    int n = static_cast<int>(tagged.size());
    std::vector<std::vector<int>> deps(n);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Rule 1: same gate => j depends on i (intra-gate ordering)
            if (tagged[i].gate_id >= 0 &&
                tagged[i].gate_id == tagged[j].gate_id) {
                deps[j].push_back(i);
                continue;
            }
            // Rule 2: share a qubit => j depends on i (cross-gate qubit ordering)
            if (tagged[i].qubit >= 0 &&
                tagged[i].qubit == tagged[j].qubit) {
                deps[j].push_back(i);
                continue;
            }
            // Rule 3: same channel => serialize (channel is a shared resource)
            // Rule 3: a global pulse (qubit < 0, e.g. readout laser)
            // serializes with EVERY other pulse.
            if (tagged[i].qubit < 0 || tagged[j].qubit < 0) {
                deps[j].push_back(i);
                continue;
            }
            // Otherwise: no dependency — parallel is allowed
        }
    }
    return deps;
}

ScheduleResult Scheduler::schedule(std::vector<TaggedPulse> tagged) {
    ScheduleResult result;
    int n = static_cast<int>(tagged.size());
    if (n == 0) return result;

    auto deps = build_dependencies(tagged);
    std::vector<double> end_times(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double earliest = 0.0;
        for (int dep : deps[i]) {
            earliest = std::max(earliest, end_times[dep]);
        }
        tagged[i].pulse.start_ns = earliest;
        end_times[i] = earliest + tagged[i].pulse.duration_ns;
    }

    double makespan = 0.0;
    for (double e : end_times) makespan = std::max(makespan, e);

    double sequential = 0.0;
    for (const auto& t : tagged) sequential += t.pulse.duration_ns;

    result.pulses.reserve(n);
    for (auto& t : tagged) result.pulses.push_back(std::move(t.pulse));
    result.total_duration_ns = makespan;
    result.makespan_improvement_ns = sequential - makespan;
    return result;
}

ScheduleResult Scheduler::schedule_program(const Program& prog) {
    std::vector<TaggedPulse> tagged;
    tagged.reserve(prog.pulses.size());
    for (const auto& p : prog.pulses) {
        TaggedPulse tp;
        tp.pulse   = p;
        tp.qubit   = p.qubit;
        tp.gate_id = p.gate_id;
        tagged.push_back(std::move(tp));
    }
    return schedule(std::move(tagged));
}

} // namespace qhal
