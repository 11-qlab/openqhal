#pragma once
#include "types.hpp"
#include <vector>
#include <unordered_map>

namespace qhal {

// Scheduling result: pulses with updated start_ns, and the new total duration.
struct ScheduleResult {
    std::vector<Pulse> pulses;
    double             total_duration_ns = 0.0;
    double             makespan_improvement_ns = 0.0;   // vs. sequential
};

// The scheduler takes a flat list of pulses (as emitted by the compiler)
// and packs them into time, respecting:
//   - Same-channel pulses cannot overlap.
//   - Pulses on the same qubit must serialize unless on different channels.
//   - Pulses from the same gate must serialize in the order emitted.
//
// It does NOT change pulse durations or amplitudes.
class Scheduler {
public:
    Scheduler() = default;

    // Assign each pulse a "resource key" — a (qubit, channel) pair.
    // By default, the scheduler infers the qubit from the pulse index.
    // Override this by setting pulse qubit tags before scheduling.
    struct TaggedPulse {
        Pulse   pulse;
        int32_t qubit   = -1;   // -1 = no qubit association (global)
        int32_t gate_id = -1;   // grouping for gate-level dependencies
    };

    // Main entry: schedule a set of tagged pulses.
    ScheduleResult schedule(std::vector<TaggedPulse> tagged);

    // Convenience: schedule a Program directly, using pulse ordering
    // as the only dependency hint.
    ScheduleResult schedule_program(const Program& prog);

private:
    // Build the "pulse i must come before pulse j" dependency list.
    std::vector<std::vector<int>> build_dependencies(
        const std::vector<TaggedPulse>& tagged) const;

    // Detect whether two pulses conflict on any resource.
    static bool conflicts(const TaggedPulse& a, const TaggedPulse& b);
};

} // namespace qhal
