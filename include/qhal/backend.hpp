#pragma once
#include "types.hpp"
#include "waveform.hpp"
#include <vector>
#include <string>
#include <memory>

namespace qhal {

// Device telemetry for a single execution.
struct ExecutionStats {
    double  upload_ms      = 0.0;
    double  trigger_ms     = 0.0;
    double  acquisition_ms = 0.0;
    double  total_ms       = 0.0;
    int32_t shots          = 1;
    int32_t pulses_uploaded = 0;
    size_t  samples_uploaded = 0;
};

// Abstract device backend. Implementations wrap real AWGs or simulators.
class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;

    virtual bool connect()    = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;

    // Upload a waveform to device memory. Returns false on failure.
    virtual bool upload(const Waveform& wf) = 0;

    // Arm the trigger. Returns false if not ready.
    virtual bool trigger() = 0;

    // Block until the acquisition completes, return measurement bits.
    virtual std::vector<int32_t> acquire(int32_t shot_count) = 0;

    // Full cycle convenience.
    virtual std::vector<int32_t> execute(const Waveform& wf,
                                         int32_t shot_count);

    virtual ExecutionStats last_stats() const = 0;
};

} // namespace qhal
