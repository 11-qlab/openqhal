#include "qhal/qhal.h"
#include "qhal/types.hpp"
#include "qhal/compiler.hpp"
#include "qhal/calibration.hpp"
#include "qhal/simulator.hpp"
#include <map>
#include <set>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

using namespace qhal;

// ---------------------------------------------------------------------
// Thread-local error message
// ---------------------------------------------------------------------
namespace {
thread_local std::string g_last_error;
void set_error(const std::string& msg) { g_last_error = msg; }
} // namespace

extern "C" {

// ---------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------
int qhal_abi_version(void) { return QHAL_ABI_VERSION; }

const char* qhal_version(void) {
    static std::string v = std::to_string(QHAL_VERSION_MAJOR) + "." +
                           std::to_string(QHAL_VERSION_MINOR) + "." +
                           std::to_string(QHAL_VERSION_PATCH);
    return v.c_str();
}

// ---------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------
int qhal_init(void) {
    g_last_error.clear();
    return QHAL_OK;
}

void qhal_shutdown(void) {
    g_last_error.clear();
}

// ---------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------
qhal_status_t qhal_compile(const qhal_gate_t*   gates,
                           int32_t              gate_count,
                           const qhal_device_t* device,
                           qhal_program_t*      out) {
    if (!gates || gate_count < 0 || !out) {
        set_error("qhal_compile: null or invalid argument");
        return QHAL_ERR_INVALID_ARG;
    }

    try {
        // Build C++ gates
        std::vector<Gate> cpp_gates;
        cpp_gates.reserve(gate_count);
        for (int32_t i = 0; i < gate_count; ++i) {
            const auto& g = gates[i];
            if (!g.name) {
                set_error("qhal_compile: gate has null name");
                return QHAL_ERR_INVALID_ARG;
            }
            Gate cg;
            cg.name = g.name;
            for (int32_t k = 0; k < g.qubit_count; ++k)
                cg.qubits.push_back(g.qubits[k]);
            for (int32_t k = 0; k < QHAL_MAX_GATE_PARAMS; ++k) {
                if (g.params[k] != 0.0) cg.params.push_back(g.params[k]);
            }
            cpp_gates.push_back(std::move(cg));
        }

        // Build calibration. If device->calibration_json is set, parse it;
        // otherwise use defaults.
        Calibration cal;
        if (device && device->calibration_json && device->calibration_json[0]) {
            cal = load_calibration_string(device->calibration_json);
        } else {
            cal = load_calibration_string("");
            if (device && device->qubit_count > 0) {
                // Restrict to the requested qubit count
                Calibration filtered;
                filtered.device_name = device->device_name ? device->device_name : "default";
                for (auto& [qid, qc] : cal.qubits) {
                    if (qid < device->qubit_count) filtered.qubits[qid] = qc;
                }
                cal = std::move(filtered);
            }
        }

        // Compile
        Compiler compiler(cal);
        Program prog = compiler.compile(cpp_gates);

        // Convert back to C representation
        out->count = static_cast<int32_t>(prog.pulses.size());
        out->total_duration_ns = prog.total_duration_ns;
        out->shot_count = prog.shot_count;
        out->pulses = new qhal_pulse_t[out->count];

        for (int32_t i = 0; i < out->count; ++i) {
            const auto& p = prog.pulses[i];
            auto& cp = out->pulses[i];
            cp.channel        = static_cast<int32_t>(p.channel);
            cp.start_ns       = p.start_ns;
            cp.duration_ns    = p.duration_ns;
            cp.frequency_hz   = p.frequency_hz;
            cp.amplitude      = p.amplitude;
            cp.phase_rad      = p.phase_rad;
            cp.envelope       = static_cast<qhal_envelope_t>(p.envelope);
            cp.drag_beta      = p.drag_beta;
            cp.envelope_param = p.envelope_param;
        }

        g_last_error.clear();
        return QHAL_OK;
    }
    catch (const std::exception& e) {
        set_error(std::string("qhal_compile: ") + e.what());
        return QHAL_ERR_GENERIC;
    }
}

void qhal_free_program(qhal_program_t* prog) {
    if (!prog || !prog->pulses) return;
    delete[] prog->pulses;
    prog->pulses = nullptr;
    prog->count = 0;
    prog->total_duration_ns = 0.0;
}

// ---------------------------------------------------------------------
// Serialize (stub — implement in Phase 1)
// ---------------------------------------------------------------------
qhal_status_t qhal_save_program(const qhal_program_t* prog, const char* path) {
    (void)prog; (void)path;
    set_error("qhal_save_program: not yet implemented");
    return QHAL_ERR_GENERIC;
}

qhal_status_t qhal_load_program(const char* path, qhal_program_t* out) {
    (void)path; (void)out;
    set_error("qhal_load_program: not yet implemented");
    return QHAL_ERR_GENERIC;
}

// ---------------------------------------------------------------------
// Execute (stub — implement in Phase 2)
// ---------------------------------------------------------------------
qhal_status_t qhal_execute(const qhal_device_t*  device,
                           const qhal_program_t* prog,
                           int32_t*              results,
                           int32_t               shot_count) {
    if (!prog || !results || shot_count <= 0) {
        set_error("qhal_execute: invalid argument");
        return QHAL_ERR_INVALID_ARG;
    }

    try {
        // Reconstruct a Program from the C struct
        Program cpp_prog;
        cpp_prog.total_duration_ns = prog->total_duration_ns;
        cpp_prog.shot_count = prog->shot_count;
        cpp_prog.pulses.reserve(prog->count);
        for (int32_t i = 0; i < prog->count; ++i) {
            Pulse p;
            p.channel        = static_cast<Channel>(prog->pulses[i].channel);
            p.start_ns       = prog->pulses[i].start_ns;
            p.duration_ns    = prog->pulses[i].duration_ns;
            p.frequency_hz   = prog->pulses[i].frequency_hz;
            p.amplitude      = prog->pulses[i].amplitude;
            p.phase_rad      = prog->pulses[i].phase_rad;
            p.envelope       = static_cast<Envelope>(prog->pulses[i].envelope);
            p.drag_beta      = prog->pulses[i].drag_beta;
            p.envelope_param = prog->pulses[i].envelope_param;
            cpp_prog.pulses.push_back(p);
        }

        // Determine qubit count from the device
        int32_t nq = (device && device->qubit_count > 0)
                     ? device->qubit_count : 4;

        // Run the simulator once per shot
        std::vector<int32_t> aggregated;
        int shots_per = 1;
        for (int32_t s = 0; s < shot_count; ++s) {
            Simulator sim(nq, 1000 + s);
            auto out = sim.run_program(cpp_prog);
            if (s == 0) {
                aggregated = out;
            } else {
                // For multiple shots, take the first shot's outcome as
                // representative. Real averaging belongs in the backend.
            }
            if (s == 0 && (int32_t)out.size() > 0) {
                shots_per = 1;
            }
        }

        // Copy results
        for (size_t i = 0; i < aggregated.size(); ++i) {
            results[i] = aggregated[i];
        }

        g_last_error.clear();
        return QHAL_OK;
    }
    catch (const std::exception& e) {
        set_error(std::string("qhal_execute: ") + e.what());
        return QHAL_ERR_BACKEND;
    }
}

// ---------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------
const char* qhal_last_error(void) {
    return g_last_error.empty() ? "" : g_last_error.c_str();
}

void qhal_clear_error(void) {
    g_last_error.clear();
}

} // extern "C"
