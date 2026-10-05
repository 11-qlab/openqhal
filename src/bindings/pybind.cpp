
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include "qhal/types.hpp"
#include "qhal/compiler.hpp"
#include "qhal/scheduler.hpp"
#include "qhal/simulator.hpp"
#include "qhal/calibration.hpp"
#include "qhal/waveform.hpp"
#include "qhal/backend.hpp"
#include "qhal/backends/mock_awg.hpp"
#include "qhal/backends/qick.hpp"
#include "qhal/backends/zurich.hpp"

namespace py = pybind11;
using namespace qhal;


// ---------------------------------------------------------------------
// Trampoline class: lets Python subclasses override Backend methods
// ---------------------------------------------------------------------
class PyBackend : public Backend {
public:
    using Backend::Backend;

    std::string name() const override {
        PYBIND11_OVERRIDE_PURE(std::string, Backend, name);
    }
    bool connect() override {
        PYBIND11_OVERRIDE_PURE(bool, Backend, connect);
    }
    void disconnect() override {
        PYBIND11_OVERRIDE_PURE(void, Backend, disconnect);
    }
    bool is_connected() const override {
        PYBIND11_OVERRIDE_PURE(bool, Backend, is_connected);
    }
    bool upload(const Waveform& wf) override {
        PYBIND11_OVERRIDE_PURE(bool, Backend, upload, wf);
    }
    bool trigger() override {
        PYBIND11_OVERRIDE_PURE(bool, Backend, trigger);
    }
    std::vector<int32_t> acquire(int32_t shots) override {
        PYBIND11_OVERRIDE_PURE(std::vector<int32_t>, Backend, acquire, shots);
    }
    ExecutionStats last_stats() const override {
        PYBIND11_OVERRIDE_PURE(ExecutionStats, Backend, last_stats);
    }
};

PYBIND11_MODULE(qhal_cpp, m) {
    m.doc() = "OpenQHAL — quantum pulse compiler core (C++ backend)";

    // -----------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------
    py::enum_<Channel>(m, "Channel")
        .value("I",       Channel::I)
        .value("Q",       Channel::Q)
        .value("Laser",   Channel::Laser)
        .value("Readout", Channel::Readout)
        .value("Flux",    Channel::Flux);

    py::enum_<Envelope>(m, "Envelope")
        .value("Square",   Envelope::Square)
        .value("Gaussian", Envelope::Gaussian)
        .value("DRAG",     Envelope::DRAG)
        .value("Sech",     Envelope::Sech)
        .value("Hermite",  Envelope::Hermite)
        .value("Custom",   Envelope::Custom);

    // -----------------------------------------------------------------
    // Data types
    // -----------------------------------------------------------------
    py::class_<Gate>(m, "Gate")
        .def(py::init<>())
        .def(py::init([](const std::string& name,
                         std::vector<int32_t> qubits,
                         std::vector<double> params) {
            Gate g;
            g.name   = name;
            g.qubits = std::move(qubits);
            g.params = std::move(params);
            return g;
        }), py::arg("name"), py::arg("qubits"),
            py::arg("params") = std::vector<double>{})
        .def_readwrite("name",   &Gate::name)
        .def_readwrite("qubits", &Gate::qubits)
        .def_readwrite("params", &Gate::params)
        .def("__repr__", [](const Gate& g) {
            std::string s = "Gate(" + g.name + ", [";
            for (size_t i = 0; i < g.qubits.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(g.qubits[i]);
            }
            s += "])";
            return s;
        });

    py::class_<Pulse>(m, "Pulse")
        .def(py::init<>())
        .def_readwrite("channel",        &Pulse::channel)
        .def_readwrite("start_ns",       &Pulse::start_ns)
        .def_readwrite("duration_ns",    &Pulse::duration_ns)
        .def_readwrite("frequency_hz",   &Pulse::frequency_hz)
        .def_readwrite("amplitude",      &Pulse::amplitude)
        .def_readwrite("phase_rad",      &Pulse::phase_rad)
        .def_readwrite("envelope",       &Pulse::envelope)
        .def_readwrite("drag_beta",      &Pulse::drag_beta)
        .def_readwrite("envelope_param", &Pulse::envelope_param)
        .def_readwrite("qubit",          &Pulse::qubit)
        .def_readwrite("gate_id",        &Pulse::gate_id)
        .def("__repr__", [](const Pulse& p) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "Pulse(ch=%d, t=%.1fns, dur=%.1fns, f=%.3fGHz, q=%d)",
                (int)p.channel, p.start_ns, p.duration_ns,
                p.frequency_hz/1e9, p.qubit);
            return std::string(buf);
        });

    py::class_<Program>(m, "Program")
        .def(py::init<>())
        .def_readwrite("pulses",            &Program::pulses)
        .def_readwrite("total_duration_ns", &Program::total_duration_ns)
        .def_readwrite("shot_count",        &Program::shot_count)
        .def("__len__",  [](const Program& p) { return p.pulses.size(); })
        .def("__repr__", [](const Program& p) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "Program(%zu pulses, %.1f ns)",
                p.pulses.size(), p.total_duration_ns);
            return std::string(buf);
        });

    py::class_<QubitCalibration>(m, "QubitCalibration")
        .def(py::init<>())
        .def_readwrite("qubit_id",            &QubitCalibration::qubit_id)
        .def_readwrite("rabi_hz",             &QubitCalibration::rabi_hz)
        .def_readwrite("anharm_hz",           &QubitCalibration::anharm_hz)
        .def_readwrite("drive_freq_hz",       &QubitCalibration::drive_freq_hz)
        .def_readwrite("t1_ns",               &QubitCalibration::t1_ns)
        .def_readwrite("t2_ns",               &QubitCalibration::t2_ns)
        .def_readwrite("drag_beta",           &QubitCalibration::drag_beta)
        .def_readwrite("sigma_frac",          &QubitCalibration::sigma_frac)
        .def_readwrite("readout_duration_ns", &QubitCalibration::readout_duration_ns)
        .def_readwrite("virtual_z",           &QubitCalibration::virtual_z);

    py::class_<Calibration>(m, "Calibration")
        .def(py::init<>())
        .def_readwrite("device_name",       &Calibration::device_name)
        .def_readwrite("qubits",            &Calibration::qubits)
        .def_readwrite("cnot_duration_ns",  &Calibration::cnot_duration_ns)
        .def_readwrite("cz_duration_ns",    &Calibration::cz_duration_ns)
        .def_readwrite("measure_duration_ns", &Calibration::measure_duration_ns)
        .def("get", &Calibration::get, py::return_value_policy::reference_internal)
        .def_static("from_json", [](const std::string& path) {
            return load_calibration_json(path);
        })
        .def_static("from_string", [](const std::string& json) {
            return load_calibration_string(json);
        })
        .def("save", &save_calibration_json);

    // -----------------------------------------------------------------
    // Compiler
    // -----------------------------------------------------------------
    py::class_<Compiler>(m, "Compiler")
        .def(py::init<Calibration>())
        .def("compile", &Compiler::compile)
        .def("calibration", &Compiler::calibration,
             py::return_value_policy::reference_internal);

    // -----------------------------------------------------------------
    // Scheduler
    // -----------------------------------------------------------------
    py::class_<ScheduleResult>(m, "ScheduleResult")
        .def_readwrite("pulses",            &ScheduleResult::pulses)
        .def_readwrite("total_duration_ns", &ScheduleResult::total_duration_ns)
        .def_readwrite("makespan_improvement_ns",
                       &ScheduleResult::makespan_improvement_ns);

    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<>())
        .def("schedule_program", &Scheduler::schedule_program);

    // -----------------------------------------------------------------
    // Simulator
    // -----------------------------------------------------------------
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int32_t, uint64_t>(),
             py::arg("num_qubits"), py::arg("seed") = 42)
        .def("apply_gate", &Simulator::apply_gate)
        .def("measure",    &Simulator::measure)
        .def("run",        &Simulator::run)
        .def("run_program",&Simulator::run_program)
        .def("prob_zero",  &Simulator::prob_zero)
        .def("state", [](const Simulator& s) {
            std::vector<std::complex<double>> out(s.state().begin(),
                                                  s.state().end());
            return out;
        });

    // -----------------------------------------------------------------
    // Waveform
    // -----------------------------------------------------------------
    py::class_<Waveform>(m, "Waveform")
        .def_readwrite("sample_rate_hz", &Waveform::sample_rate_hz)
        .def_readwrite("duration_ns",    &Waveform::duration_ns)
        .def_readwrite("num_channels",   &Waveform::num_channels)
        .def("sample_count", &Waveform::sample_count)
        .def("empty",        &Waveform::empty)
        .def("channel", [](const Waveform& wf, int ch) {
            if (ch < 0 || ch >= (int)wf.channels.size())
                throw std::out_of_range("channel index out of range");
            std::vector<std::complex<double>> out(
                wf.channels[ch].begin(), wf.channels[ch].end());
            return out;
        }, py::arg("channel"))
        .def("__repr__", [](const Waveform& wf) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "Waveform(%zu channels, %zu samples, %.1f ns)",
                wf.channels.size(), wf.sample_count(), wf.duration_ns);
            return std::string(buf);
        });

    m.def("render", &render,
          py::arg("prog"),
          py::arg("sample_rate_hz") = 1e9,
          py::arg("num_channels") = 8,
          py::arg("baseband") = true);

    m.def("save_npy",  &save_waveform_npy);
    m.def("save_raw",  &save_waveform_raw);
    m.def("save_csv",  &save_waveform_csv);
    m.def("save_seqc", &save_waveform_seqc,
          py::arg("wf"), py::arg("path"),
          py::arg("seqc_name") = "qhal_seq");

    m.def("total_energy", &total_energy);

    // -----------------------------------------------------------------
    // Backend + MockAWG
    // -----------------------------------------------------------------
    py::class_<ExecutionStats>(m, "ExecutionStats")
        .def(py::init<>())
        .def_readwrite("upload_ms",        &ExecutionStats::upload_ms)
        .def_readwrite("trigger_ms",       &ExecutionStats::trigger_ms)
        .def_readwrite("acquisition_ms",   &ExecutionStats::acquisition_ms)
        .def_readwrite("total_ms",         &ExecutionStats::total_ms)
        .def_readwrite("shots",            &ExecutionStats::shots)
        .def_readwrite("pulses_uploaded",  &ExecutionStats::pulses_uploaded)
        .def_readwrite("samples_uploaded", &ExecutionStats::samples_uploaded)
        .def("__repr__", [](const ExecutionStats& s) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "ExecutionStats(upload=%.2fms, trigger=%.2fms, "
                "acquire=%.2fms, total=%.2fms, shots=%d)",
                s.upload_ms, s.trigger_ms, s.acquisition_ms,
                s.total_ms, s.shots);
            return std::string(buf);
        });

    py::class_<Backend, PyBackend>(m, "Backend")
        .def(py::init<>())
        .def("name",         &Backend::name)
        .def("connect",      &Backend::connect)
        .def("disconnect",   &Backend::disconnect)
        .def("is_connected", &Backend::is_connected)
        .def("upload",       &Backend::upload)
        .def("trigger",      &Backend::trigger)
        .def("acquire",      &Backend::acquire, py::arg("shot_count") = 1)
        .def("last_stats",   &Backend::last_stats)
        .def("execute",      &Backend::execute,
             py::arg("wf"), py::arg("shot_count") = 1);

    py::class_<MockAWG, Backend>(m, "MockAWG")
        .def(py::init<>())
        .def(py::init([](int32_t nq, uint64_t seed, bool lat) {
            MockAWG::Config c;
            c.num_qubits = nq;
            c.seed = seed;
            c.simulate_latency = lat;
            return MockAWG(c);
        }), py::arg("num_qubits") = 4,
            py::arg("seed") = 42,
            py::arg("simulate_latency") = true)
        .def("set_circuit", &MockAWG::set_circuit)
        .def("name",         &MockAWG::name)
        .def("connect",      &MockAWG::connect)
        .def("disconnect",   &MockAWG::disconnect)
        .def("is_connected", &MockAWG::is_connected)
        .def("upload",       &MockAWG::upload)
        .def("trigger",      &MockAWG::trigger)
        .def("acquire",      &MockAWG::acquire, py::arg("shot_count") = 1)
        .def("last_stats",   &MockAWG::last_stats);

    py::class_<QICKBackend, Backend>(m, "QICKBackend")
        .def(py::init<>())
        .def(py::init([](const std::string& bitstream, int gen_ch, int ro_ch) {
            QICKBackend::Config c;
            c.bitstream_path = bitstream;
            c.gen_ch = gen_ch;
            c.ro_ch  = ro_ch;
            return QICKBackend(c);
        }), py::arg("bitstream_path"), py::arg("gen_ch") = 0,
            py::arg("ro_ch") = 0)
        .def("name",         &QICKBackend::name)
        .def("connect",      &QICKBackend::connect)
        .def("disconnect",   &QICKBackend::disconnect)
        .def("is_connected", &QICKBackend::is_connected)
        .def("upload",       &QICKBackend::upload)
        .def("trigger",      &QICKBackend::trigger)
        .def("acquire",      &QICKBackend::acquire,
             py::arg("shot_count") = 1)
        .def("last_stats",   &QICKBackend::last_stats);

    py::class_<ZurichBackend, Backend>(m, "ZurichBackend")
        .def(py::init<>())
        .def(py::init([](const std::string& host,
                         const std::string& device_id,
                         int32_t awg_index) {
            ZurichBackend::Config c;
            c.host      = host;
            c.device_id = device_id;
            c.awg_index = awg_index;
            return ZurichBackend(c);
        }), py::arg("host")      = "localhost",
            py::arg("device_id") = "dev8000",
            py::arg("awg_index") = 0)
        .def("name",         &ZurichBackend::name)
        .def("connect",      &ZurichBackend::connect)
        .def("disconnect",   &ZurichBackend::disconnect)
        .def("is_connected", &ZurichBackend::is_connected)
        .def("upload",       &ZurichBackend::upload)
        .def("trigger",      &ZurichBackend::trigger)
        .def("acquire",      &ZurichBackend::acquire,
             py::arg("shot_count") = 1)
        .def("last_stats",   &ZurichBackend::last_stats);

}