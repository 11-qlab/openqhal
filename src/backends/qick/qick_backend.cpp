#include "qhal/backends/qick.hpp"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <chrono>
#include <thread>
#include <stdexcept>

namespace py = pybind11;

namespace qhal {

QICKBackend::QICKBackend() : cfg_() {}
QICKBackend::QICKBackend(Config cfg) : cfg_(std::move(cfg)) {}

bool QICKBackend::connect() {
    if (connected_) return true;
    try {
        py::module_ qick = py::module_::import("qick");
        py::object soc = qick.attr("QickSoc")(cfg_.bitstream_path);
        py::globals()["_qhal_soc"] = soc;
        connected_ = true;
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

void QICKBackend::disconnect() {
    if (!connected_) return;
    try {
        py::globals()["_qhal_soc"] = py::none();
    } catch (...) {}
    connected_ = false;
    armed_ = false;
    uploaded_ = Waveform{};
}

bool QICKBackend::upload(const Waveform& wf) {
    if (!connected_) return false;
    auto t0 = std::chrono::steady_clock::now();
    try {
        py::object soc = py::globals()["_qhal_soc"];
        py::object numpy = py::module_::import("numpy");

        auto& ch = wf.channels[0];
        py::list i_samples, q_samples;
        for (const auto& s : ch) {
            i_samples.append(s.real());
            q_samples.append(s.imag());
        }

        py::exec(R"PY(
import numpy as _np
import qick as _qick
import qick.asm_v2 as _asm_v2

class _QhalProgram(_asm_v2.AveragerProgramV2):
    def _initialize(self, cfg):
        self.declare_gen(ch=cfg['gen_ch'], nqz=1)
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['ro_len'])
        self.add_envelope(
            ch=cfg['gen_ch'],
            name="qhal_env",
            idata=cfg['idata'],
            qdata=cfg['qdata'])
        self.add_pulse(
            ch=cfg['gen_ch'],
            name="qhal_pulse",
            style="arb",
            envelope="qhal_env",
            freq=cfg['freq'],
            phase=0,
            gain=cfg['gain'])
        self.add_readoutconfig(
            ch=cfg['ro_ch'],
            name="ro",
            freq=cfg['freq'],
            gen_ch=cfg['gen_ch'])
        self.send_readoutconfig(ch=cfg['ro_ch'], name="ro", t=0)

    def _body(self, cfg):
        self.pulse(ch=cfg['gen_ch'], name="qhal_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
)PY", py::globals());

        py::dict cfg;
        cfg["gen_ch"]    = cfg_.gen_ch;
        cfg["ro_ch"]     = cfg_.ro_ch;
        cfg["freq"]      = cfg_.freq_mhz;
        cfg["gain"]      = cfg_.gain;
        cfg["trig_time"] = cfg_.trigger_time_us;
        cfg["ro_len"]    = cfg_.readout_length_us;
        cfg["idata"]     = i_samples;
        cfg["qdata"]     = q_samples;

        py::object prog = py::globals()["_QhalProgram"](
            soc.attr("soccfg"),
            py::arg("reps") = 1,
            py::arg("final_delay") = 0.5,
            py::arg("cfg") = cfg);

        py::globals()["_qhal_prog"] = prog;
        uploaded_ = wf;

        auto t1 = std::chrono::steady_clock::now();
        stats_.upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.samples_uploaded = wf.sample_count();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool QICKBackend::trigger() {
    if (!connected_ || uploaded_.empty()) return false;
    armed_ = true;
    stats_.trigger_ms = 0.0;
    return true;
}

std::vector<int32_t> QICKBackend::acquire(int32_t shot_count) {
    std::vector<int32_t> results;
    if (!armed_) return results;

    auto t0 = std::chrono::steady_clock::now();
    try {
        py::object prog = py::globals()["_qhal_prog"];
        py::object soc  = py::globals()["_qhal_soc"];

        py::object iq = prog.attr("acquire")(
            soc, py::arg("rounds") = shot_count);

        py::object numpy = py::module_::import("numpy");
        py::object i_mean = numpy.attr("mean")(
            py::sequence(iq)[0], py::arg("axis") = 0);
        double val = i_mean.cast<double>();
        results.push_back(val > 0.0 ? 1 : 0);

        auto t1 = std::chrono::steady_clock::now();
        stats_.acquisition_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.total_ms = stats_.upload_ms + stats_.trigger_ms + stats_.acquisition_ms;
        stats_.shots = shot_count;
        stats_.pulses_uploaded = 1;
    } catch (const std::exception&) {}

    armed_ = false;
    return results;
}

} // namespace qhal
