#include "qhal/backends/zurich.hpp"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <chrono>
#include <stdexcept>

namespace py = pybind11;

namespace qhal {

ZurichBackend::ZurichBackend() : cfg_() {}
ZurichBackend::ZurichBackend(Config cfg) : cfg_(std::move(cfg)) {}

bool ZurichBackend::connect() {
    if (connected_) return true;
    try {
        py::dict cfg;
        cfg["host"]      = cfg_.host;
        cfg["api_port"]  = cfg_.api_port;
        cfg["device_id"] = cfg_.device_id;
        cfg["awg_index"] = cfg_.awg_index;
        cfg["range_v"]   = cfg_.output_range_v;

        py::exec(R"PY(
_session = zhinst.Session(host=cfg['host'], port=cfg['api_port'])
_device  = tk.Session(_session).connect_device(cfg['device_id'])
_device.factory.reset()
_awg = _device.awg(cfg['awg_index'])
_awg.output1(cfg['range_v'])
_awg.output2(cfg['range_v'])
)PY", py::globals(), cfg);

        connected_ = true;
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

void ZurichBackend::disconnect() {
    if (!connected_) return;
    try {
        py::exec(R"PY(
try:
    _awg.output1(0)
    _awg.output2(0)
    _awg.enable(0)
except Exception:
    pass
_device = None
_session = None
)PY", py::globals());
    } catch (...) {}
    connected_ = false;
    armed_ = false;
    uploaded_ = Waveform{};
}

bool ZurichBackend::upload(const Waveform& wf) {
    if (!connected_) return false;
    auto t0 = std::chrono::steady_clock::now();

    try {
        // Extract I and Q sample arrays for the chosen channel pair
        size_t total_samples = wf.sample_count();
        if (total_samples == 0) return false;

        int ch_i = cfg_.channel_pair * 2;
        int ch_q = ch_i + 1;

        py::list i_list, q_list;
        for (size_t i = 0; i < total_samples; ++i) {
            double re = (ch_i < (int)wf.channels.size())
                            ? wf.channels[ch_i][i].real() : 0.0;
            double im = (ch_q < (int)wf.channels.size())
                            ? wf.channels[ch_q][i].imag() : 0.0;
            i_list.append(re);
            q_list.append(im);
        }

        py::dict cfg;
        cfg["awg_index"]  = cfg_.awg_index;
        cfg["i_samples"]  = i_list;
        cfg["q_samples"]  = q_list;
        cfg["rate"]       = cfg_.sample_rate_hz;

        py::exec(R"PY(
import numpy as np
_awg = _device.awg(cfg['awg_index'])

# HDAWG sequences are built as .seqc programs. zhinst.toolkit
# exposes a Python API that compiles and uploads them.
# Minimum viable: single-waveform arb playback.
i_arr = np.array(cfg['i_samples'], dtype=float)
q_arr = np.array(cfg['q_samples'], dtype=float)

# Normalize to [-1, 1] for DAC range
peak = max(np.max(np.abs(i_arr)), np.max(np.abs(q_arr)), 1e-12)
i_arr /= peak
q_arr /= peak

# Upload to the AWG's waveform memory
_awg.waveforms.clear()
_awg.waveforms["qhal_i"] = i_arr
_awg.waveforms["qhal_q"] = q_arr

# Program: play once, wait for trigger
_seq = _awg.generate_sequencer_program(0)
_seq.playWave("qhal_i", "qhal_q")
_awg.upload_sequencer_program(_seq)
_awg.enable(1)
_awg.wait_done()
)PY", py::globals(), cfg);

        uploaded_ = wf;
        auto t1 = std::chrono::steady_clock::now();
        stats_.upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats_.samples_uploaded = total_samples;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool ZurichBackend::trigger() {
    if (!connected_ || uploaded_.empty()) return false;

    auto t0 = std::chrono::steady_clock::now();
    try {
        py::exec(R"PY(
_awg.enable(1)
# The HDAWG accepts external triggers on the trigger in port;
# for software trigger we use the "run" command.
try:
    _awg.run()
except Exception:
    pass
)PY", py::globals());
        armed_ = true;
    } catch (const std::exception&) {
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    stats_.trigger_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

std::vector<int32_t> ZurichBackend::acquire(int32_t shot_count) {
    std::vector<int32_t> results;
    if (!armed_) return results;

    auto t0 = std::chrono::steady_clock::now();
    try {
        py::dict cfg;
        cfg["shots"] = shot_count;
        cfg["awg_index"] = cfg_.awg_index;
        cfg["readout_ch"] = cfg_.readout_ch;

        py::exec(R"PY(
import numpy as np
# For an HDAWG + UHFQA setup, readout is acquired on the UHFQA.
# Minimal: fetch the accumulated I/Q from the UHFQA result logger.
try:
    _uhfqa = _device.uhfqa(0)
    result = _uhfqa.result_logging()
    i_data = result.get('integration', {}).get('sample', {}).get('value', [])
    i_mean = np.mean(i_data) if len(i_data) > 0 else 0.0
    bit = 1 if i_mean > 0 else 0
except Exception:
    bit = 0
results = [bit]
)PY", py::globals(), cfg);

        py::object results_obj = py::globals()["results"];
        for (auto item : results_obj) {
            results.push_back(item.cast<int32_t>());
        }

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
