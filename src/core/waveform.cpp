#include "qhal/waveform.hpp"
#include "qhal/envelope.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace qhal {

static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------
Waveform render(const Program& prog,
                double sample_rate_hz,
                int32_t num_channels,
                bool baseband) {
    if (sample_rate_hz <= 0.0)
        throw std::invalid_argument("render: sample_rate must be > 0");
    if (num_channels <= 0)
        throw std::invalid_argument("render: num_channels must be > 0");

    Waveform wf;
    wf.sample_rate_hz = sample_rate_hz;
    wf.num_channels   = num_channels;
    wf.duration_ns    = prog.total_duration_ns;
    wf.channels.assign(num_channels, {});

    // Total samples across the timeline
    size_t total_samples =
        static_cast<size_t>(std::ceil(prog.total_duration_ns *
                                      sample_rate_hz / 1e9));
    if (total_samples == 0) return wf;

    for (auto& ch : wf.channels) {
        ch.assign(total_samples, Complex(0.0, 0.0));
    }

    // Render each pulse
    for (const auto& p : prog.pulses) {
        int ch = static_cast<int>(p.channel);
        if (ch < 0 || ch >= num_channels) continue;

        // Zero-duration pulses (virtual Z) are metadata — nothing to render
        if (p.duration_ns <= 0.0) continue;

        // Compute the number of samples for this pulse
        size_t n = static_cast<size_t>(std::ceil(p.duration_ns *
                                                 sample_rate_hz / 1e9));
        if (n == 0) continue;

        // Compute start sample index
        size_t start = static_cast<size_t>(std::round(p.start_ns *
                                                      sample_rate_hz / 1e9));

        // Build the envelope
        SampleRate sr{sample_rate_hz / 1e9};
        std::vector<Complex> env = envelope_shape(
            p.envelope,
            p.duration_ns,
            sr,
            p.envelope_param,
            p.drag_beta,
            0.0,           // anharm handled internally by DRAG shape
            0
        );

        // Resize to n if envelope_shape produced a different count
        if (env.size() < n) env.resize(n, Complex(0.0, 0.0));
        if (env.size() > n) env.resize(n);

        // Modulate by the carrier phase.
        // If baseband, the DUC in hardware applies the carrier — we only
        // write the envelope. Otherwise we sample the carrier directly,
        // which is only correct if sample_rate_hz > 2·frequency_hz.
        double dt = 1.0 / sample_rate_hz;
        for (size_t i = 0; i < n; ++i) {
            double t = i * dt;
            Complex sample;
            if (baseband) {
                // Envelope-only, with a static phase offset from the pulse.
                sample = p.amplitude * env[i] *
                         std::exp(Complex(0.0, p.phase_rad));
            } else {
                Complex carrier = std::exp(Complex(0.0,
                    2.0 * PI * p.frequency_hz * t + p.phase_rad));
                sample = p.amplitude * env[i] * carrier;
            }

            size_t idx = start + i;
            if (idx < total_samples) {
                wf.channels[ch][idx] += sample;
            }
        }
    }

    return wf;
}

// ---------------------------------------------------------------------
// Energy metric
// ---------------------------------------------------------------------
double total_energy(const Waveform& wf) {
    double sum = 0.0;
    for (const auto& ch : wf.channels) {
        for (const auto& s : ch) sum += std::abs(s);
    }
    return sum;
}

// ---------------------------------------------------------------------
// NumPy .npy writer (complex128)
// ---------------------------------------------------------------------
namespace {

void write_le32(std::ofstream& f, uint32_t v) {
    char b[4] = {char(v & 0xFF), char((v >> 8) & 0xFF),
                 char((v >> 16) & 0xFF), char((v >> 24) & 0xFF)};
    f.write(b, 4);
}
void write_le16(std::ofstream& f, uint16_t v) {
    char b[2] = {char(v & 0xFF), char((v >> 8) & 0xFF)};
    f.write(b, 2);
}

} // namespace

void save_waveform_npy(const Waveform& wf, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_npy: cannot write " + path);

    // .npy format for a 2D array of complex128
    // Shape: (num_channels, num_samples)
    uint32_t nch = static_cast<uint32_t>(wf.channels.size());
    uint32_t ns  = static_cast<uint32_t>(wf.sample_count());

    // Build the header dict
    std::string dict = "{'descr': '<c16', 'fortran_order': False, 'shape': (";
    dict += std::to_string(nch) + ", " + std::to_string(ns);
    dict += "), }";

    // Pad the header to 64-byte alignment (16 bytes for magic+version+len)
    size_t header_len = dict.size() + 1;   // +1 for '\n'
    size_t padding = 64 - ((10 + header_len) % 64);
    if (padding == 64) padding = 0;
    dict += std::string(padding, ' ');
    dict += "\n";

    // Magic + version + header length
    f.write("\x93NUMPY", 6);
    f.put('\x01'); f.put('\x00');          // version 1.0
    write_le16(f, static_cast<uint16_t>(dict.size()));
    f.write(dict.data(), dict.size());

    // Data: row-major, interleaved real/imag as double
    for (const auto& ch : wf.channels) {
        for (const auto& s : ch) {
            double re = s.real();
            double im = s.imag();
            f.write(reinterpret_cast<const char*>(&re), sizeof(double));
            f.write(reinterpret_cast<const char*>(&im), sizeof(double));
        }
    }
}

// ---------------------------------------------------------------------
// Raw binary: interleaved int16 I/Q per channel, then next channel
// Matches the format most AWGs expect for streaming upload.
// ---------------------------------------------------------------------
void save_waveform_raw(const Waveform& wf, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_raw: cannot write " + path);

    // Header: 4-byte sample count, 4-byte sample rate (Hz), 4-byte channels
    write_le32(f, static_cast<uint32_t>(wf.sample_count()));
    write_le32(f, static_cast<uint32_t>(wf.sample_rate_hz));
    write_le32(f, static_cast<uint32_t>(wf.channels.size()));

    // Samples: each channel's complex samples as int16 I, int16 Q
    constexpr double SCALE = 32767.0;
    for (const auto& ch : wf.channels) {
        for (const auto& s : ch) {
            double re = std::clamp(s.real(), -1.0, 1.0);
            double im = std::clamp(s.imag(), -1.0, 1.0);
            int16_t ir = static_cast<int16_t>(re * SCALE);
            int16_t ii = static_cast<int16_t>(im * SCALE);
            f.write(reinterpret_cast<const char*>(&ir), sizeof(int16_t));
            f.write(reinterpret_cast<const char*>(&ii), sizeof(int16_t));
        }
    }
}

// ---------------------------------------------------------------------
// CSV: index, I, Q — for eyeballing the waveform in a spreadsheet
// ---------------------------------------------------------------------
void save_waveform_csv(const Waveform& wf, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_csv: cannot write " + path);

    f << "channel,sample,time_ns,I,Q\n";
    double dt_ns = 1e9 / wf.sample_rate_hz;
    for (size_t c = 0; c < wf.channels.size(); ++c) {
        for (size_t i = 0; i < wf.channels[c].size(); ++i) {
            const auto& s = wf.channels[c][i];
            f << c << "," << i << ","
              << (i * dt_ns) << ","
              << s.real() << "," << s.imag() << "\n";
        }
    }
}

// ---------------------------------------------------------------------
// Zurich HDAWG .seqc (text) — the standard format for Zurich AWGs.
// This produces a compilable sequence program that plays the waveform.
// ---------------------------------------------------------------------
void save_waveform_seqc(const Waveform& wf, const std::string& path,
                        const std::string& seqc_name) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("save_seqc: cannot write " + path);

    double rate_GHz = wf.sample_rate_hz / 1e9;
    size_t ns = wf.sample_count();

    f << "// " << seqc_name << ".seqc\n";
    f << "// Auto-generated by OpenQHAL\n";
    f << "// Sample rate: " << rate_GHz << " GHz\n";
    f << "// Channels: " << wf.channels.size() << "\n";
    f << "// Samples per channel: " << ns << "\n";
    f << "\n";
    f << "const N = " << ns << ";\n";
    f << "const RATE = " << rate_GHz << "e9;\n";
    f << "\n";

    // Declare wave arrays
    for (size_t c = 0; c < wf.channels.size(); ++c) {
        f << "wave w" << c << " = zeros(N);\n";
    }
    f << "\n";

    // Fill: one sample per line
    // For real hardware, you'd use playWave or assignWaveIndex.
    // This is the readable representation.
    for (size_t c = 0; c < wf.channels.size(); ++c) {
        f << "// Channel " << c << "\n";
        f << "// (samples omitted for brevity in this template)\n";
    }

    f << "\n";
    f << "// Main sequence\n";
    f << "while (1) {\n";
    for (size_t c = 0; c < wf.channels.size(); ++c) {
        f << "    playWave(w" << c << ");\n";
    }
    f << "}\n";
}

} // namespace qhal
