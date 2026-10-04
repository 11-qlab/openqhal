#pragma once
#include "types.hpp"
#include <vector>
#include <string>
#include <complex>

namespace qhal {

using Complex = std::complex<double>;

// Rendered waveform: one complex sample buffer per channel.
struct Waveform {
    std::vector<std::vector<Complex>> channels;
    double sample_rate_hz = 1e9;   // 1 GS/s
    double duration_ns    = 0.0;
    int32_t num_channels  = 5;     // matches qhal_channel_t

    size_t sample_count() const {
        return channels.empty() ? 0 : channels[0].size();
    }
    bool empty() const { return sample_count() == 0; }
};

// Render a Program to per-channel I/Q sample buffers.
// sample_rate_hz defaults to 1 GS/s.
Waveform render(const Program& prog,
                double sample_rate_hz = 1e9,
                int32_t num_channels = 5,
                bool baseband = true);

// Serialization
void save_waveform_npy(const Waveform& wf, const std::string& path);
void save_waveform_raw(const Waveform& wf, const std::string& path);
void save_waveform_csv(const Waveform& wf, const std::string& path);
void save_waveform_seqc(const Waveform& wf, const std::string& path,
                        const std::string& seqc_name = "qhal_seq");

// Convenience: sum of |samples| across all channels, for verification
double total_energy(const Waveform& wf);

} // namespace qhal
