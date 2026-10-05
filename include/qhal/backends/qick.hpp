#pragma once
#include "qhal/backend.hpp"
#include <string>
#include <vector>

namespace qhal {

class QICKBackend : public Backend {
public:
    struct Config {
        std::string bitstream_path;
        int32_t     gen_ch    = 0;
        int32_t     ro_ch     = 0;
        double      freq_mhz  = 500.0;
        double      gain      = 1.0;
        double      trigger_time_us = 0.2;
        int32_t     readout_length_us = 1;
    };

    QICKBackend();
    explicit QICKBackend(Config cfg);

    std::string name() const override { return "QICK"; }

    bool connect() override;
    void disconnect() override;
    bool is_connected() const override { return connected_; }

    bool upload(const Waveform& wf) override;
    bool trigger() override;
    std::vector<int32_t> acquire(int32_t shot_count) override;

    ExecutionStats last_stats() const override { return stats_; }

private:
    Config          cfg_;
    bool            connected_ = false;
    bool            armed_     = false;
    Waveform        uploaded_;
    ExecutionStats  stats_;
};

} // namespace qhal
