#pragma once
#include "qhal/backend.hpp"
#include <string>
#include <vector>

namespace qhal {

class ZurichBackend : public Backend {
public:
    struct Config {
        std::string host          = "localhost";
        int32_t     api_port      = 8004;
        std::string device_id     = "dev8000";
        int32_t     awg_index     = 0;
        int32_t     channel_pair  = 0;      // 0 or 1 on HDAWG (4 or 8 channel)
        double      sample_rate_hz = 2.4e9; // HDAWG max
        double      output_range_v = 1.0;
        int32_t     readout_ch    = 0;      // UHFQA or digitizer channel
        bool        use_toolkit   = true;   // use zhinst.toolkit if available
    };

    ZurichBackend();
    explicit ZurichBackend(Config cfg);

    std::string name() const override { return "ZurichHDAWG"; }

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
