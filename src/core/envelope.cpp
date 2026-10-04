#include "qhal/envelope.hpp"
#include <cmath>
#include <stdexcept>

namespace qhal {

static constexpr double PI = 3.14159265358979323846;

static std::vector<Complex> square(double duration_ns, const SampleRate& rate) {
    int n = rate.to_count(duration_ns);
    return std::vector<Complex>(n, Complex(1.0, 0.0));
}

static std::vector<Complex> gaussian(double duration_ns, const SampleRate& rate,
                                     double sigma_frac) {
    int n = rate.to_count(duration_ns);
    std::vector<Complex> out(n);
    double sigma = duration_ns * sigma_frac;
    double t0    = duration_ns * 0.5;
    double dt    = 1.0 / rate.samples_per_ns;
    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        double x = (t - t0) / sigma;
        out[i]   = Complex(std::exp(-0.5 * x * x), 0.0);
    }
    return out;
}

static std::vector<Complex> drag(double duration_ns, const SampleRate& rate,
                                 double sigma_frac, double beta,
                                 double anharm_hz) {
    int n = rate.to_count(duration_ns);
    std::vector<Complex> out(n);
    double sigma = duration_ns * sigma_frac;
    double t0    = duration_ns * 0.5;
    double dt    = 1.0 / rate.samples_per_ns;
// Anharmonicity in angular frequency, converted to rad/ns
// so it matches the 1/ns units of dg.
double delta = (anharm_hz != 0.0)
                   ? 2.0 * PI * anharm_hz / 1e9
                   : 2.0 * PI;
    for (int i = 0; i < n; ++i) {
        double t  = i * dt;
        double x  = (t - t0) / sigma;
        double g  = std::exp(-0.5 * x * x);
        double dg = -(x / sigma) * g;
        out[i]    = Complex(g, beta * dg / delta);
    }
    return out;
}

static std::vector<Complex> sech(double duration_ns, const SampleRate& rate,
                                 double sigma_frac) {
    int n = rate.to_count(duration_ns);
    std::vector<Complex> out(n);
    double sigma = duration_ns * sigma_frac;
    double t0    = duration_ns * 0.5;
    double dt    = 1.0 / rate.samples_per_ns;
    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        double x = (t - t0) / sigma;
        out[i]   = Complex(1.0 / std::cosh(x), 0.0);
    }
    return out;
}

static std::vector<Complex> hermite_gaussian(double duration_ns,
                                             const SampleRate& rate,
                                             double sigma_frac,
                                             int32_t order) {
    int n = rate.to_count(duration_ns);
    std::vector<Complex> out(n);
    double sigma = duration_ns * sigma_frac;
    double t0    = duration_ns * 0.5;
    double dt    = 1.0 / rate.samples_per_ns;
    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        double x = (t - t0) / sigma;
        double g = std::exp(-0.5 * x * x);
        double h = 1.0;
        if      (order == 1) h = 2.0 * x;
        else if (order == 2) h = 4.0 * x * x - 2.0;
        else if (order == 3) h = 8.0 * x * x * x - 12.0 * x;
        else if (order == 4) h = 16.0 * x * x * x * x - 48.0 * x * x + 12.0;
        out[i] = Complex(g * h, 0.0);
    }
    return out;
}

std::vector<Complex> envelope_shape(Envelope envelope,
                                    double duration_ns,
                                    const SampleRate& rate,
                                    double sigma_frac,
                                    double beta,
                                    double anharm_hz,
                                    int32_t hermite_order) {
    if (duration_ns <= 0.0)
        throw std::invalid_argument("envelope: duration must be > 0");
    switch (envelope) {
        case Envelope::Square:   return square(duration_ns, rate);
        case Envelope::Gaussian: return gaussian(duration_ns, rate, sigma_frac);
        case Envelope::DRAG:     return drag(duration_ns, rate, sigma_frac, beta, anharm_hz);
        case Envelope::Sech:     return sech(duration_ns, rate, sigma_frac);
        case Envelope::Hermite:  return hermite_gaussian(duration_ns, rate, sigma_frac, hermite_order);
        case Envelope::Custom:   throw std::invalid_argument("envelope: Custom handled externally");
    }
    throw std::invalid_argument("envelope: unknown shape");
}

double pulse_area(const std::vector<Complex>& samples, const SampleRate& rate) {
    if (samples.empty()) return 0.0;
    double dt = 1.0 / rate.samples_per_ns;
    double sum = 0.0;
    for (const auto& s : samples) sum += std::abs(s);
    return sum * dt;
}

void normalize_for_angle(std::vector<Complex>& samples,
                         const SampleRate& rate,
                         double target_angle_rad) {
    double a = pulse_area(samples, rate);
    if (a <= 0.0) throw std::runtime_error("normalize: zero-area pulse");
    double scale = target_angle_rad / a;
    for (auto& s : samples) s *= scale;
}

} // namespace qhal
