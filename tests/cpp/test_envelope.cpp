#include "qhal/envelope.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace qhal;

int main() {
    SampleRate rate{1.0};

    {
        auto s = envelope_shape(Envelope::Square, 100.0, rate);
        assert(s.size() == 100);
        printf("square      OK  (n=%zu, area=%.3f)\n", s.size(), pulse_area(s, rate));
    }
    {
        auto s = envelope_shape(Envelope::Gaussian, 100.0, rate, 0.2);
        int argmax = 0;
        for (int i = 1; i < (int)s.size(); ++i)
            if (std::abs(s[i]) > std::abs(s[argmax])) argmax = i;
        assert(std::abs(argmax - 50) <= 1);
        printf("gaussian    OK  (peak at %d, area=%.3f)\n", argmax, pulse_area(s, rate));
    }
    {
        auto s = envelope_shape(Envelope::DRAG, 100.0, rate, 0.2, 0.5, 1e9);
        double im_max = 0.0;
        for (auto& x : s) im_max = std::max(im_max, std::abs(x.imag()));
        assert(im_max > 1e-6);
        printf("drag        OK  (max imag=%.4f, area=%.3f)\n", im_max, pulse_area(s, rate));
    }
    {
        auto s = envelope_shape(Envelope::Sech, 100.0, rate, 0.2);
        printf("sech        OK  (n=%zu, area=%.3f)\n", s.size(), pulse_area(s, rate));
    }
    {
        auto s = envelope_shape(Envelope::Hermite, 100.0, rate, 0.2, 0.0, 0.0, 2);
        bool has_neg = false;
        for (auto& x : s) if (x.real() < -1e-6) has_neg = true;
        assert(has_neg);
        printf("hermite(2)  OK  (area=%.3f)\n", pulse_area(s, rate));
    }
    {
        auto s = envelope_shape(Envelope::Gaussian, 100.0, rate, 0.2);
        normalize_for_angle(s, rate, M_PI);
        assert(std::abs(pulse_area(s, rate) - M_PI) < 1e-9);
        printf("normalize   OK  (area=%.3f)\n", pulse_area(s, rate));
    }
    printf("\nall envelope tests passed\n");
    return 0;
}
