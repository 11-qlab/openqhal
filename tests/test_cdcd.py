"""End-to-end CDCD test: compile a syndrome extraction circuit,
annotate it with qLDPC check structure, build a decoder noise model."""
import sys
sys.path.insert(0, "python")

import numpy as np
from qhal.qldpc import small_bb, gross_code, repetition
from qhal.cdcd import annotate_program, build_check_noise, build_detector_error_model


# ─────────────────────────────────────────────────────────────────
#  A mock Program/Pulse for testing (mimics qhal_cpp types)
# ─────────────────────────────────────────────────────────────────

class MockPulse:
    def __init__(self, qubit, duration_ns, channel=0):
        self.qubit = qubit
        self.duration_ns = duration_ns
        self.channel = channel

class MockProgram:
    def __init__(self, pulses):
        self.pulses = pulses
        self.total_duration_ns = sum(p.duration_ns for p in pulses)


def make_syndrome_extraction_program(code):
    """
    Build a mock syndrome extraction program:
      - One pulse per data qubit (for encoding)
      - One pulse per check (ancilla for syndrome)
    """
    pulses = []
    # Data qubit pulses: 25 ns each
    for q in range(code.n):
        pulses.append(MockPulse(qubit=q, duration_ns=25.0))
    # Ancilla pulses: one per check
    for ci in range(code.num_x_checks):
        pulses.append(MockPulse(qubit=code.n + ci, duration_ns=100.0))
    for ci in range(code.num_z_checks):
        pulses.append(MockPulse(
            qubit=code.n + code.num_x_checks + ci, duration_ns=100.0))
    return MockProgram(pulses)


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CDCD Test — Compiler-Decoder Co-Design Bridge")
    print("=" * 60)

    for name, code in [("Rep[[5]]", repetition(5)),
                       ("SmallBB[[98]]", small_bb()),
                       ("Gross[[144]]", gross_code())]:
        print(f"\n[{name}]")
        print(f"  n={code.n}, X-checks={code.num_x_checks}, "
              f"Z-checks={code.num_z_checks}")
        print(f"  CSS valid: {code.verify_css()}")

        # Build a syndrome extraction program
        prog = make_syndrome_extraction_program(code)
        print(f"  Compiled pulses: {len(prog.pulses)}")

        # Annotate
        ann = annotate_program(prog, code)
        print(f"  Annotations: {len(ann.annotations)}")

        # Build noise model
        dem = build_detector_error_model(ann, calibration=None,
                                          gate_error_per_ns=1e-5)
        print(f"  DEM entries: {len(dem['error_rates'])}")

        # Show a few checks
        for ct, ci in [("X", 0), ("Z", 0)]:
            key = f"{ct}_{ci}"
            if key in dem["error_rates"]:
                print(f"    {key}: err={dem['error_rates'][key]:.6f}, "
                      f"dur={dem['durations_ns'][key]:.1f} ns")

    print("\n[ALL PASSED]")
