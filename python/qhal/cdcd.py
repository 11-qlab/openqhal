"""
Compiler-Decoder Co-Design (CDCD) bridge.

Annotates compiled pulse programs with qLDPC check structure, so the
decoder can build a circuit-level noise model directly from the
compiled waveform.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from .qldpc import QLDPCCode


# ─────────────────────────────────────────────────────────────────
#  Annotation
# ─────────────────────────────────────────────────────────────────

@dataclass
class CheckAnnotation:
    """Maps a compiled pulse to its role in a qLDPC check."""
    pulse_index: int
    check_type: str        # "X" or "Z"
    check_index: int
    qubit_index: int
    role: str              # "data" | "ancilla" | "syndrome"


@dataclass
class AnnotatedProgram:
    """A compiled Program with qLDPC check structure attached."""
    program: object                       # qhal.qhal_cpp.Program
    code: Optional[QLDPCCode] = None
    annotations: List[CheckAnnotation] = field(default_factory=list)
    # Map: qubit -> ("data" | "ancilla")
    qubit_roles: Dict[int, str] = field(default_factory=dict)

    def check_pulses(self, check_type: str, check_index: int) -> List[int]:
        """Pulse indices for a given check."""
        return [a.pulse_index for a in self.annotations
                if a.check_type == check_type and a.check_index == check_index]

    def total_pulses_for_check(self, check_type: str,
                               check_index: int) -> int:
        return len(self.check_pulses(check_type, check_index))


# ─────────────────────────────────────────────────────────────────
#  Annotation pass
# ─────────────────────────────────────────────────────────────────

def annotate_program(program, code: QLDPCCode) -> AnnotatedProgram:
    """
    Walk a compiled pulse program and attach qLDPC check structure.

    Requires each Pulse to carry a `qubit` attribute (set by the
    compiler for every pulse it emits).
    """
    ann = AnnotatedProgram(program=program, code=code)

    # Assume qubits [0 .. n_data-1] are data, the rest are ancilla
    n_data = code.n
    for q in range(n_data):
        ann.qubit_roles[q] = "data"
    # Any qubits beyond n_data are ancilla (for syndrome extraction)
    for p in program.pulses:
        if p.qubit >= n_data:
            ann.qubit_roles[p.qubit] = "ancilla"

    # Walk pulses
    for idx, pulse in enumerate(program.pulses):
        q = pulse.qubit
        if q < 0:
            continue

        role = ann.qubit_roles.get(q, "data")

        if q < n_data:
            # Data qubit: find every check it participates in
            for check_type, check_idx in code.qubit_checks(q):
                ann.annotations.append(CheckAnnotation(
                    pulse_index=idx,
                    check_type=check_type,
                    check_index=check_idx,
                    qubit_index=q,
                    role="data"))
        else:
            # Ancilla qubit: use its index to identify the check
            ancilla_idx = q - n_data
            if ancilla_idx < code.num_x_checks:
                ann.annotations.append(CheckAnnotation(
                    pulse_index=idx, check_type="X",
                    check_index=ancilla_idx, qubit_index=q, role="ancilla"))
            elif ancilla_idx < code.num_checks:
                ann.annotations.append(CheckAnnotation(
                    pulse_index=idx, check_type="Z",
                    check_index=ancilla_idx - code.num_x_checks,
                    qubit_index=q, role="ancilla"))

    return ann


# ─────────────────────────────────────────────────────────────────
#  Noise model for the decoder
# ─────────────────────────────────────────────────────────────────

@dataclass
class CheckNoise:
    """Circuit-level noise statistics for one check."""
    check_type: str
    check_index: int
    num_data_pulses: int
    num_ancilla_pulses: int
    total_duration_ns: float
    # Estimated error rate for this check's syndrome measurement
    estimated_error_rate: float


def build_check_noise(ann: AnnotatedProgram, calibration,
                      gate_error_per_ns: float = 1e-5) -> Dict[Tuple[str, int], CheckNoise]:
    """
    Build per-check circuit-level noise model from the annotated program.

    Args:
        ann: AnnotatedProgram
        calibration: object with .get(qubit) -> QubitCalibration
        gate_error_per_ns: base error rate per nanosecond of gate time
    """
    out: Dict[Tuple[str, int], CheckNoise] = {}

    for check_type in ("X", "Z"):
        n_checks = (ann.code.num_x_checks if check_type == "X"
                    else ann.code.num_z_checks)
        for ci in range(n_checks):
            pulses = [ann.program.pulses[i]
                      for i in ann.check_pulses(check_type, ci)]
            n_data = sum(1 for p in pulses if p.qubit < ann.code.n)
            n_anc  = sum(1 for p in pulses if p.qubit >= ann.code.n)
            duration = sum(p.duration_ns for p in pulses)
            # Simple noise model: error rate proportional to duration
            err_rate = 1.0 - np.exp(-gate_error_per_ns * duration)
            out[(check_type, ci)] = CheckNoise(
                check_type=check_type,
                check_index=ci,
                num_data_pulses=n_data,
                num_ancilla_pulses=n_anc,
                total_duration_ns=duration,
                estimated_error_rate=err_rate,
            )

    return out


def build_detector_error_model(ann: AnnotatedProgram,
                               calibration,
                               gate_error_per_ns: float = 1e-5) -> Dict:
    """
    Build a detector error model (DEM) from the annotated program.

    The DEM maps each check to a probability that its syndrome is
    corrupted. This is exactly what the BP+OSD decoder needs.
    """
    check_noise = build_check_noise(ann, calibration, gate_error_per_ns)

    dem = {
        "code_name": ann.code.name,
        "n": ann.code.n,
        "k_x_checks": ann.code.num_x_checks,
        "k_z_checks": ann.code.num_z_checks,
        "error_rates": {
            f"{ct}_{ci}": cn.estimated_error_rate
            for (ct, ci), cn in check_noise.items()
        },
        "durations_ns": {
            f"{ct}_{ci}": cn.total_duration_ns
            for (ct, ci), cn in check_noise.items()
        },
        "check_noise": check_noise,
    }
    return dem
