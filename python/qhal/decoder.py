"""
Decoder bridge: connect the CDCD detector error model to BP+OSD.

Consumes the output of cdcd.build_detector_error_model and runs the
ldpc package's BP+OSD decoder on sampled syndromes.

This closes the compiler-decoder loop:
    compile -> annotate -> build DEM -> sample -> decode -> verify
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .qldpc import QLDPCCode


# ─────────────────────────────────────────────────────────────────
#  Detector error model for BP+OSD
# ─────────────────────────────────────────────────────────────────

@dataclass
class DetectorErrorModel:
    """A sparse parity-check matrix + error probabilities + logical ops.

    Matches the format BP+OSD expects:
      H: (num_detectors, num_error_mechanisms) uint8
      p: (num_error_mechanisms,) float
      L: (num_logicals, num_error_mechanisms) uint8  -- logical observables
    """
    H: np.ndarray
    p: np.ndarray
    code: QLDPCCode
    num_detectors: int
    num_mechanisms: int
    L: np.ndarray = None   # (num_logicals, num_mechanisms)

    def __post_init__(self):
        if self.L is None:
            # Identity fallback: each mechanism is its own logical.
            # This will overcount errors, but won't undercount.
            self.L = np.eye(self.num_mechanisms, dtype=np.uint8)

    def __repr__(self) -> str:
        return (f"DEM(detectors={self.num_detectors}, "
                f"mechanisms={self.num_mechanisms}, "
                f"mean_p={self.p.mean():.5f})")


def build_dem_from_code(code: QLDPCCode,
                        per_check_error: float = 0.0025) -> DetectorErrorModel:
    """
    Build a detector error model directly from the code structure.

    Each check contributes one detector. Each data qubit that
    participates in a check creates an error mechanism that flips
    that detector.

    Args:
        code: QLDPCCode with hx, hz
        per_check_error: probability that a check's syndrome is
                         incorrectly reported (from cdcd's noise model)
    """
    n = code.n
    num_x = code.num_x_checks
    num_z = code.num_z_checks
    num_det = num_x + num_z

    # Each data qubit has two independent error mechanisms:
    #   - X error on qubit q flips all Z checks containing q
    #   - Z error on qubit q flips all X checks containing q
    #
    # Combined: 2 * n error mechanisms, one per (qubit, error_type)

    rows: List[List[int]] = []
    probs: List[float] = []

    # Z errors on data qubits -> flip X-check detectors
    for q in range(n):
        dets = [ci for ci in range(num_x) if code.hx[ci, q]]
        if dets:
            rows.append(dets)
            probs.append(per_check_error)

    # X errors on data qubits -> flip Z-check detectors
    for q in range(n):
        dets = [num_x + ci for ci in range(num_z) if code.hz[ci, q]]
        if dets:
            rows.append(dets)
            probs.append(per_check_error)

    num_mech = len(rows)
    H = np.zeros((num_det, num_mech), dtype=np.uint8)
    for j, row in enumerate(rows):
        for i in row:
            H[i, j] = 1

    # Build logical observables from the code's lx, lz matrices.
    # If lx/lz are not provided, fall back to identity (conservative).
    if code.lx is not None and code.lz is not None:
        lx = np.array(code.lx, dtype=np.uint8) if not hasattr(code.lx, "toarray") else code.lx.toarray().astype(np.uint8)
        lz = np.array(code.lz, dtype=np.uint8) if not hasattr(code.lz, "toarray") else code.lz.toarray().astype(np.uint8)
        k = lx.shape[0]
        L = np.zeros((2 * k, num_mech), dtype=np.uint8)
        # First k rows: X-logical checks against Z-error mechanisms
        # (which are the first `n` mechanisms in our ordering)
        for i in range(k):
            L[i, :code.n] = lx[i]
        # Next k rows: Z-logical checks against X-error mechanisms
        for i in range(k):
            L[k + i, code.n:] = lz[i]
    else:
        L = np.eye(num_mech, dtype=np.uint8)

    return DetectorErrorModel(
        H=H,
        p=np.array(probs, dtype=np.float64),
        code=code,
        num_detectors=num_det,
        num_mechanisms=num_mech,
        L=L,
    )


# ─────────────────────────────────────────────────────────────────
#  BP+OSD decoding
# ─────────────────────────────────────────────────────────────────

@dataclass
class DecodeResult:
    shots: int
    logical_errors: int
    logical_error_rate: float
    mean_iterations: float
    total_time_s: float

    def __repr__(self) -> str:
        return (f"DecodeResult(shots={self.shots}, "
                f"LER={self.logical_error_rate:.6f}, "
                f"iters={self.mean_iterations:.1f}, "
                f"time={self.total_time_s:.2f}s)")


def decode_with_bposd(dem: DetectorErrorModel,
                      physical_error_rate: float,
                      shots: int = 5000,
                      max_iter: int = 30,
                      seed: int = 42,
                      verbose: bool = False) -> DecodeResult:
    """
    Run BP+OSD decoding on the detector error model.

    Generates random error patterns at the given physical error rate,
    computes syndromes, decodes, and counts logical failures.
    """
    import time
    from ldpc import bposd_decoder

    rng = np.random.default_rng(seed)
    n_mech = dem.num_mechanisms
    n_det = dem.num_detectors

    decoder = bposd_decoder(
        dem.H,
        error_rate=physical_error_rate,
        max_iter=max_iter,
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=4,
    )

    logical_errors = 0
    total_iters = 0
    t0 = time.time()

    for shot in range(shots):
        # Random error pattern at rate p
        e = (rng.random(n_mech) < physical_error_rate).astype(np.uint8)

        # Syndrome = H @ e mod 2
        s = (dem.H @ e) % 2

        # Decode
        correction = decoder.decode(s)
        total_iters += getattr(decoder, "iterations", 0) or 0

        # Residual error after correction
        residual = (e + correction) % 2

        # Syndrome satisfied? If not, this is a "detected" failure
        # (decoder didn't converge). Count it.
        syndrome_ok = not ((dem.H @ residual) % 2).any()

        # Logical observable flipped? If yes, this is a logical error.
        # We need the residual to commute with stabilizers AND
        # anticommute with at least one logical.
        if syndrome_ok and dem.L is not None:
            # L @ residual gives the logical parity flips
            logical_flips = (dem.L @ residual) % 2
            if logical_flips.any():
                logical_errors += 1
        elif not syndrome_ok:
            logical_errors += 1

        if verbose and (shot + 1) % 1000 == 0:
            print(f"    {shot+1}/{shots}  "
                  f"LER={logical_errors/(shot+1):.6f}  "
                  f"({time.time()-t0:.1f}s)")

    dt = time.time() - t0
    return DecodeResult(
        shots=shots,
        logical_errors=logical_errors,
        logical_error_rate=logical_errors / shots,
        mean_iterations=total_iters / shots if shots > 0 else 0.0,
        total_time_s=dt,
    )


# ─────────────────────────────────────────────────────────────────
#  End-to-end convenience
# ─────────────────────────────────────────────────────────────────

def co_design_benchmark(code: QLDPCCode,
                        physical_rates: List[float] = None,
                        shots: int = 5000,
                        per_check_error: float = 0.0025) -> Dict:
    """
    Full CDCD benchmark: build DEM, decode at several physical rates,
    return a results table.
    """
    if physical_rates is None:
        physical_rates = [0.001, 0.003, 0.005, 0.007, 0.010]

    dem = build_dem_from_code(code, per_check_error=per_check_error)
    results = {}

    for p in physical_rates:
        r = decode_with_bposd(dem, physical_error_rate=p, shots=shots)
        results[p] = r

    return {
        "code": code.name,
        "dem": dem,
        "results": results,
    }
