"""
Zero-noise extrapolation (ZNE) for syndrome extraction.

Wraps a circuit with gate-folding and extrapolates to zero noise.

Limitations:
  - Works best when the error is approximately linear in the scale factor.
  - Richardson extrapolation can amplify variance; use with many shots.
  - For syndrome extraction, apply per-observable, not per-circuit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Gate folding
# ─────────────────────────────────────────────────────────────────────

def fold_gates(qasm: str, scale: int = 3) -> str:
    """
    Return a QASM circuit equivalent to the input but with each
    2-qubit gate (cx, cz) replaced by (scale-1)/2 identity insertions:
        cx q[a], q[b];
    becomes
        cx q[a], q[b]; cx q[a], q[b]; cx q[a], q[b];
    for scale=3.

    Only supports odd integer scale factors >= 1.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be odd and >= 1, got {scale}")

    if scale == 1:
        return qasm

    lines = qasm.split("\n")
    out: List[str] = []
    reps = scale  # cx is its own inverse, so scale copies = identity^n

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("cx ") and stripped.endswith(";"):
            for _ in range(reps):
                out.append(line)
        elif stripped.startswith("cz ") and stripped.endswith(";"):
            for _ in range(reps):
                out.append(line)
        else:
            out.append(line)

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
#  Extrapolation methods
# ─────────────────────────────────────────────────────────────────────

def linear_extrapolation(scales: np.ndarray, values: np.ndarray) -> float:
    """Fit a*x+b and return b (value at scale=0)."""
    if len(scales) < 2:
        return float(values[0])
    coeffs = np.polyfit(scales, values, 1)
    return float(coeffs[1])


def richardson_extrapolation(scales: np.ndarray, values: np.ndarray,
                             order: int = 1) -> float:
    """Richardson extrapolation to zero noise using polynomial order."""
    if len(scales) < 2:
        return float(values[0])
    deg = min(len(scales) - 1, order + 1)
    coeffs = np.polyfit(scales, values, deg)
    return float(np.polyval(coeffs, 0.0))


def exponential_extrapolation(scales: np.ndarray, values: np.ndarray) -> float:
    """Fit a * exp(-b*x) + c and return c."""
    if len(scales) < 3:
        return float(values[0])
    try:
        from scipy.optimize import curve_fit
        def model(x, a, b, c):
            return a * np.exp(-b * x) + c
        p0 = (float(values[0] - values[-1]), 1.0, float(values[-1]))
        popt, _ = curve_fit(model, scales, values, p0=p0, maxfev=5000)
        return float(popt[2])
    except Exception:
        return linear_extrapolation(scales, values)


# ─────────────────────────────────────────────────────────────────────
#  High-level wrapper
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ZNEResult:
    scales: List[int]
    raw_values: List[float]
    mitigated: float
    method: str

    def __repr__(self) -> str:
        s = ", ".join(f"{v:.4f}" for v in self.raw_values)
        return (f"ZNEResult(method={self.method}, "
                f"raw=[{s}], mitigated={self.mitigated:.4f})")


def zne_run(run_fn: Callable[[str, int], float],
            qasm: str,
            scales: Tuple[int, ...] = (1, 3, 5),
            shots_per_scale: int = 2048,
            method: str = "linear") -> ZNEResult:
    """
    Run ZNE on a QASM circuit.

    Args:
        run_fn: callable(qasm, shots) -> scalar expectation value
        qasm: base circuit
        scales: odd integer scale factors
        shots_per_scale: shots at each scale
        method: "linear" | "richardson" | "exponential"

    Returns ZNEResult with raw and mitigated values.
    """
    raw_values = []
    for s in scales:
        folded = fold_gates(qasm, s)
        v = run_fn(folded, shots_per_scale)
        raw_values.append(float(v))

    scales_arr = np.array(scales, dtype=float)
    vals_arr = np.array(raw_values, dtype=float)

    if method == "linear":
        mit = linear_extrapolation(scales_arr, vals_arr)
    elif method == "richardson":
        mit = richardson_extrapolation(scales_arr, vals_arr)
    elif method == "exponential":
        mit = exponential_extrapolation(scales_arr, vals_arr)
    else:
        raise ValueError(f"unknown method: {method}")

    return ZNEResult(
        scales=list(scales),
        raw_values=raw_values,
        mitigated=mit,
        method=method,
    )


# ─────────────────────────────────────────────────────────────────────
#  Simplest use case: mitigate syndrome measurement fidelity
# ─────────────────────────────────────────────────────────────────────

def estimate_check_error(run_fn: Callable[[str, int], float],
                          qasm_single_check: str,
                          scales: Tuple[int, ...] = (1, 3, 5),
                          shots: int = 2048) -> Tuple[float, float]:
    """
    Measure a single check's syndrome extraction circuit at multiple
    noise scales and return (raw_error, mitigated_error).

    run_fn should return the probability that the syndrome is wrong
    for a known-prepared state.
    """
    result = zne_run(run_fn, qasm_single_check,
                     scales=scales, shots_per_scale=shots,
                     method="linear")
    raw = result.raw_values[0]
    return raw, result.mitigated
