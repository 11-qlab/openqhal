"""
qLDPC code construction and check structure.

Supports Bivariate Bicycle (BB) codes and generic CSS codes.
Exposes Hx, Hz, and logical operators for compiler-decoder co-design.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  Bivariate Bicycle code construction
# ─────────────────────────────────────────────────────────────────

def _poly_matrix(l: int, m: int, powers: list) -> np.ndarray:
    """Build (l*m) x (l*m) circulant from polynomial monomials."""
    M = np.zeros((l * m, l * m), dtype=np.uint8)
    for i in range(l):
        for j in range(m):
            row = i * m + j
            for (a, b) in powers:
                col_i = (i + a) % l
                col_j = (j + b) % m
                M[row, col_i * m + col_j] ^= 1
    return M


@dataclass
class QLDPCCode:
    """A CSS qLDPC code with X and Z parity checks."""
    name: str
    hx: np.ndarray          # (num_X_checks, n) uint8
    hz: np.ndarray          # (num_Z_checks, n) uint8
    lx: Optional[np.ndarray] = None   # (k, n) logical X operators
    lz: Optional[np.ndarray] = None   # (k, n) logical Z operators

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def num_x_checks(self) -> int:
        return self.hx.shape[0]

    @property
    def num_z_checks(self) -> int:
        return self.hz.shape[0]

    @property
    def num_checks(self) -> int:
        return self.num_x_checks + self.num_z_checks

    def check_qubits(self, check_type: str, check_index: int) -> list:
        """Which data qubits participate in a given check."""
        H = self.hx if check_type == "X" else self.hz
        if check_index < 0 or check_index >= H.shape[0]:
            raise IndexError(f"check index {check_index} out of range")
        return [int(q) for q in range(self.n) if H[check_index, q]]

    def qubit_checks(self, qubit: int) -> list:
        """Which checks a given qubit participates in."""
        out = []
        for ci in range(self.num_x_checks):
            if self.hx[ci, qubit]:
                out.append(("X", ci))
        for ci in range(self.num_z_checks):
            if self.hz[ci, qubit]:
                out.append(("Z", ci))
        return out

    def verify_css(self) -> bool:
        """Verify Hx @ Hz^T = 0 (mod 2)."""
        return not ((self.hx @ self.hz.T) % 2).any()

    def __repr__(self) -> str:
        return (f"QLDPCCode({self.name}, n={self.n}, "
                f"checks={self.num_checks})")


# ─────────────────────────────────────────────────────────────────
#  Prebuilt codes
# ─────────────────────────────────────────────────────────────────

def bb_code(l: int, m: int, c_powers: list, d_powers: list,
            name: str = "BB") -> QLDPCCode:
    """Build a Bivariate Bicycle code from polynomials."""
    A = _poly_matrix(l, m, c_powers)
    B = _poly_matrix(l, m, d_powers)
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])
    return QLDPCCode(name=f"{name}[[{2*l*m}]]", hx=Hx, hz=Hz)


def gross_code() -> QLDPCCode:
    """The [[144, 12, 12]] gross code (Bravyi et al., Nature 2024)."""
    c_powers = [(3, 0), (0, 1), (0, 2)]
    d_powers = [(0, 3), (1, 0), (2, 0)]
    return bb_code(12, 6, c_powers, d_powers, name="Gross")


def small_bb() -> QLDPCCode:
    """The [[98, 6, 12]] variant — smaller, faster to test."""
    c_powers = [(1, 0), (0, 3), (0, 4)]
    d_powers = [(0, 1), (3, 0), (4, 0)]
    return bb_code(7, 7, c_powers, d_powers, name="SmallBB")


def repetition(n: int) -> QLDPCCode:
    """Trivial repetition code as a CSS code."""
    hx = np.zeros((1, n), dtype=np.uint8)
    hx[0, :] = 1
    hz = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        hz[i, i] = 1
        hz[i, i + 1] = 1
    return QLDPCCode(name=f"Rep[[{n}]]", hx=hx, hz=hz)
