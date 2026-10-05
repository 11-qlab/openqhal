"""
Bivariate Bicycle code construction and decoding benchmark.

Implements the [[144, 12, 12]] Gross Code from Bravyi et al., Nature 2024.
Uses bposd.css for CSS construction and ldpc for BP+OSD decoding.

Requires: pip install --no-deps ldpc bposd
"""
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
sys.path.insert(0, "python")

import numpy as np
import time
from ldpc import bposd_decoder
from bposd.css import css_code
from qhal.joint import repetition_fidelity


# ─────────────────────────────────────────────────────────────────
#  BB code construction
# ─────────────────────────────────────────────────────────────────

def poly_matrix(l, m, powers):
    """Build (l*m) x (l*m) circulant matrix from polynomial monomials.

    powers: list of (x_power, y_power) tuples.
    """
    M = np.zeros((l * m, l * m), dtype=np.uint8)
    for i in range(l):
        for j in range(m):
            row = i * m + j
            for (a, b) in powers:
                col_i = (i + a) % l
                col_j = (j + b) % m
                M[row, col_i * m + col_j] ^= 1
    return M


def bb_code(l, m, c_powers, d_powers):
    """Build BB code parity check matrices.

    Returns (Hx, Hz) where each is (l*m) x (2*l*m).
    """
    A = poly_matrix(l, m, c_powers)
    B = poly_matrix(l, m, d_powers)
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])
    return Hx, Hz


def make_gross_code():
    """The [[144, 12, 12]] gross code from Bravyi et al.

    c = x^3 + y + y^2
    d = y^3 + x + x^2
    """
    c_powers = [(3, 0), (0, 1), (0, 2)]
    d_powers = [(0, 3), (1, 0), (2, 0)]
    return bb_code(12, 6, c_powers, d_powers)


def make_small_bb():
    """Smaller [[98, 6, 12]] variant for faster testing.

    c = x + y^3 + y^4
    d = y + x^3 + x^4
    """
    c_powers = [(1, 0), (0, 3), (0, 4)]
    d_powers = [(0, 1), (3, 0), (4, 0)]
    return bb_code(7, 7, c_powers, d_powers)


# ─────────────────────────────────────────────────────────────────
#  Decoding
# ─────────────────────────────────────────────────────────────────

def bb_logical_fidelity(Hx, Hz, eta, shots=2000, seed=42, verbose=False):
    """BP+OSD decoding over random-overwrite channel.

    Success = residual error commutes with all logical operators
    (i.e., no logical X or Z error remains).
    """
    rng = np.random.default_rng(seed)
    n = Hx.shape[1]
    H = np.vstack([Hx, Hz])

    decoder = bposd_decoder(
        H, error_rate=(1 - eta) / 2, max_iter=30,
        bp_method="ms", osd_method="osd_cs", osd_order=4)

    qcode = css_code(hx=Hx, hz=Hz)
    # bposd returns scipy sparse csr_matrix — convert to dense uint8
    lx = qcode.lx.toarray().astype(np.uint8)
    lz = qcode.lz.toarray().astype(np.uint8)

    correct = 0
    t0 = time.time()
    for i in range(shots):
        e = (rng.random(n) >= eta).astype(np.uint8)
        s = (H @ e) % 2
        c = decoder.decode(s)
        residual = (e + c) % 2
        if not (lx @ residual % 2).any() and not (lz @ residual % 2).any():
            correct += 1
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i+1}/{shots}  ({time.time()-t0:.1f}s)")
    return correct / shots


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Bivariate Bicycle Code Benchmark")
    print("=" * 60)

    # ── Build the gross code ──
    print("\n[1] Building [[144, 12, 12]] gross code...")
    Hx, Hz = make_gross_code()
    n = Hx.shape[1]
    m = Hx.shape[0]
    print(f"    Hx: {Hx.shape}  Hz: {Hz.shape}")

    qcode = css_code(hx=Hx, hz=Hz)
    qcode.test()
    # bposd uses capitalized attribute names
    print(f"    N = {qcode.N}")
    print(f"    K = {qcode.K}  (expect 12)")
    # D is nan for large codes — bposd cannot compute exact distance
    d_val = getattr(qcode, 'D', None)
    if d_val is not None and not (isinstance(d_val, float) and np.isnan(d_val)):
        print(f"    D = {d_val}")
    else:
        print(f"    D = (not computed — use published value 12)")
    print(f"    rate = {qcode.K / qcode.N:.4f}")

    # ── Sanity check: CSS commutation ──
    if ((Hx @ Hz.T) % 2).any():
        print("    ERROR: CSS commutation failed")
    else:
        print("    CSS commutation: OK")

    # ── Decode at several efficiencies ──
    print("\n[2] BP+OSD decoding over random-overwrite channel")
    print(f"    (each point: 2000 shots, ~5-15 min)")

    print()
    print(f"{'eta':>6} | {'Rep N=15':>12} | {'BB [[144,12]]':>14} | "
          f"{'Rep rate':>10} | {'BB rate':>10}")
    print("-" * 72)

    for eta in [0.85, 0.90, 0.95, 0.99]:
        rep = repetition_fidelity(15, eta)
        t0 = time.time()
        bb = bb_logical_fidelity(Hx, Hz, eta, shots=2000)
        dt = time.time() - t0
        print(f"{eta:>6.2f} | {rep:>12.6f} | {bb:>14.6f} | "
              f"{1/15:>10.4f} | {qcode.K/qcode.N:>10.4f}   [{dt:.0f}s]")

    print()
    print("[3] Overhead comparison at matched distance d=12")
    print(f"    BB [[144,12,12]] : 144 physical, 12 logical, 1 block")
    print(f"    Surface d=12     : 144 physical per logical, 12 blocks = 1728 physical")
    print(f"    Overhead ratio   : 1728 / 144 = 12x")
