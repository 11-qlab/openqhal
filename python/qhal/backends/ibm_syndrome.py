"""
IBM syndrome extraction backend.

Two output paths:
  1. to_qasm()      -> OpenQASM 3 text (for QICK, Zurich, storage)
  2. to_qiskit()    -> Qiskit QuantumCircuit (for IBM hardware)

The Qiskit path uses qc.reset() so IBM's transpiler can convert to
hardware-native MidCircuitReset. Raw QASM with `reset` fails on IBM
Runtime with error 6056.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

from ..qldpc import QLDPCCode
from ..qldpc_scheduler import Schedule


# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SyndromeConfig:
    num_rounds: int = 3
    reset_ancillas: bool = True
    basis: str = "Z"           # initial data state: "Z" or "X"
    add_barriers: bool = False


# ─────────────────────────────────────────────────────────────────────
#  Syndrome circuit builder
# ─────────────────────────────────────────────────────────────────────

class SyndromeCircuit:
    """Build syndrome extraction circuits from a Schedule."""

    def __init__(self, schedule: Schedule, config: SyndromeConfig = None):
        self.schedule = schedule
        self.code = schedule.code
        self.config = config or SyndromeConfig()

        # Qubit index layout
        self.n_data = self.code.n
        self.n_x = self.code.num_x_checks
        self.n_z = self.code.num_z_checks
        self.n_anc = self.n_x + self.n_z
        self.n_total = self.n_data + self.n_anc

    def _ancilla_index(self, check_type: str, check_index: int) -> int:
        if check_type == "X":
            return self.n_data + check_index
        return self.n_data + self.n_x + check_index

    # ── Qiskit path (preferred for IBM hardware) ──

    def to_qiskit(self, use_reset: bool = True) -> Any:
        """
        Build a Qiskit QuantumCircuit.
        Uses a single, flat classical register to avoid IBM compiler Error 6056.
        """
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        qr = QuantumRegister(self.n_total, "q")
        # FIX: Single flat classical register for all rounds
        cr = ClassicalRegister(self.n_anc * self.config.num_rounds, "syn")
        qc = QuantumCircuit(qr, cr, name="syndrome")
        self._use_reset = use_reset

        # Optional: prepare data qubits in |+> for X basis
        if self.config.basis == "X":
            for d in range(self.n_data):
                qc.h(qr[d])

        for rnd in range(self.config.num_rounds):
            # FIX: Skip reset on rnd==0. Hardware is already in |0>. 
            # Resetting at t=0 causes hardware compiler faults.
            if self.config.reset_ancillas and getattr(self, "_use_reset", True) and rnd > 0:
                for a in range(self.n_anc):
                    qc.reset(qr[self.n_data + a])

            # X-ancillas need H for X-basis measurement
            for xi in range(self.n_x):
                qc.h(qr[self._ancilla_index("X", xi)])

            # Apply the schedule layers
            for layer in self.schedule.layers:
                if self.config.add_barriers:
                    qc.barrier(qr)
                for (ct, data, ci) in layer.gates:
                    anc = self._ancilla_index(ct, ci)
                    if ct == "X":
                        qc.cx(qr[anc], qr[data])
                    else:
                        qc.cx(qr[data], qr[anc])

            # X-ancillas need H before measurement
            for xi in range(self.n_x):
                qc.h(qr[self._ancilla_index("X", xi)])

            # Measure ancillas into the flat register for this round
            for a in range(self.n_anc):
                qc.measure(qr[self.n_data + a], cr[rnd * self.n_anc + a])

        return qc

    # ── QASM path (for non-IBM backends, storage, debugging) ──

    def to_qasm(self) -> str:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{self.n_total}] q;",
            f"bit[{self.n_anc}] syn;",
            "",
        ]

        if self.config.basis == "X":
            for d in range(self.n_data):
                lines.append(f"h q[{d}];")

        for rnd in range(self.config.num_rounds):
            lines.append(f"// ---- round {rnd} ----")

            if self.config.reset_ancillas:
                for a in range(self.n_anc):
                    lines.append(f"reset q[{self.n_data + a}];")

            for xi in range(self.n_x):
                lines.append(f"h q[{self._ancilla_index('X', xi)}];")

            for layer in self.schedule.layers:
                if self.config.add_barriers:
                    lines.append("barrier q;")
                for (ct, data, ci) in layer.gates:
                    anc = self._ancilla_index(ct, ci)
                    if ct == "X":
                        lines.append(f"cx q[{anc}], q[{data}];")
                    else:
                        lines.append(f"cx q[{data}], q[{anc}];")

            for xi in range(self.n_x):
                anc = self._ancilla_index("X", xi)
                lines.append(f"h q[{anc}];")
                lines.append(f"syn[{xi}] = measure q[{anc}];")
            for zi in range(self.n_z):
                anc = self._ancilla_index("Z", zi)
                lines.append(f"syn[{self.n_x + zi}] = measure q[{anc}];")

        return "\n".join(lines) + "\n"

    # ── Summary ──

    def summary(self) -> Dict:
        return {
            "n_data": self.n_data,
            "n_x_checks": self.n_x,
            "n_z_checks": self.n_z,
            "n_total_qubits": self.n_total,
            "n_layers": self.schedule.depth,
            "n_2q_gates_per_round": self.schedule.num_gates,
            "n_rounds": self.config.num_rounds,
            "n_2q_gates_total": self.schedule.num_gates * self.config.num_rounds,
            "n_measurements": self.n_anc * self.config.num_rounds,
        }


# Backwards-compat alias
SyndromeQASM = SyndromeCircuit


# ─────────────────────────────────────────────────────────────────────
#  Detector event extraction
# ─────────────────────────────────────────────────────────────────────

def extract_detector_events(raw_syndromes: np.ndarray,
                            n_x: int, n_z: int,
                            num_rounds: int) -> np.ndarray:
    """
    Convert raw ancilla measurements into detector events.

    Detector = XOR of same-ancilla measurements in consecutive rounds.
    Round 0 compares against the deterministic initial value (0).
    """
    shots = raw_syndromes.shape[0]
    n_anc = n_x + n_z
    n_det = n_anc * num_rounds
    events = np.zeros((shots, n_det), dtype=np.uint8)

    for shot in range(shots):
        prev = np.zeros(n_anc, dtype=np.uint8)
        for r in range(num_rounds):
            curr = raw_syndromes[shot, r * n_anc:(r + 1) * n_anc]
            events[shot, r * n_anc:(r + 1) * n_anc] = curr ^ prev
            prev = curr

    return events
