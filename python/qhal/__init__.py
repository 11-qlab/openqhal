"""
OpenQHAL — Quantum Hardware Abstraction Layer
Python frontend for the C++ core.
"""
from .qhal_cpp import (
    Channel,
    Envelope,
    Gate,
    Pulse,
    Program,
    QubitCalibration,
    Calibration,
    Compiler,
    Scheduler,
    ScheduleResult,
    Simulator,
    Waveform,
    render,
    save_npy,
    save_raw,
    save_csv,
    save_seqc,
    total_energy,
    Backend,
    MockAWG,
    ExecutionStats,
)

__version__ = "0.1.0"
__all__ = [
    "Channel", "Envelope",
    "Gate", "Pulse", "Program",
    "QubitCalibration", "Calibration",
    "Compiler", "Scheduler", "ScheduleResult",
    "Simulator",
    "Waveform", "render",
    "save_npy", "save_raw", "save_csv", "save_seqc",
    "total_energy",
    "Backend", "MockAWG", "ExecutionStats",
]


# ---------------------------------------------------------------------
# Convenience: a fluent Circuit DSL
# ---------------------------------------------------------------------
class Circuit:
    """Fluent circuit builder. Wraps a list of Gate objects."""

    def __init__(self, num_qubits: int = 1):
        self.num_qubits = num_qubits
        self._gates: list[Gate] = []

    def _add(self, name: str, qubits, params=None):
        self._gates.append(Gate(name, list(qubits), list(params or [])))
        return self

    # Single-qubit gates
    def h(self, q):           return self._add("H", [q])
    def x(self, q):           return self._add("X", [q])
    def y(self, q):           return self._add("Y", [q])
    def z(self, q):           return self._add("Z", [q])
    def s(self, q):           return self._add("S", [q])
    def t(self, q):           return self._add("T", [q])
    def rx(self, theta, q):   return self._add("RX", [q], [theta])
    def ry(self, theta, q):   return self._add("RY", [q], [theta])
    def rz(self, theta, q):   return self._add("RZ", [q], [theta])

    # Two-qubit gates
    def cnot(self, ctrl, tgt): return self._add("CNOT", [ctrl, tgt])
    def cx(self, ctrl, tgt):   return self._add("CNOT", [ctrl, tgt])

    # Measurement
    def measure(self, qubits):
        if isinstance(qubits, int):
            qubits = [qubits]
        return self._add("MEASURE", qubits)

    # Access
    @property
    def gates(self) -> list[Gate]:
        return list(self._gates)

    def __len__(self):
        return len(self._gates)

    def __repr__(self):
        lines = [f"Circuit({self.num_qubits} qubits, {len(self._gates)} gates)"]
        for g in self._gates:
            lines.append(f"  {g}")
        return "\n".join(lines)
