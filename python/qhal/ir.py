"""
Immutable Circuit IR for OpenQHAL.

Replaces the mutable list-of-Gates approach with a frozen, hashable,
serializable representation. Every operation returns a new Circuit.

Why immutable:
  - Hashable -> cacheable (memoize compilation results)
  - Serializable -> shareable across processes
  - Structural equality -> diffable, testable
  - No hidden mutation -> safe to pass across threads
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import json


# ─────────────────────────────────────────────────────────────────────
#  Frozen gate
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class Gate:
    """A single quantum gate. Immutable, hashable."""
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "params", tuple(self.params))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "qubits": list(self.qubits),
                "params": list(self.params)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Gate":
        return Gate(d["name"], tuple(d["qubits"]),
                    tuple(d.get("params", [])))

    def is_measure(self) -> bool:
        return self.name in ("MEASURE", "M")


# ─────────────────────────────────────────────────────────────────────
#  Immutable circuit
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Circuit:
    """An immutable sequence of gates on a fixed number of qubits.

    Build with fluent methods; each returns a new Circuit.
    """
    num_qubits: int
    gates: Tuple[Gate, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "gates", tuple(self.gates))
        for g in self.gates:
            for q in g.qubits:
                if q < 0 or q >= self.num_qubits:
                    raise ValueError(
                        f"gate {g.name} references qubit {q} but "
                        f"circuit has {self.num_qubits} qubits")

    # ── fluent builders ──
    def _add(self, gate: Gate) -> "Circuit":
        return Circuit(self.num_qubits, self.gates + (gate,))

    def h(self, q: int) -> "Circuit":           return self._add(Gate("H", (q,)))
    def x(self, q: int) -> "Circuit":           return self._add(Gate("X", (q,)))
    def y(self, q: int) -> "Circuit":           return self._add(Gate("Y", (q,)))
    def z(self, q: int) -> "Circuit":           return self._add(Gate("Z", (q,)))
    def s(self, q: int) -> "Circuit":           return self._add(Gate("S", (q,)))
    def t(self, q: int) -> "Circuit":           return self._add(Gate("T", (q,)))
    def rx(self, theta: float, q: int) -> "Circuit":
        return self._add(Gate("RX", (q,), (float(theta),)))
    def ry(self, theta: float, q: int) -> "Circuit":
        return self._add(Gate("RY", (q,), (float(theta),)))
    def rz(self, theta: float, q: int) -> "Circuit":
        return self._add(Gate("RZ", (q,), (float(theta),)))
    def cnot(self, ctrl: int, tgt: int) -> "Circuit":
        return self._add(Gate("CNOT", (ctrl, tgt)))
    def cx(self, ctrl: int, tgt: int) -> "Circuit": return self.cnot(ctrl, tgt)
    def cz(self, ctrl: int, tgt: int) -> "Circuit":
        return self._add(Gate("CZ", (ctrl, tgt)))
    def swap(self, a: int, b: int) -> "Circuit":
        return self._add(Gate("SWAP", (a, b)))
    def measure(self, qubits) -> "Circuit":
        if isinstance(qubits, int):
            qubits = (qubits,)
        return self._add(Gate("MEASURE", tuple(qubits)))

    # ── inspection ──
    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def depth(self) -> int:
        """Circuit depth: longest chain of gates on any single qubit."""
        per_qubit = [0] * self.num_qubits
        for g in self.gates:
            if g.is_measure():
                continue
            d = max(per_qubit[q] for q in g.qubits) + 1
            for q in g.qubits:
                per_qubit[q] = d
        return max(per_qubit) if per_qubit else 0

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for g in self.gates:
            out[g.name] = out.get(g.name, 0) + 1
        return out

    def qubits_used(self) -> int:
        used = set()
        for g in self.gates:
            used.update(g.qubits)
        return len(used)

    # ── serialization ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "gates": [g.to_dict() for g in self.gates],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Circuit":
        return Circuit(
            num_qubits=d["num_qubits"],
            gates=tuple(Gate.from_dict(g) for g in d["gates"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "Circuit":
        return Circuit.from_dict(json.loads(s))

    def to_qasm(self) -> str:
        """Emit OpenQASM 3."""
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{self.num_qubits}] q;",
            f"bit[{self.num_qubits}] meas;",
        ]
        for g in self.gates:
            n = g.name
            if g.is_measure():
                for q in g.qubits:
                    lines.append(f"meas[{q}] = measure q[{q}];")
            elif n in ("H", "X", "Y", "Z", "S", "T"):
                lines.append(f"{n.lower()} q[{g.qubits[0]}];")
            elif n in ("RX", "RY", "RZ"):
                lines.append(f"{n.lower()}({g.params[0]}) q[{g.qubits[0]}];")
            elif n in ("CNOT", "CX"):
                lines.append(f"cx q[{g.qubits[0]}], q[{g.qubits[1]}];")
            elif n == "CZ":
                lines.append(f"cz q[{g.qubits[0]}], q[{g.qubits[1]}];")
            elif n == "SWAP":
                lines.append(f"swap q[{g.qubits[0]}], q[{g.qubits[1]}];")
            else:
                raise ValueError(f"IR: cannot emit unknown gate '{n}'")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return (f"Circuit({self.num_qubits} qubits, "
                f"{len(self.gates)} gates, depth={self.depth()})")


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def bell_pair(q0: int = 0, q1: int = 1) -> Circuit:
    """The canonical Bell state circuit."""
    return Circuit(2).h(q0).cnot(q0, q1).measure((q0, q1))


def ghz(n: int) -> Circuit:
    """GHZ state on n qubits, measured."""
    c = Circuit(n).h(0)
    for i in range(n - 1):
        c = c.cnot(i, i + 1)
    return c.measure(tuple(range(n)))
