"""
Circuit transpiler: gate decomposition, routing, optimization.

Takes a Circuit in the logical qubit space and produces a Circuit
that respects the hardware's coupling map (which pairs of qubits
can interact directly).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional
from collections import deque

from .ir import Circuit, Gate


# ─────────────────────────────────────────────────────────────────────
#  Coupling map
# ─────────────────────────────────────────────────────────────────────

class CouplingMap:
    """Which pairs of qubits can directly interact."""

    def __init__(self, edges: List[Tuple[int, int]], n: int):
        self.n = n
        self.adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for a, b in edges:
            self.adj[a].add(b)
            self.adj[b].add(a)

    def connected(self, a: int, b: int) -> bool:
        return b in self.adj[a]

    def shortest_path(self, a: int, b: int) -> List[int]:
        """BFS path from a to b. Returns [a, ..., b] or [] if unreachable."""
        if a == b:
            return [a]
        prev: Dict[int, int] = {a: -1}
        q = deque([a])
        while q:
            node = q.popleft()
            for nb in self.adj[node]:
                if nb in prev:
                    continue
                prev[nb] = node
                if nb == b:
                    path = [b]
                    while prev[path[-1]] != -1:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                q.append(nb)
        return []

    @staticmethod
    def line(n: int) -> "CouplingMap":
        """0-1-2-...-(n-1) chain."""
        return CouplingMap([(i, i + 1) for i in range(n - 1)], n)

    @staticmethod
    def grid(rows: int, cols: int) -> "CouplingMap":
        edges = []
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    edges.append((i, i + 1))
                if r + 1 < rows:
                    edges.append((i, i + cols))
        return CouplingMap(edges, rows * cols)

    @staticmethod
    def all_to_all(n: int) -> "CouplingMap":
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return CouplingMap(edges, n)


# ─────────────────────────────────────────────────────────────────────
#  Gate decomposition
# ─────────────────────────────────────────────────────────────────────

def decompose_swap(a: int, b: int) -> List[Gate]:
    """SWAP as 3 CNOTs."""
    return [Gate("CNOT", (a, b)),
            Gate("CNOT", (b, a)),
            Gate("CNOT", (a, b))]


def decompose_cz(a: int, b: int) -> List[Gate]:
    """CZ as H(target); CNOT; H(target)."""
    return [Gate("H", (b,)),
            Gate("CNOT", (a, b)),
            Gate("H", (b,))]


# ─────────────────────────────────────────────────────────────────────
#  Transpiler
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TranspileResult:
    circuit: Circuit
    swaps_inserted: int
    depth_before: int
    depth_after: int
    gates_before: int
    gates_after: int


class Transpiler:
    """Maps logical circuits to physical topologies."""

    def __init__(self, coupling: CouplingMap,
                 optimize: bool = True):
        self.coupling = coupling
        self.optimize = optimize

    def transpile(self, circ: Circuit) -> TranspileResult:
        """Route and optimize a circuit for the coupling map."""
        # Step 1 — decompose SWAP/CZ to native gates
        gates = self._decompose(circ.gates)

        # Step 2 — route: insert SWAPs to make CNOTs adjacent
        routed, swaps = self._route(gates, self.coupling.n)

        # Step 3 — optimize if requested
        if self.optimize:
            routed = self._optimize(routed)

        out = Circuit(self.coupling.n, tuple(routed))
        return TranspileResult(
            circuit=out,
            swaps_inserted=swaps,
            depth_before=circ.depth(),
            depth_after=out.depth(),
            gates_before=len(circ.gates),
            gates_after=len(out.gates),
        )

    # ── decomposition ──
    def _decompose(self, gates: Tuple[Gate, ...]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if g.name == "SWAP":
                out.extend(decompose_swap(*g.qubits))
            elif g.name == "CZ":
                out.extend(decompose_cz(*g.qubits))
            else:
                out.append(g)
        return out

    # ── routing ──
    def _route(self, gates: List[Gate], n: int) -> Tuple[List[Gate], int]:
        """Insert SWAPs so that every CNOT acts on adjacent qubits.

        Uses a simple greedy strategy: for each CNOT that isn't adjacent,
        insert SWAPs along the shortest path from control to target.
        """
        # virtual -> physical mapping
        v2p = list(range(n))
        p2v = list(range(n))
        swaps = 0
        out: List[Gate] = []

        for g in gates:
            if g.name not in ("CNOT", "CX") or g.is_measure():
                # Map qubits to physical
                out.append(Gate(g.name,
                                tuple(v2p[q] for q in g.qubits),
                                g.params))
                continue

            ctrl, tgt = g.qubits
            pc, pt = v2p[ctrl], v2p[tgt]

            if self.coupling.connected(pc, pt):
                out.append(Gate("CNOT", (pc, pt)))
                continue

            # Find shortest physical path, SWAP along it
            path = self.coupling.shortest_path(pc, pt)
            if not path:
                raise RuntimeError(
                    f"no path from physical qubit {pc} to {pt}")

            # Move ctrl closer to tgt by SWAPping along path
            for i in range(len(path) - 2):
                a, b = path[i], path[i + 1]
                # SWAP physical a, b
                out.append(Gate("CNOT", (a, b)))
                out.append(Gate("CNOT", (b, a)))
                out.append(Gate("CNOT", (a, b)))
                swaps += 1
                # Update mapping
                va, vb = p2v[a], p2v[b]
                v2p[va], v2p[vb] = b, a
                p2v[a], p2v[b] = vb, va

            # Now v2p[ctrl] should be adjacent to pt
            out.append(Gate("CNOT", (v2p[ctrl], pt)))

        return out, swaps

    # ── optimization passes ──
    def _optimize(self, gates: List[Gate]) -> List[Gate]:
        """Iterate peephole passes until fixpoint (max 5 rounds)."""
        for _ in range(5):
            before = len(gates)
            gates = self._cancel_inverses(gates)
            gates = self._merge_rotations(gates)
            if len(gates) == before:
                break
        return gates

    @staticmethod
    def _cancel_inverses(gates: List[Gate]) -> List[Gate]:
        """Remove adjacent self-inverse pairs on the same qubit(s)."""
        self_inverse = {"X", "Y", "Z", "H", "CNOT", "CX", "CZ"}
        out: List[Gate] = []
        for g in gates:
            if (out
                    and g.name in self_inverse
                    and out[-1].name == g.name
                    and out[-1].qubits == g.qubits
                    and out[-1].params == g.params):
                out.pop()
            else:
                out.append(g)
        return out

    @staticmethod
    def _merge_rotations(gates: List[Gate]) -> List[Gate]:
        """Combine adjacent rotations on the same qubit and axis."""
        out: List[Gate] = []
        for g in gates:
            if (out
                    and g.name in ("RX", "RY", "RZ")
                    and out[-1].name == g.name
                    and out[-1].qubits == g.qubits):
                total = out[-1].params[0] + g.params[0]
                out.pop()
                # Drop the pair entirely if it sums to 0
                if abs(total) > 1e-12:
                    out.append(Gate(g.name, g.qubits, (total,)))
            else:
                out.append(g)
        return out
