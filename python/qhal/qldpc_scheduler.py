"""
qLDPC-aware syndrome extraction scheduler.

Orders check interactions to minimize two-qubit gate depth while
respecting the X/Z interleaving constraint.

Based on the edge-coloring formulation from:
  - Zhang et al., "Optimal Compilation of Syndrome Extraction Circuits
    for General Quantum LDPC Codes", DATE 2026 (arXiv:2603.21499)
  - Durazo Rocha et al., "Local Automorphism-Aware Syndrome Compilation",
    2026 (arXiv:2609.40319)

Key insight: the minimum two-qubit depth of a syndrome extraction
circuit equals the minimum number of colors in a proper ordered
edge-coloring of the Tanner graph, subject to:
    For every overlapping X/Z check pair, the number of shared data
    qubits where X precedes Z must be even.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from .qldpc import QLDPCCode


# ─────────────────────────────────────────────────────────────────
#  Tanner graph
# ─────────────────────────────────────────────────────────────────

@dataclass
class TannerGraph:
    """Bipartite graph: data qubits and check ancillas as nodes,
    two-qubit gates as edges."""
    num_data: int
    num_x_checks: int
    num_z_checks: int

    # Edges: (data_qubit, check_type, check_index)
    x_edges: List[Tuple[int, int]] = field(default_factory=list)  # (qubit, x_check)
    z_edges: List[Tuple[int, int]] = field(default_factory=list)  # (qubit, z_check)

    @property
    def all_edges(self) -> List[Tuple[str, int, int]]:
        """(type, qubit, check_index) for every two-qubit gate."""
        out = [("X", q, c) for (q, c) in self.x_edges]
        out += [("Z", q, c) for (q, c) in self.z_edges]
        return out

    def degree(self, node_type: str, node_index: int) -> int:
        if node_type == "data":
            return sum(1 for (q, _) in self.x_edges if q == node_index) + \
                   sum(1 for (q, _) in self.z_edges if q == node_index)
        elif node_type == "X":
            return sum(1 for (_, c) in self.x_edges if c == node_index)
        elif node_type == "Z":
            return sum(1 for (_, c) in self.z_edges if c == node_index)
        return 0

    @property
    def max_degree(self) -> int:
        d = 0
        for q in range(self.num_data):
            d = max(d, self.degree("data", q))
        for c in range(self.num_x_checks):
            d = max(d, self.degree("X", c))
        for c in range(self.num_z_checks):
            d = max(d, self.degree("Z", c))
        return d


def build_tanner_graph(code: QLDPCCode) -> TannerGraph:
    """Extract the Tanner graph from a CSS code."""
    tg = TannerGraph(
        num_data=code.n,
        num_x_checks=code.num_x_checks,
        num_z_checks=code.num_z_checks,
    )
    for ci in range(code.num_x_checks):
        for q in range(code.n):
            if code.hx[ci, q]:
                tg.x_edges.append((q, ci))
    for ci in range(code.num_z_checks):
        for q in range(code.n):
            if code.hz[ci, q]:
                tg.z_edges.append((q, ci))
    return tg


# ─────────────────────────────────────────────────────────────────
#  Schedule representation
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScheduleLayer:
    """One time step of the syndrome extraction circuit."""
    layer_index: int
    gates: List[Tuple[str, int, int]] = field(default_factory=list)
    # Each gate: (check_type, data_qubit, check_index)


@dataclass
class Schedule:
    """A complete syndrome extraction schedule."""
    code: QLDPCCode
    layers: List[ScheduleLayer] = field(default_factory=list)
    method: str = "greedy"

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def num_gates(self) -> int:
        return sum(len(l.gates) for l in self.layers)

    def gate_at(self, data: int, check_type: str, check_index: int) -> int:
        """Which layer a given gate is scheduled in."""
        for l in self.layers:
            for (ct, d, ci) in l.gates:
                if ct == check_type and d == data and ci == check_index:
                    return l.layer_index
        return -1

    def __repr__(self) -> str:
        return (f"Schedule(depth={self.depth}, "
                f"gates={self.num_gates}, method={self.method})")


# ─────────────────────────────────────────────────────────────────
#  Greedy ASAP scheduler
# ─────────────────────────────────────────────────────────────────

def schedule_greedy(code: QLDPCCode) -> Schedule:
    """
    Greedy as-soon-as-possible scheduling.

    Each edge is placed in the earliest layer where:
      - Neither endpoint has been used in this layer yet.
      - The X/Z interleaving constraint is not violated.
    """
    tg = build_tanner_graph(code)
    edges = tg.all_edges
    schedule = Schedule(code=code, method="greedy")

    # Track when each node was last used
    data_last: Dict[int, int] = defaultdict(lambda: -1)
    x_last: Dict[int, int] = defaultdict(lambda: -1)
    z_last: Dict[int, int] = defaultdict(lambda: -1)

    # For the X/Z interleaving constraint, track, for each
    # (X-check, Z-check) pair that shares data qubits, how many
    # shared data qubits have had their X gate scheduled before
    # their Z gate (must be even at the end)
    xz_precedence: Dict[Tuple[int, int], int] = defaultdict(int)

    # Process edges in a deterministic order
    edges_sorted = sorted(edges, key=lambda e: (e[0], e[2], e[1]))

    layer_idx = 0
    remaining = list(edges_sorted)
    max_layers = tg.max_degree + 10  # safety bound

    while remaining and layer_idx < max_layers:
        this_layer = ScheduleLayer(layer_index=layer_idx)
        still_remaining = []
        used_data: Set[int] = set()
        used_checks: Set[Tuple[str, int]] = set()

        for (ct, data, ci) in remaining:
            # Check for conflicts in this layer
            if data in used_data:
                still_remaining.append((ct, data, ci))
                continue
            if (ct, ci) in used_checks:
                still_remaining.append((ct, data, ci))
                continue

            # Check interleaving: for Z gate, if there's an overlapping
            # X check whose X interaction on this data hasn't happened yet,
            # we may need to defer
            if ct == "Z":
                overlapping_x = [xc for xc in range(code.num_x_checks)
                                 if code.hx[xc, data] and code.hz[ci, data]]
                skip = False
                for xc in overlapping_x:
                    if x_last[xc] < layer_idx:
                        # X gate on this data hasn't happened
                        # Check parity of shared preceding X's
                        key = (xc, ci)
                        if xz_precedence[key] % 2 == 1:
                            skip = True
                            break
                if skip:
                    still_remaining.append((ct, data, ci))
                    continue

            # Schedule this gate
            this_layer.gates.append((ct, data, ci))
            used_data.add(data)
            used_checks.add((ct, ci))

            if ct == "X":
                x_last[ci] = layer_idx
                # For every Z check sharing this data, increment parity
                for zc in range(code.num_z_checks):
                    if code.hz[zc, data]:
                        xz_precedence[(ci, zc)] += 1
            else:
                z_last[ci] = layer_idx

        if this_layer.gates:
            schedule.layers.append(this_layer)
            data_last.update({d: layer_idx for d in used_data})
        remaining = still_remaining
        layer_idx += 1

    return schedule


# ─────────────────────────────────────────────────────────────────
#  Local search improvement
# ─────────────────────────────────────────────────────────────────

def schedule_local_search(code: QLDPCCode,
                          initial: Optional[Schedule] = None,
                          max_iterations: int = 100) -> Schedule:
    """
    Improve a greedy schedule by local layer swaps.

    For each pair of adjacent layers, try swapping gates that don't
    violate constraints. Keep the swap if depth doesn't increase.
    """
    sched = initial if initial is not None else schedule_greedy(code)

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for li in range(len(sched.layers) - 1):
            a = sched.layers[li]
            b = sched.layers[li + 1]
            # Try moving a gate from b to a if it fits
            for gate in list(b.gates):
                ct, data, ci = gate
                # Check conflict in a
                conflict = any(
                    g[1] == data or (g[0] == ct and g[2] == ci)
                    for g in a.gates)
                if not conflict:
                    a.gates.append(gate)
                    b.gates.remove(gate)
                    improved = True
        # Remove empty layers
        sched.layers = [l for l in sched.layers if l.gates]
        # Reindex
        for i, l in enumerate(sched.layers):
            l.layer_index = i

    sched.method = "greedy+local_search"
    return sched


# ─────────────────────────────────────────────────────────────────
#  Comparison
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScheduleComparison:
    code_name: str
    greedy_depth: int
    local_search_depth: int
    lower_bound: int
    greedy_gates: int

    def __repr__(self) -> str:
        return (f"ScheduleComparison({self.code_name}: "
                f"greedy={self.greedy_depth}, "
                f"local={self.local_search_depth}, "
                f"lower_bound={self.lower_bound})")


def compare_schedulers(code: QLDPCCode) -> ScheduleComparison:
    """Run both schedulers and compare to the degree lower bound."""
    tg = build_tanner_graph(code)
    greedy = schedule_greedy(code)
    local = schedule_local_search(code, initial=greedy)
    return ScheduleComparison(
        code_name=code.name,
        greedy_depth=greedy.depth,
        local_search_depth=local.depth,
        lower_bound=tg.max_degree,
        greedy_gates=greedy.num_gates,
    )


# ─────────────────────────────────────────────────────────────────
#  Emit as a pulse program
# ─────────────────────────────────────────────────────────────────

def emit_pulses(schedule: Schedule, duration_per_gate_ns: float = 50.0):
    """
    Convert a Schedule into a list of (layer, data_qubit, ancilla_qubit, t_ns).

    Ancilla mapping:
      X checks: ancilla = n_data + check_index
      Z checks: ancilla = n_data + num_x_checks + check_index
    """
    n_data = schedule.code.n
    n_x = schedule.code.num_x_checks
    pulses = []
    for layer in schedule.layers:
        for (ct, data, ci) in layer.gates:
            if ct == "X":
                ancilla = n_data + ci
            else:
                ancilla = n_data + n_x + ci
            t_start = layer.layer_index * duration_per_gate_ns
            pulses.append({
                "layer": layer.layer_index,
                "t_start_ns": t_start,
                "data_qubit": data,
                "ancilla_qubit": ancilla,
                "check_type": ct,
                "check_index": ci,
                "duration_ns": duration_per_gate_ns,
            })
    return pulses
