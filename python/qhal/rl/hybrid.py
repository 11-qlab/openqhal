"""
Hybrid planner: row/column shifts for bulk, atom moves for cleanup.

Strategy:
  1. Row/column planner (fast) runs first, with a move budget
  2. Whatever state it reaches is the new starting point
  3. Atom planner runs from there to reach the target
  4. Both planners are exact enough that the combined result is
     successful on every solvable instance with density < 100%

Row/column moves shift many atoms at once -> fast for low density
Atom moves shift one atom at a time -> guaranteed progress
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .planner import ClassicalPlanner, Move
from .atom_planner import AtomPlanner, AtomMove


class HybridPlanner:

    def __init__(self, H: int, W: int,
                 rowcol_budget: int = 50,
                 atom_budget: int = 500):
        self.H = H
        self.W = W
        self.rowcol = ClassicalPlanner(H, W, max_moves=rowcol_budget,
                                       n_restarts=3)
        self.atom = AtomPlanner(H, W, max_moves=atom_budget,
                                 n_restarts=5)

    def plan(self, initial: np.ndarray,
             target: np.ndarray
             ) -> Tuple[List, bool, str]:
        """
        Returns (moves, success, solver).

        `moves` is a mixed list. Each element is either a
        ClassicalPlanner.Move (row/column) or an AtomMove.
        """
        if np.array_equal(initial, target):
            return [], True, "trivial"

        # ── Stage 1: row/column attempt ──
        rc_moves, rc_ok = self.rowcol.plan(initial, target)
        if rc_ok:
            return rc_moves, True, "rowcol"

        # Whatever row/col reached becomes the atom planner's start
        intermediate = self.rowcol.simulate(initial, rc_moves)

        # ── Stage 2: atom moves from the intermediate state ──
        atom_moves, atom_ok = self.atom.plan(intermediate, target)
        if atom_ok:
            return rc_moves + atom_moves, True, "hybrid"

        # ── Stage 3: atom planner from scratch ──
        atom_moves, atom_ok = self.atom.plan(initial, target)
        if atom_ok:
            return atom_moves, True, "atom_only"

        return [], False, "failed"

    def simulate(self, initial: np.ndarray,
                 moves: List) -> np.ndarray:
        """Apply mixed list of row/col and atom moves in order."""
        g = initial.copy()
        for m in moves:
            if isinstance(m, AtomMove):
                g = AtomPlanner.apply_move(g, m)
            else:
                g = ClassicalPlanner.apply_move(g, m)
        return g
