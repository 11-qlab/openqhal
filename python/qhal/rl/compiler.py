"""
RL-based atom rearrangement compiler, integrated with OpenQHAL.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch

from .env import AtomArrangementEnv
from .policy import PolicyValueNet
from .policy_transformer import TransformerAdapter


@dataclass
class Move:
    axis: int          # 0 = row, 1 = column
    index: int
    direction: int     # -1 or +1
    magnitude: int     # 1 or 2

    def __repr__(self):
        ax = "row" if self.axis == 0 else "col"
        d = "-" if self.direction < 0 else "+"
        return f"Move({ax} {self.index} {d}{self.magnitude})"

    def to_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "index": self.index,
                "direction": self.direction, "magnitude": self.magnitude}


class RLCompiler:
    """RL-based compiler for atom rearrangement."""

    def __init__(self, model_path: Optional[str] = None,
                 grid_size: int = 3,
                 n_atoms: int = 3,
                 use_transformer: bool = False):
        self.grid_size = grid_size
        self.n_atoms = n_atoms
        self.use_transformer = use_transformer

        self.env = AtomArrangementEnv(grid_size=grid_size,
                                       n_atoms=n_atoms)

        if use_transformer:
            self.model = TransformerAdapter(
                H=grid_size, W=grid_size, max_grid_size=8,
                hidden=64, n_layers=2, n_heads=4)
        else:
            self.model = PolicyValueNet(self.env.state_dim,
                                         self.env.n_actions, hidden=64)

        if model_path:
            ckpt = torch.load(model_path, map_location="cpu",
                              weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            try:
                self.model.load_state_dict(state)
            except Exception as e:
                print(f"warn: could not load weights ({e}); using fresh model")
            self.model.eval()

    def compile(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        max_steps: int = 10,
        greedy: bool = True,
    ) -> Tuple[List[Move], bool]:
        self.env.grid = torch.tensor(initial, dtype=torch.int8)
        self.env.target = torch.tensor(target, dtype=torch.int8)
        self.env.steps = 0
        self.env.prev_dist = self.env._dist()

        moves: List[Move] = []
        state = self.env._obs()
        done = False

        while not done and self.env.steps < max_steps:
            with torch.no_grad():
                logits, _ = self.model(state)
            if greedy:
                action = int(logits.argmax().item())
            else:
                from torch.distributions import Categorical
                action = int(Categorical(logits=logits).sample().item())

            axis, index, direction, magnitude = self.env.decode_action(action)
            moves.append(Move(axis, index, direction, magnitude))

            result = self.env.step(action)
            state = result.state
            done = result.done

        success = (self.env._dist() == 0)
        return moves, success


# ─────────────────────────────────────────────────────────────────
#  OpenQHAL Schedule integration
# ─────────────────────────────────────────────────────────────────

def moves_to_program(moves: List[Move],
                     n_atoms: int,
                     duration_per_move_ns: float = 200.0):
    """
    Convert AOD moves into an OpenQHAL Program.

    Each move becomes one pulse per affected atom, on a dedicated
    "atom transport" channel. The Program can be passed to any
    Backend (mock, IBM, QICK, Zurich) for execution.
    """
    from ..qhal_cpp import Pulse, Program, Channel, Envelope

    prog = Program()
    prog.total_duration_ns = len(moves) * duration_per_move_ns
    prog.shot_count = 1

    for i, m in enumerate(moves):
        # One pulse per atom — encode axis/index as channel metadata
        # In a real neutral-atom compiler, this maps to an AOD waveform.
        ch_idx = (m.axis << 4) | (m.index & 0xF)
        p = Pulse()
        p.channel        = ch_idx % 8   # wrap to available channels
        p.start_ns       = i * duration_per_move_ns
        p.duration_ns    = duration_per_move_ns
        p.frequency_hz   = 0.0
        p.amplitude      = float(m.magnitude) / 2.0
        p.phase_rad      = 0.0 if m.direction < 0 else 3.141592653589793
        p.envelope       = Envelope.Square
        p.drag_beta      = 0.0
        p.envelope_param = 0.2
        p.qubit          = -1     # global op
        prog.pulses.append(p)

    return prog


def moves_to_schedule(moves: List[Move], code=None):
    """
    Wrap moves in a qLDPC Schedule-like object for pipeline compatibility.
    """
    try:
        from ..qldpc_scheduler import Schedule, ScheduleLayer
    except ImportError:
        return None

    if code is None:
        from ..qldpc import repetition
        code = repetition(max(3, len(moves) + 1))

    sched = Schedule(code=code, method="rl")
    for i, m in enumerate(moves):
        layer = ScheduleLayer(layer_index=i)
        layer.gates.append(("AOD", m.index, i))   # encoded as pseudo-gate
        sched.layers.append(layer)
    return sched
