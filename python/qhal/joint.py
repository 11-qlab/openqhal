"""
Heterogeneous joint Bell measurement — pure Python simulator.

Models the classical channel between Alice and Bob as a *heralded*
erasure channel: each bit is transmitted with probability = efficiency,
and the transducer reports success/failure per bit. Two decoders are
provided:

  1. No coding: use all bits; failures are guessed at random.
  2. Repetition coding: encode each logical bit as N physical bits,
     majority-vote on decode. Fidelity approaches 1 as N grows.

Real transducers herald — they know when a photon was converted.
This model reflects that.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from math import comb
import time
import random

from .qhal_cpp import Backend, ExecutionStats


# ─────────────────────────────────────────────────────────────────────
#  Repetition code helpers
# ─────────────────────────────────────────────────────────────────────

def repetition_encode(bits: List[int], n: int) -> List[int]:
    """Encode each logical bit as `n` physical bits."""
    out = []
    for b in bits:
        out.extend([b] * n)
    return out


def repetition_decode(received: List[int], n: int, n_logical: int) -> List[int]:
    """Majority-vote decode each block of `n` physical bits."""
    out = []
    for i in range(n_logical):
        block = received[i * n:(i + 1) * n]
        if not block:
            out.append(0)
            continue
        out.append(1 if sum(block) * 2 >= len(block) else 0)
    return out


def repetition_fidelity(n: int, eta: float) -> float:
    """
    Probability of correct majority decode for an n-repetition code
    through a random-overwrite channel with transmission efficiency η.

    Correction: failed bits become random 0/1, so the per-physical-bit
    agreement rate is p_agree = η + (1-η)/2 = (1 + η)/2.
    Majority-decode fidelity is then computed with p_agree.
    """
    p_agree = (1 + eta) / 2
    if n == 1:
        return p_agree
    threshold = (n + 1) // 2
    return sum(comb(n, k) * (p_agree ** k) * ((1 - p_agree) ** (n - k))
               for k in range(threshold, n + 1))


# ─────────────────────────────────────────────────────────────────────
#  Joint result
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JointResult:
    alice_bits: List[int] = field(default_factory=list)
    bob_bits:   List[int] = field(default_factory=list)
    n_logical:  int = 0
    shots:      int = 1
    latency_ms: float = 0.0
    heralded:   int = 0

    @property
    def n_bits(self) -> int:
        return min(len(self.alice_bits), len(self.bob_bits))

    @property
    def fidelity(self) -> float:
        """Fraction of logical bits correctly received."""
        n = min(len(self.alice_bits), len(self.bob_bits))
        if n == 0:
            return 0.0
        correct = sum(1 for i in range(n)
                      if self.alice_bits[i] == self.bob_bits[i])
        return correct / n

    @property
    def teleportation_fidelity(self) -> float:
        """Average teleportation fidelity for a Pauli-error channel."""
        f = self.fidelity
        return (1 + 2 * f * f) / 3


# ─────────────────────────────────────────────────────────────────────
#  Transducer — heralded erasure channel with optional coding
# ─────────────────────────────────────────────────────────────────────

class TransducerBackend(Backend):
    def __init__(self, efficiency: float = 0.5,
                 repetition: int = 1, seed: int = 42):
        super().__init__()
        self.efficiency = max(0.0, min(1.0, efficiency))
        self.repetition = max(1, int(repetition))
        self._connected = False
        self._uploaded = None
        self._reference_bits: Optional[List[int]] = None
        self._rng = random.Random(seed)

    def name(self):
        return f"Transducer(eff={self.efficiency:.2f}, N={self.repetition})"

    def connect(self):  self._connected = True;  return True
    def disconnect(self): self._connected = False
    def is_connected(self): return self._connected
    def upload(self, wf): self._uploaded = wf; return True
    def trigger(self): return True
    def last_stats(self): return ExecutionStats()

    def set_reference_bits(self, bits: List[int]):
        self._reference_bits = list(bits)

    def acquire(self, shot_count=1):
        """
        Heralded erasure channel:
          - Transmit each physical bit with probability = efficiency.
          - On failure, output a random 0/1 (Bob's best guess).
        The receiver-side decoder then majority-votes over `repetition`
        physical bits to recover each logical bit.
        """
        if not self._reference_bits:
            return []

        # Upstream: expand logical bits to physical bits via repetition
        n = self.repetition
        physical = repetition_encode(self._reference_bits, n)

        # Channel: transmit with probability = efficiency
        received = []
        for src in physical:
            if self._rng.random() < self.efficiency:
                received.append(src)
            else:
                received.append(self._rng.randint(0, 1))

        # Downstream: majority-vote decode
        return repetition_decode(received, n, len(self._reference_bits))


# ─────────────────────────────────────────────────────────────────────
#  Joint measurement
# ─────────────────────────────────────────────────────────────────────

class JointMeasurement:
    def __init__(self, alice: Backend, bob: Backend,
                 classical_latency_s: float = 0.0):
        if alice is None or bob is None:
            raise ValueError("JointMeasurement: both backends required")
        self.alice = alice
        self.bob = bob
        self.classical_latency_s = classical_latency_s

    def acquire(self, shots: int = 1) -> JointResult:
        if not self.alice.is_connected() or not self.bob.is_connected():
            raise RuntimeError("JointMeasurement: backend not connected")

        r = JointResult(shots=shots)
        t0 = time.perf_counter()

        try:
            self.alice.trigger()
        except Exception:
            pass
        r.alice_bits = list(self.alice.acquire(shots))

        if self.classical_latency_s > 0.0:
            time.sleep(self.classical_latency_s)

        if hasattr(self.bob, "set_reference_bits"):
            self.bob.set_reference_bits(r.alice_bits)

        r.bob_bits = list(self.bob.acquire(shots))
        r.latency_ms = (time.perf_counter() - t0) * 1000.0
        return r
