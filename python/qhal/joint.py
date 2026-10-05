"""
Heterogeneous joint Bell measurement — pure Python simulator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time
import random

from .qhal_cpp import Backend, ExecutionStats


@dataclass
class JointResult:
    alice_bits: List[int] = field(default_factory=list)
    bob_bits:   List[int] = field(default_factory=list)
    parity:     int = 0
    correlation_tag: str = ""
    shots:      int = 1
    latency_ms: float = 0.0


class TransducerBackend(Backend):
    """
    Models a quantum transducer (optical <-> microwave) as a classical
    channel with finite efficiency.

    Correct model:
        bob_bit = alice_bit        with probability = efficiency
        bob_bit = random(0,1)      with probability = 1 - efficiency

    The transducer does NOT generate bits. It transmits them. The
    caller must supply the source bits via `set_reference_bits()`.
    """

    def __init__(self, efficiency: float = 0.5, seed: int = 42):
        super().__init__()
        self.efficiency = max(0.0, min(1.0, efficiency))
        self._connected = False
        self._uploaded = None
        self._reference_bits: Optional[List[int]] = None
        self._rng = random.Random(seed)

    def name(self):
        return f"Transducer(eff={self.efficiency:.2f})"

    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def upload(self, wf):
        self._uploaded = wf
        return True

    def trigger(self):
        return True

    def set_reference_bits(self, bits: List[int]):
        self._reference_bits = list(bits)

    def acquire(self, shot_count=1):
        if self._reference_bits is None:
            # No source: return random (should not happen in normal flow)
            return [self._rng.randint(0, 1) for _ in range(shot_count)]

        out = []
        for i in range(shot_count):
            src = self._reference_bits[i % len(self._reference_bits)]
            if self._rng.random() < self.efficiency:
                out.append(src)                     # transmitted
            else:
                out.append(self._rng.randint(0, 1)) # corrupted
        return out

    def last_stats(self):
        return ExecutionStats()


class JointMeasurement:
    """Compose two Backends into a single joint Bell measurement."""

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

        # 1. Alice measures
        r.alice_bits = list(self.alice.acquire(shots))

        # 2. Classical channel delay
        if self.classical_latency_s > 0.0:
            time.sleep(self.classical_latency_s)

        # 3. If Bob's backend is a transducer, feed it Alice's bits.
        #    In a real experiment, the Bell partner's state is what
        #    drives Bob's qubit — the transducer carries that.
        if hasattr(self.bob, "set_reference_bits"):
            self.bob.set_reference_bits(r.alice_bits)

        # 4. Bob measures
        r.bob_bits = list(self.bob.acquire(shots))

        # 5. Joint parity
        p = 0
        for b in r.alice_bits: p ^= (b & 1)
        for b in r.bob_bits:   p ^= (b & 1)
        r.parity = p
        r.correlation_tag = f"bell_{p}"

        r.latency_ms = (time.perf_counter() - t0) * 1000.0
        return r
