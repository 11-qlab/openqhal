"""
IBM Quantum backend for OpenQHAL.

Pure Python. Subclasses the qhal.Backend ABC so it plugs into the same
pipeline as MockAWG, QICKBackend, and ZurichBackend — no C++ involved.

Emits OpenQASM 3 from the gate list, submits via Qiskit Runtime, and
returns the most-likely measurement bitstring.
"""
from __future__ import annotations
import os
from typing import List

from ..qhal_cpp import (
    Backend, ExecutionStats, Waveform, Gate,
)


class IBMBackend(Backend):
    """IBM Quantum hardware backend via Qiskit Runtime."""

    def __init__(
        self,
        channel: str = "ibm_quantum_platform",
        backend_name: str = "",
        token_env: str = "QISKIT_IBM_TOKEN",
        optimization: int = 3,
        shots: int = 1024,
    ):
        super().__init__()
        self.channel       = channel
        self.backend_name  = backend_name
        self.token_env     = token_env
        self.optimization  = optimization
        self.shots         = shots

        self._connected = False
        self._gates: List[Gate] = []
        self._qasm: str = ""
        self._service = None
        self._backend = None
        self._job = None
        self._counts = None
        self._stats = ExecutionStats()

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "IBMQuantum"

    def connect(self) -> bool:
        token = os.environ.get(self.token_env, "").strip()
        if not token:
            return False
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(
                channel=self.channel, token=token)
            if self.backend_name:
                self._backend = self._service.backend(self.backend_name)
            else:
                self._backend = self._service.least_busy(
                    operational=True, simulator=False)
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._service = None
        self._backend = None
        self._job = None
        self._qasm = ""

    def is_connected(self) -> bool:
        return self._connected

    def set_circuit(self, gates: List[Gate]) -> None:
        """Attach the logical circuit. Required before upload."""
        self._gates = list(gates)

    def upload(self, wf: Waveform) -> bool:
        if not self._connected:
            return False
        try:
            self._qasm = self._emit_qasm()
            return True
        except Exception:
            return False

    def trigger(self) -> bool:
        if not self._connected or not self._qasm:
            return False
        try:
            from qiskit import qasm3, transpile
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            circuit = qasm3.loads(self._qasm)
            isa_circuit = transpile(
                circuit,
                backend=self._backend,
                optimization_level=self.optimization,
            )
            sampler = Sampler(mode=self._backend)
            self._job = sampler.run([isa_circuit], shots=self.shots)
            return True
        except Exception as e:
            import traceback
            print("--- trigger failed ---", file=__import__('sys').stderr)
            traceback.print_exc()
            print("--- end ---", file=__import__('sys').stderr)
            return False

    def acquire(self, shot_count: int = 1024) -> List[int]:
        if self._job is None:
            return []
        try:
            result = self._job.result()
            self._counts = result[0].data.meas.get_counts()
            # Most likely bitstring
            best = max(self._counts.items(), key=lambda kv: kv[1])[0]
            # Qiskit counts are MSB-first; reverse to qubit-0-first
            return [int(b) for b in best][::-1]
        except Exception:
            return []

    def last_stats(self) -> ExecutionStats:
        return self._stats

    # ------------------------------------------------------------------
    # QASM emission
    # ------------------------------------------------------------------
    def _emit_qasm(self) -> str:
        nq = 1
        for g in self._gates:
            for q in g.qubits:
                if q + 1 > nq:
                    nq = q + 1

        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{nq}] q;",
            f"bit[{nq}] meas;",
        ]
        for g in self._gates:
            n = g.name
            if n in ("MEASURE", "M"):
                for q in g.qubits:
                    lines.append(f"meas[{q}] = measure q[{q}];")
            elif n in ("H", "X", "Y", "Z", "S", "T"):
                lines.append(f"{n.lower()} q[{g.qubits[0]}];")
            elif n in ("RX", "RY", "RZ"):
                theta = g.params[0] if g.params else 0.0
                lines.append(f"{n.lower()}({theta}) q[{g.qubits[0]}];")
            elif n in ("CNOT", "CX"):
                lines.append(f"cx q[{g.qubits[0]}], q[{g.qubits[1]}];")
            elif n == "CZ":
                lines.append(f"cz q[{g.qubits[0]}], q[{g.qubits[1]}];")
            elif n == "SWAP":
                lines.append(f"swap q[{g.qubits[0]}], q[{g.qubits[1]}];")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def qasm(self) -> str:
        return self._qasm

    @property
    def counts(self):
        return self._counts


def available_backends(channel: str = "ibm_quantum_platform",
                       token_env: str = "QISKIT_IBM_TOKEN"):
    """List operational IBM QPUs visible to this account."""
    token = os.environ.get(token_env, "").strip()
    if not token:
        return []
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel=channel, token=token)
    return service.backends(operational=True, simulator=False)
