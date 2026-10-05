"""
Background job queue for OpenQHAL.

Submit circuits, get a job ID, poll for results. Executes in a
worker thread so submissions don't block.

Thread-safe. In-memory only (v0.1). Persistence is a future task.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime
import threading
import queue as _queue
import uuid

from .ir import Circuit


# ─────────────────────────────────────────────────────────────────────
#  Job
# ─────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    circuit: Circuit
    backend_name: str
    shots: int
    status: JobStatus = JobStatus.PENDING
    result: Optional[List[int]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "circuit": self.circuit.to_dict(),
            "backend": self.backend_name,
            "shots": self.shots,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ─────────────────────────────────────────────────────────────────────
#  Executor protocol
# ─────────────────────────────────────────────────────────────────────

Executor = Callable[[Circuit, int], List[int]]


class JobQueue:
    """In-memory job queue with a single background worker."""

    def __init__(self):
        # Set all attributes BEFORE starting the worker thread,
        # otherwise the thread can observe missing attributes.
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._q: "_queue.Queue[Optional[str]]" = _queue.Queue()
        self._executors: Dict[str, Executor] = {}
        self._shutdown = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── registration ──
    def register_executor(self, backend_name: str, fn: Executor) -> None:
        """Backend name -> function(circuit, shots) -> bit list."""
        with self._lock:
            self._executors[backend_name] = fn

    def list_backends(self) -> List[str]:
        with self._lock:
            return sorted(self._executors.keys())

    # ── submission ──
    def submit(self, circuit: Circuit, backend_name: str,
               shots: int = 1024) -> str:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, circuit=circuit,
                  backend_name=backend_name, shots=shots)
        with self._lock:
            self._jobs[job_id] = job
        self._q.put(job_id)
        return job_id

    # ── status ──
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in (JobStatus.PENDING,
                                                 JobStatus.RUNNING):
                return False
            job.status = JobStatus.CANCELLED
            return True

    # ── worker ──
    def _run(self) -> None:
        while not self._shutdown:
            job_id = self._q.get()
            if job_id is None:
                break
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status == JobStatus.CANCELLED:
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow().isoformat()

            try:
                with self._lock:
                    fn = self._executors.get(job.backend_name)
                if fn is None:
                    raise RuntimeError(
                        f"no executor for backend '{job.backend_name}'")
                result = fn(job.circuit, job.shots)
                with self._lock:
                    if job.status == JobStatus.CANCELLED:
                        continue
                    job.result = result
                    job.status = JobStatus.DONE
                    job.finished_at = datetime.utcnow().isoformat()
            except Exception as e:
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = f"{type(e).__name__}: {e}"
                    job.finished_at = datetime.utcnow().isoformat()

    def shutdown(self) -> None:
        self._shutdown = True
        self._q.put(None)
        self._worker.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────
#  Default executors — bridge to OpenQHAL's C++ backends
# ─────────────────────────────────────────────────────────────────────

def make_simulator_executor(seed: int = 42):
    """Return an executor that runs circuits on the local simulator."""
    from .qhal_cpp import Simulator
    def run(circuit: Circuit, shots: int) -> List[int]:
        sim = Simulator(circuit.num_qubits, seed)
        # C++ Simulator.run expects a list of Gate objects
        from .qhal_cpp import Gate as CppGate
        cpp_gates = [CppGate(g.name, list(g.qubits), list(g.params))
                     for g in circuit.gates]
        return list(sim.run(cpp_gates))
    return run


def make_mock_awg_executor(seed: int = 42):
    """Return an executor that runs circuits on MockAWG."""
    from .qhal_cpp import MockAWG, Gate as CppGate
    def run(circuit: Circuit, shots: int) -> List[int]:
        awg = MockAWG(circuit.num_qubits, seed, False)
        cpp_gates = [CppGate(g.name, list(g.qubits), list(g.params))
                     for g in circuit.gates]
        awg.set_circuit(cpp_gates)
        awg.connect()
        awg.trigger()
        return list(awg.acquire(shots))
    return run
