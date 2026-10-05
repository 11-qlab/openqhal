"""
HTTP API gateway for OpenQHAL.

Exposes the HAL over REST:
    GET  /                         -> service info
    GET  /backends                 -> list registered backends
    POST /compile                  -> circuit -> QASM
    POST /submit                   -> submit a job, returns job_id
    GET  /jobs                     -> list recent jobs
    GET  /jobs/{job_id}            -> poll job status

Run with:
    uvicorn qhal.api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ir import Circuit
from .queue import JobQueue, make_simulator_executor


# ─────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────

class GateIn(BaseModel):
    name: str
    qubits: List[int]
    params: List[float] = Field(default_factory=list)


class CircuitIn(BaseModel):
    num_qubits: int
    gates: List[GateIn]


class SubmitIn(BaseModel):
    circuit: CircuitIn
    backend: str = "simulator"
    shots: int = 1024


class SubmitOut(BaseModel):
    job_id: str


class JobOut(BaseModel):
    job_id: str
    backend: str
    shots: int
    status: str
    result: Optional[List[int]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class CompileIn(BaseModel):
    circuit: CircuitIn


class CompileOut(BaseModel):
    qasm: str
    depth: int
    gate_count: int
    gate_histogram: Dict[str, int]


# ─────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="OpenQHAL API", version="0.6.0")
queue = JobQueue()
queue.register_executor("simulator", make_simulator_executor())


def _circuit_from(inp: CircuitIn) -> Circuit:
    try:
        return Circuit(
            num_qubits=inp.num_qubits,
            gates=tuple(GateIn_to_gate(g) for g in inp.gates),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def GateIn_to_gate(g: GateIn):
    from .ir import Gate
    return Gate(g.name, tuple(g.qubits), tuple(g.params))


@app.get("/")
def root():
    return {
        "service": "OpenQHAL",
        "version": app.version,
        "endpoints": [
            "/backends", "/compile", "/submit",
            "/jobs", "/jobs/{job_id}",
        ],
    }


@app.get("/backends")
def list_backends():
    return {"backends": queue.list_backends()}


@app.post("/compile", response_model=CompileOut)
def compile_circuit(inp: CompileIn):
    c = _circuit_from(inp.circuit)
    return CompileOut(
        qasm=c.to_qasm(),
        depth=c.depth(),
        gate_count=len(c.gates),
        gate_histogram=c.counts(),
    )


@app.post("/submit", response_model=SubmitOut)
def submit(inp: SubmitIn):
    if inp.backend not in queue.list_backends():
        raise HTTPException(
            status_code=404,
            detail=f"backend '{inp.backend}' not registered")
    c = _circuit_from(inp.circuit)
    job_id = queue.submit(c, inp.backend, inp.shots)
    return SubmitOut(job_id=job_id)


@app.get("/jobs")
def list_jobs(limit: int = 20):
    jobs = queue.list_jobs(limit=limit)
    return {"jobs": [JobOut(**j.to_dict()).dict() for j in jobs]}


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(**job.to_dict())


# ─────────────────────────────────────────────────────────────────────
#  Lifespan: shut down the worker cleanly
# ─────────────────────────────────────────────────────────────────────

@app.on_event("shutdown")
def on_shutdown():
    queue.shutdown()
