# OpenQHAL

A portable quantum pulse compiler and hardware abstraction layer.

OpenQHAL takes a logical quantum circuit and turns it into per-channel
I/Q samples ready for an arbitrary waveform generator. The same code
drives NV centers, superconducting transmons, trapped ions, or a
built-in simulator — swap the backend, keep the rest.

## What it is

A clean separation between three concerns that every quantum lab
reinvents badly:

1. **What the algorithm wants** — a circuit of gates.
2. **What the hardware needs** — I/Q samples with DRAG envelopes,
   per-qubit drive channels, and per-channel carrier registers.
3. **The bridge** — a frozen C ABI with pluggable backends.

Everything above the ABI is portable. Everything below is a driver.

## Status

| Layer | Component | State |
|---|---|---|
| ABI | `include/qhal/qhal.h` | Frozen, v1 |
| Core | envelopes, compiler, scheduler, waveform | Working |
| Calibration | JSON I/O, per-qubit parameters | Working |
| Simulator | statevector, up to 24 qubits | Working |
| Python | pybind11 bindings, fluent Circuit DSL | Working |
| CLI | `qhalc` | Working |
| Backend | `MockAWG` (statevector simulator) | Working |
| Backend | IBM Quantum (Qiskit Runtime) | **Verified on hardware** |
| Backend | QICK (RFSoC) | Compiles, loads, hardware-pending |
| Backend | Zurich HDAWG | Compiles, loads, hardware-pending |
| API | Python `Backend` ABC | Working |

### Hardware validation

Verified end-to-end on **IBM Quantum `ibm_kingston`** (156-qubit Heron r2),
1024 shots per circuit:

| Circuit | Qubits | P(target state) | Notes |
|---|---|---|---|
| Bell | 2 | 0.9824 | GHZ fidelity (00 + 11) |
| GHZ-3 | 3 | 0.9707 | GHZ fidelity (000 + 111) |
| GHZ-4 | 4 | 0.9561 | GHZ fidelity (0000 + 1111) |

Latency breakdown (Bell, 1024 shots):

| Phase | Time |
|---|---|
| Compile | 17 µs |
| Schedule | 7 µs |
| Render | 58 µs |
| Upload | 0.1 ms |
| Submit + queue + execute | ~10.4 s |

Over 99.99% of wall-clock time is network and queue. The HAL's
contribution is invisible at cloud timescales — which is exactly why
a portable abstraction is essentially free.

## Install

Requires Python 3.11+, a C++20 compiler, and pybind11.

```bash
python3.13 -m venv quantum
source quantum/bin/activate
pip install -r requirements.txt
./scripts/build.sh
```

This produces:

· /tmp/qhalc — the command-line compiler
· /tmp/libqhal.so — the shared C ABI library
· python/qhal/qhal_cpp*.so — the Python extension

On Debian/Ubuntu, the build also requires g++, cmake, and ninja:

```bash
apt install build-essential cmake ninja-build
```

Quick start

Python

```python
import sys
sys.path.insert(0, "python")
from qhal import Circuit, Calibration, Compiler, render, MockAWG

# 1. Build a circuit
c = Circuit(2)
c.h(0).cnot(0, 1).measure([0, 1])

# 2. Load a device calibration
cal = Calibration.from_json("calibration/nv_template.json")

# 3. Compile to pulses
prog = Compiler(cal).compile(c.gates)

# 4. Render to baseband I/Q samples
wf = render(prog, sample_rate_hz=1e9, num_channels=8)

# 5. Run on a backend (built-in simulator here)
awg = MockAWG(num_qubits=2, seed=42)
awg.set_circuit(c.gates)
awg.connect()
awg.upload(wf)
awg.trigger()
bits = awg.acquire(shot_count=100)
print(bits)   # -> [0, 0] or [1, 1]
```

IBM Quantum hardware

```python
import os
from qhal import Circuit, Calibration, Compiler, render, IBMBackend

os.environ["QISKIT_IBM_TOKEN"] = "your_44_char_token"

c = Circuit(2)
c.h(0).cnot(0, 1).measure([0, 1])

cal = Calibration.from_json("calibration/nv_template.json")
prog = Compiler(cal).compile(c.gates)
wf = render(prog, 1e9, 8)

ibm = IBMBackend(backend_name="ibm_kingston", shots=1024)
ibm.set_circuit(c.gates)

if ibm.connect():
    print(f"connected: {ibm._backend.name}")
    ibm.upload(wf)
    ibm.trigger()
    bits = ibm.acquire(1024)
    print(f"bits:   {bits}")
    print(f"counts: {ibm.counts}")
    ibm.disconnect()
```

CLI

```bash
# Sequential
echo 'H 0
CNOT 0 1
MEASURE 0 1' | /tmp/qhalc -c calibration/nv_template.json -v

# With scheduling (parallel pulse packing)
echo 'H 0
H 1
H 2
H 3
MEASURE 0 1 2 3' | /tmp/qhalc -c calibration/nv_template.json -s -v

# With simulation
echo 'H 0
MEASURE 0' | /tmp/qhalc --simulate
```

The ABI

include/qhal/qhal.h is the contract. Once frozen, it does not change.
The C++ internals, the Python bindings, the CLI, and every backend sit
on top of it. Adding a new backend does not touch the ABI.

Key types:

· qhal_gate_t — logical gate (name, qubits, params)
· qhal_pulse_t — physical pulse (channel, time, envelope, amplitude)
· qhal_program_t — a compiled program (list of pulses)
· qhal_device_t — device descriptor (name, qubits, calibration JSON)

Key functions:

· qhal_compile — gates → pulse program
· qhal_execute — run a program on the registered backend
· qhal_save_program, qhal_load_program — serialization

The pipeline

```
             ┌─────────────────────────────────────────────────────────┐
             │  Application (Python, C++, anything with a C FFI)       │
             └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │  Circuit DSL                                            │
             │    Gate("H", {0}); Gate("CNOT", {0, 1}); ...            │
             └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │  Compiler                                                │
             │    gates → pulses (DRAG, Gaussian, sech, ...)           │
             └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │  Scheduler                                               │
             │    parallel packing, channel arbitration, dependencies  │
             └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │  Waveform renderer                                       │
             │    pulses → baseband I/Q samples, per-qubit channels    │
             └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │  Backend                                                 │
             │    MockAWG (sim)   IBM Quantum (QASM/Runtime)           │
             │    QICK (RFSoC)    Zurich HDAWG                          │
             │    any Python subclass of qhal.Backend                   │
             └─────────────────────────────────────────────────────────┘
```

Directory layout

```
openqhal/
├── include/qhal/           # C ABI and C++ headers
├── src/core/               # envelopes, compiler, scheduler, waveform
├── src/backends/           # simulator, mock AWG, QICK, Zurich
├── src/bindings/           # C ABI impl, pybind11
├── src/cli/                # qhalc
├── python/qhal/            # Python frontend
│   └── backends/           # pure-Python backends (IBM, ...)
├── calibration/            # device calibration JSON templates
├── examples/               # worked examples
├── tests/                  # C++ and Python tests
├── docs/                   # architecture notes
└── scripts/build.sh        # one-shot build
```

Calibration format

```json
{
  "version": 1,
  "device_name": "nv-sample-01",
  "cnot_duration_ns": 200.0,
  "measure_duration_ns": 300.0,
  "qubits": {
    "0": {
      "rabi_hz": 10000000.0,
      "drive_freq_hz": 2870000000.0,
      "t1_ns": 1000000.0,
      "t2_ns": 100000.0,
      "drag_beta": 0.5,
      "sigma_frac": 0.2,
      "readout_duration_ns": 300.0,
      "virtual_z": true
    }
  }
}
```

Every field has a sensible default. A minimal calibration file works.

Adding a backend

Subclass qhal.Backend in Python. No C++ recompilation needed:

```python
from qhal import Backend, ExecutionStats

class MyBackend(Backend):
    def __init__(self):
        super().__init__()
        self._connected = False
        self._stats = ExecutionStats()

    def name(self):         return "MyBackend"
    def connect(self):      self._connected = True; return True
    def disconnect(self):   self._connected = False
    def is_connected(self): return self._connected
    def upload(self, wf):   return True
    def trigger(self):      return True
    def acquire(self, n):   return [0, 0]
    def last_stats(self):   return self._stats
```

Why this exists

Every quantum lab builds its own version of this, badly. Pulses are
hardcoded, channel mapping is ad-hoc, calibration lives in a notebook,
and reproducibility across devices is a nightmare. OpenQHAL is the
layer that shouldn't have to be rebuilt.

The physics is not new. The software infrastructure is.

Roadmap

☑ Phase 0 — frozen C ABI, build system
☑ Phase 1 — pulse compiler and scheduler
☑ Phase 2 — statevector simulator
☑ Phase 3 — JSON calibration
☑ Phase 4 — Python bindings
☑ Phase 5 — waveform generation (baseband, per-qubit)
☑ Phase 6 — MockAWG backend
☑ Phase 7 — README, docs
☑ Phase 8 — IBM Quantum backend, verified on hardware
☑ Phase 9 — QICK and Zurich drivers (compile-verified)
☑ Phase 10 — Python Backend ABC
☐ Phase 11 — QICK or Zurich on real lab hardware
☐ Phase 12 — cross-platform latency benchmark paper

Requirements

· Python 3.11+
· C++20 compiler (g++ 11+, clang 14+)
· pybind11 2.12+
· numpy, scipy
· qiskit, qiskit-ibm-runtime (for the IBM backend)
· qutip (optional, for accurate pulse simulation)

License

Apache 2.0. See LICENSE.

Citation

If you use OpenQHAL in academic work, cite the repository:

```
@misc{openqhal,
  title  = {OpenQHAL: A Portable Quantum Pulse Compiler and HAL},
  author = {11-qlab},
  year   = {2026},
  url    = {https://github.com/11-qlab/openqhal}
}
```

