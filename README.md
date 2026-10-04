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
| Backend | `MockAWG` (simulated device) | Working |
| Backend | Zurich HDAWG, Spectrum M4i, QICK | Not yet |

## Install

Requires Python 3.11+, a C++20 compiler, and pybind11.

```bash
python3.13 -m venv quantum
source quantum/bin/activate
pip install -r requirements.txt
./scripts/build.sh





## File: `docs/architecture.md`

```bash
cat > docs/architecture.md << 'EOF'
# Architecture

## The three-layer model

OpenQHAL separates three concerns that are usually tangled together:

1. **Circuit** — what the algorithm wants. Gates, qubits, measurements.
2. **Pulse** — what the hardware needs. Timings, envelopes, amplitudes.
3. **Sample** — what the DAC actually plays. I/Q values at 1 GS/s.

Every layer has a stable representation, and every transition between
layers is a pure function. No global state. No hidden channels.

## The compiler

`Compiler::compile(gates) → Program`

Reads a calibration file to get per-qubit parameters, then emits pulses.
Gate set:

| Gate | Implementation | Duration |
|---|---|---|
| I | no-op | 0 |
| X | π rotation about X | 1/(2·f_Rabi) |
| Y | π rotation about Y | 1/(2·f_Rabi) |
| Z | virtual frame rotation | 0 |
| H | π/2 about Y, then Z(π) | 1/(4·f_Rabi) |
| S | virtual Z(π/2) | 0 |
| T | virtual Z(π/4) | 0 |
| RX(θ) | θ rotation about X | θ/(2π·f_Rabi) |
| RY(θ) | θ rotation about Y | θ/(2π·f_Rabi) |
| RZ(θ) | virtual Z(θ) | 0 |
| CNOT | π on control, π on target | platform-dependent |
| MEASURE | laser pulse | readout_duration_ns |

Virtual Z gates are free. They update the frame phase in the pulse
sequence, so subsequent pulses get an extra phase offset. This is a
standard technique in superconducting and NV hardware.

## The scheduler

`Scheduler::schedule_program(prog) → ScheduleResult`

Three rules for dependency:

1. Pulses from the same logical gate must serialize (intra-gate order).
2. Pulses sharing a qubit must serialize (cross-gate qubit order).
3. Global pulses (qubit = -1, e.g. readout laser) serialize with all.

Pulses on different qubits with different channels run in parallel.

The scheduler is greedy ASAP: each pulse starts as soon as its
dependencies finish. Optimal for small circuits; NP-hard in general.

## The waveform renderer

`render(prog, sample_rate_hz, num_channels, baseband) → Waveform`

For each pulse:

1. Compute the envelope samples (Gaussian, DRAG, sech, ...).
2. If baseband, apply only the phase offset. The carrier is a hardware
   register (DUC) that multiplies after the DAC.
3. If not baseband, multiply by `exp(i·2π·f·t)` explicitly. Only valid
   when sample_rate_hz > 2·frequency_hz.
4. Add to the channel buffer at the correct offset.

Baseband is the default and the only correct choice for real hardware.
The pulse's carrier frequency is metadata — it's what the DUC register
gets set to, not what the samples contain.

## The backend interface

```cpp
class Backend {
    virtual bool connect() = 0;
    virtual bool upload(const Waveform&) = 0;
    virtual bool trigger() = 0;
    virtual std::vector<int32_t> acquire(int32_t shots) = 0;
    virtual void disconnect() = 0;
};






## File: `.gitignore`

```bash
cat > .gitignore << 'EOF'
# Build artifacts
build/
*.o
*.so
*.dylib
*.dll
*.a

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
quantum/
ENVS/

# Editor
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Test outputs
/tmp/*.npy
/tmp/*.csv
/tmp/*.seqc
/tmp/*.bin
EOF


