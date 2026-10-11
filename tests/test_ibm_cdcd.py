"""
End-to-end CDCD test on IBM hardware.

Pipeline:
  1. Build a repetition[[5]] code
  2. Schedule syndrome extraction
  3. Emit Qiskit QuantumCircuit
  4. Transpile + convert mid-circuit resets
  5. Run on simulator or IBM hardware
  6. Decode with BP+OSD
"""
import sys, os
sys.path.insert(0, "python")

import numpy as np

from qhal.qldpc import repetition
from qhal.qldpc_scheduler import schedule_greedy
from qhal.backends.ibm_syndrome import (
    SyndromeCircuit, SyndromeConfig, extract_detector_events,
)
from qhal.decoder import build_dem_from_code, decode_with_bposd


def run_local_simulator(qc, shots: int) -> np.ndarray:
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    shots_list = []
    for k in sorted(counts.keys()):
        # Aer formats multiple registers with spaces. We just want a flat MSB-first array.
        flat_k = k.replace(" ", "")
        bits = [int(b) for b in reversed(flat_k)]
        shots_list.extend([bits] * counts[k])
    return np.array(shots_list, dtype=np.uint8)


def run_ibm_hardware(qc, shots: int, backend_name: str, num_rounds: int) -> np.ndarray:
    """Run a Qiskit circuit on real IBM hardware.

    Requires ConvertToMidCircuitResetAndMeasure after transpilation.
    """
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.transpiler.passes.basis import (
        ConvertToMidCircuitResetAndMeasure,
    )

    token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set")

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                    token=token)
    backend = service.backend(backend_name)

    # FIX: Optimization level 1 is mandatory for mid-circuit dynamic circuits.
    # Level 3 illegally reorders measures/resets causing Error 6056.
    isa = transpile(qc, backend=backend, optimization_level=1)
    n_reset_before = sum(1 for i in isa.data if i.operation.name == "reset")
    print(f"    after transpile: {isa.num_qubits} qubits, depth {isa.depth()}")
    print(f"    reset ops: {n_reset_before}")

    if n_reset_before > 0:
        pm = PassManager([ConvertToMidCircuitResetAndMeasure(target=backend.target)])
        isa = pm.run(isa)
        n_reset = sum(1 for i in isa.data if i.operation.name == "reset")
        n_reset2 = sum(1 for i in isa.data if i.operation.name == "reset_2")
        print(f"    after conversion: reset={n_reset}, reset_2={n_reset2}")
    else:
        print("    no reset ops — skipping MidCircuitReset pass")

    sampler = Sampler(mode=backend)
    job = sampler.run([isa], shots=shots)
    print(f"    job id: {job.job_id()}")
    result = job.result()

    data = result[0].data
    bitstrings = data.syn.get_bitstrings()

    # The single ClassicalRegister yields a flat bitstring. 
    # Qiskit outputs MSB-first (cr[N] ... cr[0]), so reversing it 
    # perfectly aligns the indices to [cr[0], cr[1], ..., cr[N]].
    shots_list = [[int(b) for b in reversed(bs)] for bs in bitstrings]
    
    return np.array(shots_list, dtype=np.uint8)


def main():
    print("=" * 68)
    print("  CDCD end-to-end: syndrome extraction -> decode")
    print("=" * 68)

    code = repetition(5)
    print(f"\n[1] Code: {code.name}")
    print(f"    n={code.n}, X-checks={code.num_x_checks}, "
          f"Z-checks={code.num_z_checks}")

    sched = schedule_greedy(code)
    print(f"\n[2] Schedule: depth={sched.depth}, gates={sched.num_gates}")

    cfg = SyndromeConfig(num_rounds=2)
    sc = SyndromeCircuit(sched, cfg)
    summary = sc.summary()
    print(f"\n[3] Circuit summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    shots = 500
    use_hardware = os.environ.get("QHAL_USE_IBM", "").strip() == "1"
    backend_name = os.environ.get("QHAL_IBM_BACKEND", "ibm_kingston")

    qc = sc.to_qiskit(use_reset=True)
    print(f"    Qiskit circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    print(f"    use_reset: True")

    if use_hardware:
        print(f"\n[4] Running on IBM hardware: {backend_name}")
        raw = run_ibm_hardware(qc, shots, backend_name, cfg.num_rounds)
    else:
        print(f"\n[4] Running on local simulator (Aer)")
        raw = run_local_simulator(qc, shots)

    print(f"    shots: {raw.shape[0]}, bits per shot: {raw.shape[1]}")
    print(f"    mean syndrome weight: {raw.mean():.4f}")

    events = extract_detector_events(raw,
                                      summary["n_x_checks"],
                                      summary["n_z_checks"],
                                      cfg.num_rounds)
    print(f"\n[5] Detector events shape: {events.shape}")
    print(f"    event rate: {events.mean():.4f}")

    dem = build_dem_from_code(code, per_check_error=0.01)
    print(f"\n[6] DEM: {dem}")
    for p in [0.005, 0.010, 0.020]:
        r = decode_with_bposd(dem, physical_error_rate=p, shots=300)
        print(f"    p={p:.3f}  predicted LER={r.logical_error_rate:.6f}")

    print(f"\n[7] Comparison")
    print(f"    Observed syndrome weight: {raw.mean():.4f}")
    print(f"    Observed detector rate:   {events.mean():.4f}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
