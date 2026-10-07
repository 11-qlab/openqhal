"""
OpenQHAL — Quantum Hardware Abstraction Layer
Python frontend for the C++ core.

The C++ extension (qhal_cpp) is optional for pure-Python modules
(RL env, gym wrapper, IR, transpiler, queue). It is required for
compilation, waveform rendering, and backend dispatch.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
#  C++ core (optional)
# ─────────────────────────────────────────────────────────────────
QHAL_CPP_AVAILABLE = False
try:
    from .qhal_cpp import (
        Channel, Envelope, Gate, Pulse, Program,
        QubitCalibration, Calibration,
        Compiler, Scheduler, ScheduleResult,
        Simulator, Waveform, render,
        save_npy, save_raw, save_csv, save_seqc,
        total_energy,
        Backend, MockAWG, ExecutionStats,
    )
    QHAL_CPP_AVAILABLE = True

    try:
        from .qhal_cpp import QICKBackend, ZurichBackend
    except ImportError:
        QICKBackend = None
        ZurichBackend = None

    __all__ = [
        "Channel", "Envelope",
        "Gate", "Pulse", "Program",
        "QubitCalibration", "Calibration",
        "Compiler", "Scheduler", "ScheduleResult",
        "Simulator",
        "Waveform", "render",
        "save_npy", "save_raw", "save_csv", "save_seqc",
        "total_energy",
        "Backend", "MockAWG", "ExecutionStats",
        "QICKBackend", "ZurichBackend",
    ]
except ImportError:
    # C++ extension not built. Pure-Python modules below still load.
    pass

# ─────────────────────────────────────────────────────────────────
#  Pure-Python modules (always available)
# ─────────────────────────────────────────────────────────────────

# IBM backend (pure Python)
try:
    from .backends.ibm import IBMBackend, available_backends
    __all__ = list(globals().get("__all__", []))
    __all__ += ["IBMBackend", "available_backends"]
except ImportError:
    pass

# Circuit IR + transpiler
try:
    from .ir import Circuit as IRCircuit, Gate as IRGate, bell_pair, ghz
    from .transpiler import Transpiler, CouplingMap, TranspileResult
    from .queue import JobQueue, Job, JobStatus, make_simulator_executor
    __all__ = list(globals().get("__all__", []))
    __all__ += [
        "IRCircuit", "IRGate", "bell_pair", "ghz",
        "Transpiler", "CouplingMap", "TranspileResult",
        "JobQueue", "Job", "JobStatus", "make_simulator_executor",
    ]
except ImportError:
    pass

# qLDPC + CDCD
try:
    from .qldpc import QLDPCCode, bb_code, gross_code, small_bb, repetition
    from .cdcd import (
        CheckAnnotation, AnnotatedProgram,
        annotate_program, build_check_noise, build_detector_error_model,
    )
    __all__ = list(globals().get("__all__", []))
    __all__ += [
        "QLDPCCode", "bb_code", "gross_code", "small_bb", "repetition",
        "CheckAnnotation", "AnnotatedProgram",
        "annotate_program", "build_check_noise", "build_detector_error_model",
    ]
except ImportError:
    pass

# Decoder + qLDPC scheduler
try:
    from .decoder import (
        DetectorErrorModel, DecodeResult,
        build_dem_from_code, decode_with_bposd, co_design_benchmark,
    )
    from .qldpc_scheduler import (
        TannerGraph, Schedule, ScheduleLayer, ScheduleComparison,
        build_tanner_graph, schedule_greedy, schedule_local_search,
        compare_schedulers, emit_pulses,
    )
    __all__ = list(globals().get("__all__", []))
    __all__ += [
        "DetectorErrorModel", "DecodeResult",
        "build_dem_from_code", "decode_with_bposd", "co_design_benchmark",
        "TannerGraph", "Schedule", "ScheduleLayer", "ScheduleComparison",
        "build_tanner_graph", "schedule_greedy", "schedule_local_search",
        "compare_schedulers", "emit_pulses",
    ]
except ImportError:
    pass

# RL (pure Python + torch)
try:
    from .rl.env import AtomArrangementEnv, StepResult
    from .rl.policy import PolicyValueNet, RolloutBuffer
    from .rl.ppo import PPOTrainer, PPOConfig
    __all__ = list(globals().get("__all__", []))
    __all__ += [
        "AtomArrangementEnv", "StepResult",
        "PolicyValueNet", "RolloutBuffer",
        "PPOTrainer", "PPOConfig",
    ]
except ImportError:
    pass

try:
    from .rl.gym_wrapper import ShapedAtomGym, ShapedAtomEnv
    from .rl.features import AtomGridCNN
    __all__ = list(globals().get("__all__", []))
    __all__ += ["ShapedAtomGym", "ShapedAtomEnv", "AtomGridCNN"]
except ImportError:
    pass

__version__ = "0.11.0"
