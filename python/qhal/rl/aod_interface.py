"""
Move → AOD pulse sequence interface specification.

This module defines the contract between the RL model's output
(abstract Move objects) and a physical Acousto-Optic Deflector
(AOD) controller.

Real hardware vendors (QuEra, Pasqal, Infleqtion, university labs)
do not expose AOD control through a standard API. This module
provides the missing specification and a reference implementation
that a lab can adapt to its own controller.

The contract:

    Lab implements:  AODDriver interface (5 methods)
    Model provides:  List[Move] from qhal.rl.infer
    This module:     Converts one to the other

Physical units:
    - Positions:     micrometers (µm)
    - Durations:     microseconds (µs)
    - Slew rates:    µm/µs
    - Frequencies:   MHz

Coordinate system:
    - Origin (0,0) at the top-left of the tweezer array
    - Rows increase downward (r), columns increase rightward (c)
    - Lattice spacing = distance between adjacent tweezer sites

References:
    - QuEra Bloqade AOD API: https://queracomputing.github.io/bloqade/
    - Pasqal Pulser AOD: https://pulser.readthedocs.io/
    - Endres group AOD control (Caltech), arXiv:2205.08520
"""
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  Physical configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class AODConfig:
    """Hardware-specific AOD parameters. Provided by the lab."""
    lattice_spacing_um: float = 4.0
    """Distance between adjacent tweezer sites in micrometers."""

    slew_rate_um_per_us: float = 1.0
    """AOD acoustic velocity: how fast the beam can shift position."""

    n_channels: int = 2
    """Number of independent AOD axes (2 = row + column)."""

    max_deflection_um: float = 400.0
    """Maximum beam displacement from the center of the array."""

    settling_time_us: float = 0.5
    """Minimum wait after a move for the AOD to stabilize."""

    rf_ramp_us: float = 0.1
    """RF power ramp time at the start and end of a move."""


# ─────────────────────────────────────────────────────────────────
#  Physical AOD pulse
# ─────────────────────────────────────────────────────────────────

@dataclass
class AODPulse:
    """A single AOD move, in physical units."""
    axis: int               # 0 = row, 1 = column
    index: int              # which row/column to grab
    delta_um: float         # signed displacement in µm
    duration_us: float      # total pulse duration including ramps
    rf_frequency_mhz: float # carrier frequency (device-specific)
    description: str = ""

    def __repr__(self) -> str:
        ax = "row" if self.axis == 0 else "col"
        sign = "+" if self.delta_um >= 0 else ""
        return (f"AODPulse({ax} {self.index:2d}, "
                f"Δ={sign}{self.delta_um:.2f}µm, "
                f"τ={self.duration_us:.2f}µs, "
                f"f={self.rf_frequency_mhz:.1f}MHz)")


@dataclass
class AODSchedule:
    """A complete sequence of AOD pulses."""
    pulses: List[AODPulse]
    total_duration_us: float

    def __repr__(self) -> str:
        return (f"AODSchedule({len(self.pulses)} pulses, "
                f"{self.total_duration_us:.1f}µs)")


# ─────────────────────────────────────────────────────────────────
#  Move → AODPulse conversion
# ─────────────────────────────────────────────────────────────────

def move_to_pulse(
    move,
    config: AODConfig,
    rf_freq_mhz: float = 80.0,
) -> AODPulse:
    """
    Convert a single abstract Move to a physical AOD pulse.

    Args:
        move: qhal.rl.infer.Move instance
        config: hardware parameters
        rf_freq_mhz: AOD acoustic carrier (device-specific)

    Returns:
        AODPulse with physical units.

    Raises:
        ValueError if the move would exceed max_deflection_um.
    """
    # Convert magnitude (lattice sites) to physical displacement
    distance_um = move.magnitude * config.lattice_spacing_um
    signed_um = distance_um * move.direction

    # Check against the AOD's maximum deflection
    if abs(signed_um) > config.max_deflection_um:
        raise ValueError(
            f"Move requires {signed_um:.1f}µm but max is "
            f"{config.max_deflection_um:.1f}µm")

    # Duration = distance / slew rate + ramp + settle
    move_time_us = abs(signed_um) / config.slew_rate_um_per_us
    total_us = move_time_us + config.rf_ramp_us + config.settling_time_us

    return AODPulse(
        axis=move.axis,
        index=move.index,
        delta_um=signed_um,
        duration_us=total_us,
        rf_frequency_mhz=rf_freq_mhz,
        description=f"{move!r}",
    )


def moves_to_schedule(
    moves: List,
    config: AODConfig,
    rf_freq_mhz: float = 80.0,
) -> AODSchedule:
    """
    Convert a full move sequence from the RL model.

    Args:
        moves: List[Move] from AtomRearranger.solve()
        config: hardware parameters
        rf_freq_mhz: AOD carrier frequency

    Returns:
        AODSchedule that the lab's controller can execute.

    Raises:
        ValueError on any individual move exceeding AOD limits.
    """
    pulses = [
        move_to_pulse(m, config, rf_freq_mhz)
        for m in moves
    ]
    total_us = sum(p.duration_us for p in pulses)
    return AODSchedule(pulses=pulses, total_duration_us=total_us)


# ─────────────────────────────────────────────────────────────────
#  Driver interface — the lab implements this
# ─────────────────────────────────────────────────────────────────

class AODDriver(Protocol):
    """
    Protocol a lab implements to accept AODPulse from the model.

    A real driver for a QuEra/Caltech/Pasqal AOD controller would
    implement these five methods. The model never calls the driver
    directly — the lab's orchestration code does.
    """

    def set_aod_position(
        self,
        axis: int,
        index: int,
        position_um: float,
        time_us: float,
    ) -> None:
        """
        Command the AOD to move a row/column to a given position.

        Args:
            axis: 0 = row AOD, 1 = column AOD
            index: which row/column to grab (0..H-1 or 0..W-1)
            position_um: absolute position from array origin
            time_us: when to execute (in the current schedule clock)
        """
        ...

    def set_aod_rf(
        self,
        axis: int,
        freq_mhz: float,
        amplitude: float,
    ) -> None:
        """Set the RF carrier for the AOD."""
        ...

    def start_sequence(self) -> None:
        """Arm the sequence for execution."""
        ...

    def trigger(self) -> None:
        """Fire the sequence."""
        ...

    def wait(self, time_us: float) -> None:
        """Block until the sequence reaches the given time."""
        ...


# ─────────────────────────────────────────────────────────────────
#  Reference execution loop (what the lab would do)
# ─────────────────────────────────────────────────────────────────

def execute_schedule(
    schedule: AODSchedule,
    driver: AODDriver,
    config: AODConfig,
    initial_positions_um: Optional[np.ndarray] = None,
) -> None:
    """
    Reference implementation of schedule execution.

    Assumes the driver holds absolute positions; caller supplies
    the initial position of every row/column. If None, assumes
    positions start at their default grid coordinates.

    Args:
        schedule: AODSchedule from moves_to_schedule()
        driver: lab-implemented AOD driver
        config: hardware parameters
        initial_positions_um: shape (H, ) for rows and (W, ) for cols
            Default: [0, lattice_spacing, 2*lattice_spacing, ...]
    """
    # Default positions: uniform grid
    H = W = max(p.index for p in schedule.pulses) + 1
    if initial_positions_um is None:
        initial_positions_um = np.arange(H) * config.lattice_spacing_um

    # Track positions per axis
    row_positions = initial_positions_um.copy()
    col_positions = initial_positions_um.copy()

    driver.start_sequence()
    t = 0.0

    for pulse in schedule.pulses:
        positions = row_positions if pulse.axis == 0 else col_positions
        old_pos = positions[pulse.index]
        new_pos = old_pos + pulse.delta_um

        driver.set_aod_rf(pulse.axis, pulse.rf_frequency_mhz, 1.0)
        driver.set_aod_position(pulse.axis, pulse.index, new_pos, t)
        driver.wait(pulse.duration_us)
        t += pulse.duration_us

        # Update local state
        positions[pulse.index] = new_pos

    driver.trigger()


# ─────────────────────────────────────────────────────────────────
#  Example: convert a 5x5 plan and print the schedule
# ─────────────────────────────────────────────────────────────────

def example():
    import sys, os
    sys.path.insert(0, "python")
    from qhal.rl.infer import AtomRearranger

    r = AtomRearranger(
        "models/best_5x5_float64/sb3_5x5_final.zip",
        grid_size=5, n_atoms=5)

    result = r.random_problem(seed=42, scramble=3)
    print(f"Model plan: {len(result.moves)} moves")
    for m in result.moves:
        print(f"  {m}")
    print()

    config = AODConfig(
        lattice_spacing_um=4.0,
        slew_rate_um_per_us=1.0,
        max_deflection_um=400.0,
        settling_time_us=0.5,
        rf_ramp_us=0.1,
    )

    schedule = moves_to_schedule(result.moves, config)
    print(f"Physical schedule: {schedule}")
    print()
    for p in schedule.pulses:
        print(f"  {p}")


if __name__ == "__main__":
    example()
