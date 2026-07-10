from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticRunResult:
    """Result returned by run_static()."""

    A_final: Any
    theta_final: Any
    theta_bias: Any

    power_initial: float
    power_final: float

    grid_summary: dict
    launch_summary: dict
    bias_summary: dict

    n_steps: int
    method: str

    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimeDependentRunResult:
    """Result returned by run_timedependent()."""

    A_final: Any
    theta_final: Any
    theta_bias: Any


    power_initial: float
    power_final: float

    grid_summary: dict
    launch_summary: dict
    bias_summary: dict

    Nt: int
    method: str
    A_initial: Any | None = None
    theta_initial: Any | None = None
    initial_intensity_stack: Any | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
