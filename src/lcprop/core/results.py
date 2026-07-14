from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticIterationRecord:
    """Serializable diagnostics for one director relaxation iteration."""

    z_index: int
    z_um: float
    optical_pass: int
    relax_iteration: int
    residual_rms: float
    residual_max: float
    delta_theta_rms: float
    delta_theta_max: float
    theta_min: float
    theta_max: float
    intensity_peak: float
    normalized_intensity_integral: float
    converged: bool


@dataclass(frozen=True)
class StaticSliceSummary:
    """Final convergence diagnostics for one z slice."""

    z_index: int
    z_um: float
    optical_passes: int
    relaxation_iterations: int
    final_residual_rms: float
    final_residual_max: float
    final_delta_theta_rms: float
    final_delta_theta_max: float
    theta_min: float
    theta_max: float
    converged: bool
    termination_reason: str


@dataclass(frozen=True)
class StaticRunResult:
    """Result returned by run_static().

    ``power_initial`` and ``power_final`` are retained compatibility names for
    dimensionless normalized channel-field integrals. Physical powers are
    reported separately in milliwatts.
    """

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

    physical_power_initial_mW: float | None = None
    physical_power_final_mW: float | None = None
    A_initial: Any | None = None
    intensity_stack: Any | None = None
    iteration_records: tuple[StaticIterationRecord, ...] = field(default_factory=tuple)
    slice_summaries: tuple[StaticSliceSummary, ...] = field(default_factory=tuple)
    all_slices_converged: bool | None = None
    max_final_residual_rms: float | None = None
    median_final_residual_rms: float | None = None
    rms_over_z_final_residual: float | None = None
    max_final_residual_max: float | None = None
    worst_slice_index: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimeDependentRunResult:
    """Result returned by run_timedependent(); power fields follow StaticRunResult."""

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
    physical_power_initial_mW: float | None = None
    physical_power_final_mW: float | None = None
    A_initial: Any | None = None
    theta_initial: Any | None = None
    initial_intensity_stack: Any | None = None
    final_intensity_stack: Any | None = None
    initial_source_intensity_stack: Any | None = None
    final_source_intensity_stack: Any | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
