from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamStack


StaticStrategy = Literal[
    "fixed_theta",
    "local_self_consistent",
]

ThetaSolver = Literal[
    "none",
    "picard_cn",
]

OpticsSolver = Literal[
    "splitstep",
]

CouplingMode = Literal[
    "frozen",
    "self_consistent",
]




Precision = Literal["float32", "float64"]


@dataclass(frozen=True)
class RuntimeOptions:
    """Numerical execution options shared by workflows."""

    precision: Precision = "float64"
    optical_substeps_enabled: bool = True
    optical_dn_max_est: float = 0.02
    optical_max_phase_per_substep_rad: float = 0.30
    optical_max_substeps: int = 16

    def validate(self) -> None:
        if self.optical_dn_max_est < 0.0:
            raise ValueError("optical_dn_max_est must be >= 0")
        if self.optical_max_phase_per_substep_rad <= 0.0:
            raise ValueError(
                "optical_max_phase_per_substep_rad must be > 0"
            )
        if int(self.optical_max_substeps) < 1:
            raise ValueError("optical_max_substeps must be >= 1")


@dataclass(frozen=True)
class StaticWorkflowOptions:
    """
    High-level static workflow choices.

    strategy:
        fixed_theta:
            propagate through prepared theta without solving self-consistency.

        local_self_consistent:
            alternate local static theta relaxation with split-step optics.

    theta_solver:
        none:
            no theta update.

        picard_cn:
            Picard-corrected CN theta relaxation.

    optics_solver:
        splitstep:
            channel-stack split-step propagation.

    coupling:
        frozen:
            theta/intensity coupling is not iterated.

        self_consistent:
            theta and optical intensity are iterated.
    """

    strategy: StaticStrategy = "fixed_theta"
    theta_solver: ThetaSolver = "none"
    optics_solver: OpticsSolver = "splitstep"
    coupling: CouplingMode = "frozen"


@dataclass(frozen=True)
class StaticSolverOptions:
    workflow: StaticWorkflowOptions = StaticWorkflowOptions()
    max_iterations: int = 100
    tolerance_rms: Optional[float] = None
    tolerance_max: Optional[float] = None
    static_residual_rms_tol: Optional[float] = 5.0e-3
    static_residual_max_tol: Optional[float] = 2.0e-2
    static_delta_theta_rms_tol: Optional[float] = None
    static_delta_theta_max_tol: Optional[float] = None
    static_max_relax_iterations: int = 200
    static_max_coupled_passes: int = 3
    record_iteration_history: bool = True

    @property
    def method(self) -> str:
        """
        Backward-compatible label for reports/tests.

        Prefer workflow.strategy in new code.
        """
        return self.workflow.strategy

    @property
    def resolved_static_max_coupled_passes(self) -> int:
        """Return the explicit local optical/director coupled-pass limit."""
        return int(self.static_max_coupled_passes)

    @property
    def resolved_delta_theta_rms_tol(self) -> Optional[float]:
        """Map legacy ``tolerance_rms`` deliberately to theta-update RMS."""
        if self.static_delta_theta_rms_tol is not None:
            return float(self.static_delta_theta_rms_tol)
        return None if self.tolerance_rms is None else float(self.tolerance_rms)

    @property
    def resolved_delta_theta_max_tol(self) -> Optional[float]:
        """Map legacy ``tolerance_max`` deliberately to theta-update max."""
        if self.static_delta_theta_max_tol is not None:
            return float(self.static_delta_theta_max_tol)
        return None if self.tolerance_max is None else float(self.tolerance_max)


@dataclass(frozen=True)
class OutputOptions:
    run_dir: Optional[Path] = None
    save_slices: bool = True
    save_full: bool = False


@dataclass(frozen=True)
class StaticRunRequest:
    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec
    beams: BeamStack
    solver: StaticSolverOptions
    output: OutputOptions
    runtime: RuntimeOptions = RuntimeOptions()
    initial_A: Any | None = None
    initial_theta: Any | None = None


@dataclass(frozen=True)
class TimeDependentSolverOptions:
    workflow: StaticWorkflowOptions = StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )
    Nt: int = 20
    dt: float = 7.5e-4
    gamma_z: float = 0.0
    max_picard_iter: int = 4
    tolerance_update: float = 1e-6


@dataclass(frozen=True)
class TimeDependentRunRequest:
    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec
    beams: BeamStack
    solver: TimeDependentSolverOptions
    output: OutputOptions
    runtime: RuntimeOptions = RuntimeOptions()
    initial_A: Any | None = None
    initial_theta: Any | None = None
