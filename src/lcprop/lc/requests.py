"""Canonical liquid-crystal workflow request types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.lc.specs import BiasSpec, LCMaterial


StaticStrategy = Literal["fixed_theta", "local_self_consistent"]
ThetaSolver = Literal["none", "picard_cn"]
OpticsSolver = Literal["splitstep"]
CouplingMode = Literal["frozen", "self_consistent"]
Precision = Literal["float32", "float64"]
ExistenceSolver = Literal["soliton"]
SweepExperiment = Literal["soliton"]
SweepParameter = Literal["power_mW"]
SweepExecution = Literal["sequential", "parallel"]
SweepMemberStatus = Literal["not_started", "completed", "failed", "cancelled"]


def _canonical_soliton_mode(mode: str) -> str:
    normalized = str(mode).upper()
    aliases = {
        "TEM00": "00",
        "TEM10": "10",
        "TEM01": "01",
        "TEM11": "11",
        "CUSTOM": "custom",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class RuntimeOptions:
    """Numerical execution options used by the LC workflows."""

    precision: Precision = "float64"
    optical_substeps_enabled: bool = True
    optical_dn_max_est: float = 0.02
    optical_max_phase_per_substep_rad: float = 0.30
    optical_max_substeps: int = 16

    def validate(self) -> None:
        if self.optical_dn_max_est < 0.0:
            raise ValueError("optical_dn_max_est must be >= 0")
        if self.optical_max_phase_per_substep_rad <= 0.0:
            raise ValueError("optical_max_phase_per_substep_rad must be > 0")
        if int(self.optical_max_substeps) < 1:
            raise ValueError("optical_max_substeps must be >= 1")


@dataclass(frozen=True)
class StaticWorkflowOptions:
    """High-level LC static workflow choices."""

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
        """Backward-compatible label for reports and tests."""

        return self.workflow.strategy

    @property
    def resolved_static_max_coupled_passes(self) -> int:
        return int(self.static_max_coupled_passes)

    @property
    def resolved_delta_theta_rms_tol(self) -> Optional[float]:
        if self.static_delta_theta_rms_tol is not None:
            return float(self.static_delta_theta_rms_tol)
        return None if self.tolerance_rms is None else float(self.tolerance_rms)

    @property
    def resolved_delta_theta_max_tol(self) -> Optional[float]:
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


@dataclass(frozen=True)
class SolitonRequest:
    base: "StaticRunRequest"
    mode: "str" = "00"
    max_outer: "int" = 100
    theta_steps_per_outer: "int" = 50
    field_mix: "float" = 0.5
    theta_mix: "float" = 1.0
    tol_field: "float" = 1e-4
    tol_theta: "float" = 1e-5
    tol_residual_rms: "float" = 5e-3
    tol_residual_max: "float" = 5e-2
    initial_A: "Any | None" = None
    initial_theta: "Any | None" = None
    refine_transverse: "bool" = False
    transverse_max_outer: "int" = 100
    transverse_theta_steps_per_outer: "int" = 50
    transverse_field_mix: "float" = 0.5
    transverse_theta_mix: "float" = 0.5

    def validate(self) -> None:
        self.base.grid.validate()
        self.base.material.validate()
        self.base.bias.validate()
        self.base.beams.validate()
        canonical = _canonical_soliton_mode(self.mode)
        allowed_modes = {"00", "10", "01", "11", "custom"}
        if canonical not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)} or TEM aliases, "
                f"got {self.mode!r}"
            )
        if self.max_outer < 1:
            raise ValueError("max_outer must be >= 1")
        if self.theta_steps_per_outer < 1:
            raise ValueError("theta_steps_per_outer must be >= 1")
        if not (0.0 < self.field_mix <= 1.0):
            raise ValueError("field_mix must be in (0, 1]")
        if not (0.0 < self.theta_mix <= 1.0):
            raise ValueError("theta_mix must be in (0, 1]")
        if self.transverse_max_outer < 1:
            raise ValueError("transverse_max_outer must be >= 1")
        if self.transverse_theta_steps_per_outer < 1:
            raise ValueError("transverse_theta_steps_per_outer must be >= 1")
        if not (0.0 < self.transverse_field_mix <= 1.0):
            raise ValueError("transverse_field_mix must be in (0, 1]")
        if not (0.0 < self.transverse_theta_mix <= 1.0):
            raise ValueError("transverse_theta_mix must be in (0, 1]")


@dataclass(frozen=True)
class SolitonExistenceRequest:
    base: "StaticRunRequest"
    mode: "str" = "00"
    powers_mW: "tuple[float, ...]" = (0.1, 0.2, 0.5, 1.0, 2.0)
    continuation: "bool" = True
    solver: "ExistenceSolver" = "soliton"
    soliton_max_outer: "int" = 80
    theta_steps_per_outer: "int" = 50
    field_mix: "float" = 0.25
    theta_mix: "float" = 1.0
    tol_field: "float" = 1e-4
    tol_theta: "float" = 1e-5
    tol_residual_rms: "float" = 1e-3
    tol_residual_max: "float" = 1e-2

    def validate(self) -> None:
        self.base.grid.validate()
        self.base.material.validate()
        self.base.bias.validate()
        self.base.beams.validate()
        allowed_modes = {"00", "10", "01", "11", "custom"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)}, got {self.mode!r}"
            )
        if self.solver != "soliton":
            raise ValueError("solver must be 'soliton'")
        if len(self.powers_mW) == 0:
            raise ValueError("powers_mW must be nonempty")
        for power in self.powers_mW:
            if power < 0.0:
                raise ValueError("powers_mW must be nonnegative")
        if self.soliton_max_outer < 1:
            raise ValueError("soliton_max_outer must be >= 1")
        if self.theta_steps_per_outer < 1:
            raise ValueError("theta_steps_per_outer must be >= 1")
        if not (0.0 < self.field_mix <= 1.0):
            raise ValueError("field_mix must be in (0, 1]")
        if not (0.0 < self.theta_mix <= 1.0):
            raise ValueError("theta_mix must be in (0, 1]")


@dataclass(frozen=True)
class ParameterSweepRequest:
    experiment: "SweepExperiment"
    parameter: "SweepParameter"
    values: "tuple[float, ...]"
    base: "SolitonRequest"
    continuation: "bool" = True
    execution: "SweepExecution" = "sequential"
    max_workers: "int | None" = None

    def validate(self) -> None:
        if self.experiment != "soliton":
            raise ValueError("Only experiment='soliton' is currently supported")
        if self.parameter != "power_mW":
            raise ValueError("Only parameter='power_mW' is currently supported")
        if self.execution not in {"sequential", "parallel"}:
            raise ValueError("execution must be 'sequential' or 'parallel'")
        if self.execution == "parallel" and self.continuation:
            raise ValueError("Continuation sweeps must be executed sequentially")
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1 or None")
        if not self.values:
            raise ValueError("values must contain at least one point")
        for value in self.values:
            if float(value) < 0.0:
                raise ValueError("sweep values must be nonnegative")
        self.base.validate()


__all__ = [
    "CouplingMode",
    "ExistenceSolver",
    "OpticsSolver",
    "OutputOptions",
    "ParameterSweepRequest",
    "Precision",
    "RuntimeOptions",
    "SolitonExistenceRequest",
    "SolitonRequest",
    "StaticRunRequest",
    "StaticSolverOptions",
    "StaticStrategy",
    "StaticWorkflowOptions",
    "SweepExecution",
    "SweepExperiment",
    "SweepMemberStatus",
    "SweepParameter",
    "ThetaSolver",
    "TimeDependentRunRequest",
    "TimeDependentSolverOptions",
]
