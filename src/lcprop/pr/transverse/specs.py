"""Frozen specifications for the full-transverse PR Profile v1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.specs import PRMaterialSpec


PR_FULL_TRANSVERSE_PROFILE_V1 = "pr_full_transverse_unbiased_reference_v1"
PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW = "pr_transverse_timedependent"
PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE = "explicit_euler_reference"


@dataclass(frozen=True)
class PRTransverseTransportProfile:
    """Frozen isotropic nearest-neighbour transverse transport profile."""

    profile_id: str = PR_FULL_TRANSVERSE_PROFILE_V1
    m_y: float = 1.0

    def validate(self) -> None:
        if self.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
            raise ValueError("unsupported transverse transport profile")
        if not math.isfinite(float(self.m_y)) or float(self.m_y) != 1.0:
            raise ValueError("Profile v1 requires isotropic transport m_y=1")


@dataclass(frozen=True)
class PRTransverseDielectricProfile:
    """Frozen isotropic transverse dielectric/Gauss profile."""

    profile_id: str = PR_FULL_TRANSVERSE_PROFILE_V1
    h_y: float = 1.0

    def validate(self) -> None:
        if self.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
            raise ValueError("unsupported transverse dielectric profile")
        if not math.isfinite(float(self.h_y)) or float(self.h_y) != 1.0:
            raise ValueError("Profile v1 requires isotropic dielectric h_y=1")


@dataclass(frozen=True)
class PRTransverseProjectionProfile:
    """Explicit scalar optical projection, separate from carrier transport."""

    profile_id: str = PR_FULL_TRANSVERSE_PROFILE_V1
    g_x: float = 1.0
    g_y: float = 0.0

    def validate(self) -> None:
        if self.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
            raise ValueError("unsupported transverse optical projection profile")
        if float(self.g_x) != 1.0 or float(self.g_y) != 0.0:
            raise ValueError("Profile v1 requires E_active = E_x")


@dataclass(frozen=True)
class PRTransverseBoundaryProfile:
    """Periodic unbiased electrical boundary and potential-gauge profile."""

    profile_id: str = PR_FULL_TRANSVERSE_PROFILE_V1
    x_boundary: str = "periodic"
    y_boundary: str = "periodic"
    potential_gauge: str = "zero_mean_per_z_slice"
    applied_field_x: float = 0.0

    def validate(self) -> None:
        if self.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
            raise ValueError("unsupported transverse electrical boundary profile")
        if self.x_boundary != "periodic" or self.y_boundary != "periodic":
            raise ValueError("Profile v1 requires periodic x and y boundaries")
        if self.potential_gauge != "zero_mean_per_z_slice":
            raise ValueError("Profile v1 requires zero-mean psi on every z slice")
        if float(self.applied_field_x) != 0.0:
            raise ValueError("Profile v1 is unbiased and requires E_app=0")


@dataclass(frozen=True)
class PRTransverseSolverOptions:
    """Reference material-time and optical controls for Profile v1.

    Explicit Euler is exposed only as the transparent first NumPy reference;
    it is not designated as the future production-default integrator.
    """

    Nt: int = 1
    dt_normalized: float = 1.0e-3
    optical_substeps: int = 1
    integrator: str = PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE

    def validate(self) -> None:
        if int(self.Nt) < 0:
            raise ValueError("Nt must be nonnegative")
        if not math.isfinite(float(self.dt_normalized)) or self.dt_normalized <= 0:
            raise ValueError("dt_normalized must be finite and positive")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        if self.integrator != PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE:
            raise ValueError("the first NumPy transverse API supports explicit Euler only")


@dataclass(frozen=True)
class PRTransverseRunRequest:
    """Complete request for the frozen full-transverse NumPy TD workflow."""

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    transport: PRTransverseTransportProfile = PRTransverseTransportProfile()
    dielectric: PRTransverseDielectricProfile = PRTransverseDielectricProfile()
    boundary: PRTransverseBoundaryProfile = PRTransverseBoundaryProfile()
    projection: PRTransverseProjectionProfile = PRTransverseProjectionProfile()
    solver: PRTransverseSolverOptions = PRTransverseSolverOptions()
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    initial_A: Any | None = None
    initial_psi: Any | None = None


@dataclass(frozen=True)
class PRTransverseRunResult:
    """Accepted potential state and reconstructible transverse PR result."""

    A_initial: Any
    A_final: Any
    psi_initial: Any
    psi_final: Any
    power_initial: float
    power_final: float
    completed_steps: int
    time_normalized: float
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    resolved_profile: dict[str, Any]
    status: str = "completed"
    requested_steps: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PR_FULL_TRANSVERSE_PROFILE_V1",
    "PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE",
    "PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW",
    "PRTransverseBoundaryProfile",
    "PRTransverseDielectricProfile",
    "PRTransverseProjectionProfile",
    "PRTransverseRunRequest",
    "PRTransverseRunResult",
    "PRTransverseSolverOptions",
    "PRTransverseTransportProfile",
]
