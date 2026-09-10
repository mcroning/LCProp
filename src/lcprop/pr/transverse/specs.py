"""Frozen specifications for the full-transverse PR Profile v1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.screens import ChannelLaunchElements
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import PRMaterialSpec


PR_FULL_TRANSVERSE_PROFILE_V1 = "pr_full_transverse_unbiased_reference_v1"
PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1 = (
    "pr_full_transverse_periodic_biased_current_v1"
)
PR_MATERIAL_RESPONSE_NONLINEAR = "nonlinear"
PR_MATERIAL_RESPONSE_LINEARIZED = "linearized"
PR_MATERIAL_RESPONSE_MODELS = (
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PR_MATERIAL_RESPONSE_LINEARIZED,
)
PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW = "pr_transverse_timedependent"
PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE = "explicit_euler_reference"
PR_TRANSVERSE_IMEX_EULER = "spectral_imex_euler"
PR_TRANSVERSE_INTEGRATORS = (
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
)


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
    """Periodic electrical boundary, mean-field, and potential-gauge profile."""

    profile_id: str = PR_FULL_TRANSVERSE_PROFILE_V1
    x_boundary: str = "periodic"
    y_boundary: str = "periodic"
    potential_gauge: str = "zero_mean_per_z_slice"
    applied_field_x: float = 0.0

    def validate(self) -> None:
        if self.profile_id not in (
            PR_FULL_TRANSVERSE_PROFILE_V1,
            PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
        ):
            raise ValueError("unsupported transverse electrical boundary profile")
        if self.x_boundary != "periodic" or self.y_boundary != "periodic":
            raise ValueError("Profile v1 requires periodic x and y boundaries")
        if self.potential_gauge != "zero_mean_per_z_slice":
            raise ValueError("Profile v1 requires zero-mean psi on every z slice")
        if not math.isfinite(float(self.applied_field_x)):
            raise ValueError("transverse applied field must be finite")
        if (
            self.profile_id == PR_FULL_TRANSVERSE_PROFILE_V1
            and float(self.applied_field_x) != 0.0
        ):
            raise ValueError("Profile v1 is unbiased and requires E_app=0")


@dataclass(frozen=True)
class PRTransverseMaterialResponseSpec:
    """Independent constitutive response axis shared by static PR transport.

    The nonlinear default preserves the established reduced and transverse
    workflows.
    Linearized execution requires an explicit uniform total transport
    reference intensity; no background or beam-derived value is inferred.
    """

    model: str = PR_MATERIAL_RESPONSE_NONLINEAR
    reference_intensity: float | None = None

    def validate(self) -> None:
        if self.model not in PR_MATERIAL_RESPONSE_MODELS:
            raise ValueError(
                "material response must be one of "
                + ", ".join(PR_MATERIAL_RESPONSE_MODELS)
            )
        if self.model == PR_MATERIAL_RESPONSE_NONLINEAR:
            if self.reference_intensity is not None:
                raise ValueError(
                    "nonlinear material response does not use reference_intensity"
                )
            return
        value = self.reference_intensity
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                "linearized material response requires a finite positive "
                "reference_intensity"
            )

    def validate_configuration(
        self,
        *,
        material_applied_field: float,
        transport: PRTransverseTransportProfile,
        dielectric: PRTransverseDielectricProfile,
        boundary: PRTransverseBoundaryProfile,
        projection: PRTransverseProjectionProfile,
    ) -> None:
        """Validate profile identity and the single transverse bias owner."""

        self.validate()
        if float(material_applied_field) != 0.0:
            raise ValueError(
                "full-transverse workflows require zero normalized applied "
                "field in the reduced x-only material setting"
            )
        component_profile_ids = {
            transport.profile_id,
            dielectric.profile_id,
            projection.profile_id,
        }
        expected_boundary_profile = (
            PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
            if self.model == PR_MATERIAL_RESPONSE_LINEARIZED
            else PR_FULL_TRANSVERSE_PROFILE_V1
        )
        if (
            component_profile_ids != {PR_FULL_TRANSVERSE_PROFILE_V1}
            or boundary.profile_id != expected_boundary_profile
        ):
            raise ValueError(
                "transverse component and electrical boundary profiles must "
                "match the selected material response"
            )


@dataclass(frozen=True)
class PRTransverseSolverOptions:
    """Material-time integrator and optical controls for Profile v1.

    First-order spectral IMEX Euler is the production default. Explicit Euler
    remains available as the transparent reference integrator.
    """

    Nt: int = 1
    dt_normalized: float = 1.0e-3
    optical_substeps: int = 1
    integrator: str = PR_TRANSVERSE_IMEX_EULER

    def validate(self) -> None:
        if int(self.Nt) < 0:
            raise ValueError("Nt must be nonnegative")
        if not math.isfinite(float(self.dt_normalized)) or self.dt_normalized <= 0:
            raise ValueError("dt_normalized must be finite and positive")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        if self.integrator not in PR_TRANSVERSE_INTEGRATORS:
            raise ValueError(
                "integrator must be one of " + ", ".join(PR_TRANSVERSE_INTEGRATORS)
            )


@dataclass(frozen=True)
class PRTransverseRunRequest:
    """Complete request for the frozen full-transverse TD workflow.

    ``scattering`` optionally supplies the same canonical physical-z phase
    realization used by the reduced PR workflow. ``None`` preserves the
    original no-scattering Profile v1 execution.
    """

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
    scattering: PRCanonicalScatteringSpec | None = None
    launch_elements: tuple[ChannelLaunchElements, ...] = ()


@dataclass(frozen=True)
class PRTransverseRunResult:
    """Accepted potential state and reconstructible transverse PR result."""

    A_initial: Any
    A_final: Any
    psi_initial: Any | None
    psi_final: Any | None
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
    retention_summary: dict[str, Any] = field(
        default_factory=lambda: {"policy": "full", "omitted_fields": []}
    )


__all__ = [
    "PR_FULL_TRANSVERSE_PROFILE_V1",
    "PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1",
    "PR_MATERIAL_RESPONSE_LINEARIZED",
    "PR_MATERIAL_RESPONSE_MODELS",
    "PR_MATERIAL_RESPONSE_NONLINEAR",
    "PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE",
    "PR_TRANSVERSE_IMEX_EULER",
    "PR_TRANSVERSE_INTEGRATORS",
    "PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW",
    "PRTransverseBoundaryProfile",
    "PRTransverseDielectricProfile",
    "PRTransverseMaterialResponseSpec",
    "PRTransverseProjectionProfile",
    "PRTransverseRunRequest",
    "PRTransverseRunResult",
    "PRTransverseSolverOptions",
    "PRTransverseTransportProfile",
]
