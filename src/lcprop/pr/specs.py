"""Immutable specifications for the minimal headless PR workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.screens import ChannelLaunchElements
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.pr.carrier_power import carrier_channels_from_beams
from lcprop.pr.scattering import PRCanonicalScatteringSpec


_ELEMENTARY_CHARGE_C = 1.602e-19
_VACUUM_PERMITTIVITY_F_PER_M = 8.854e-12
_BOLTZMANN_J_PER_K = 1.380649e-23

PR_MATERIAL_ID = "pr"
PR_TIMEDEPENDENT_WORKFLOW = "pr_timedependent"
PR_EULER_INTEGRATOR = "euler"
PR_SEMI_IMPLICIT_INTEGRATOR = "semi_implicit_trapezoidal"
PR_EXACT_MODAL_INTEGRATOR = "exact_modal"
PR_INTEGRATORS = (
    PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_EXACT_MODAL_INTEGRATOR,
)

if TYPE_CHECKING:
    from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec


def _default_material_response() -> "PRTransverseMaterialResponseSpec":
    # The response axis historically lives in transverse.specs, which imports
    # PRMaterialSpec from this module.  Resolve the nonlinear default lazily to
    # preserve that public module layout without an import cycle.
    from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec

    return PRTransverseMaterialResponseSpec()


@dataclass(frozen=True)
class PRMaterialSpec:
    """Normalized hopping-model and scalar electro-optic parameters."""

    dark_intensity: float = 0.01
    uniform_background_intensity: float = 0.0
    applied_field: float = 0.0
    gain_length_product: float = 0.0
    refractive_index: float = 2.4

    relative_permittivity: float = 2500.0
    mobile_charge_density_m3: float = 6.4e22
    temperature_K: float = 293.0
    characteristic_wavenumber_per_um_override: float | None = None

    def validate(self) -> None:
        for name in (
            "dark_intensity",
            "uniform_background_intensity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "applied_field",
            "gain_length_product",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in (
            "refractive_index",
            "relative_permittivity",
            "mobile_charge_density_m3",
            "temperature_K",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        override = self.characteristic_wavenumber_per_um_override
        if override is not None and (
            not math.isfinite(float(override)) or float(override) <= 0.0
        ):
            raise ValueError(
                "characteristic_wavenumber_per_um_override must be finite and positive"
            )

    @property
    def background_intensity(self) -> float:
        """Total normalized dark plus externally applied uniform background."""

        return float(self.dark_intensity) + float(
            self.uniform_background_intensity
        )

    @property
    def characteristic_wavenumber_per_um(self) -> float:
        """Return paper Equation (A9), converted from inverse metres to inverse µm."""

        override = self.characteristic_wavenumber_per_um_override
        if override is not None:
            return float(override)
        numerator = float(self.mobile_charge_density_m3)
        denominator = (
            float(self.relative_permittivity)
            * _VACUUM_PERMITTIVITY_F_PER_M
            * _BOLTZMANN_J_PER_K
            * float(self.temperature_K)
        )
        return _ELEMENTARY_CHARGE_C * math.sqrt(numerator / denominator) * 1e-6


@dataclass(frozen=True)
class PRSolverOptions:
    """Normalized material-time, integrator, and optical-step controls."""

    Nt: int = 1
    dt_normalized: float = 1e-3
    optical_substeps: int = 1
    integrator: str = PR_EULER_INTEGRATOR

    def validate(self) -> None:
        if int(self.Nt) < 0:
            raise ValueError("Nt must be nonnegative")
        if not math.isfinite(float(self.dt_normalized)) or self.dt_normalized <= 0.0:
            raise ValueError("dt_normalized must be finite and positive")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        if self.integrator not in PR_INTEGRATORS:
            raise ValueError(
                "integrator must be one of " + ", ".join(PR_INTEGRATORS)
            )


@dataclass(frozen=True)
class PRRunRequest:
    """Request for the minimal headless time-dependent PR vertical slice.

    ``initial_A``, when supplied, has shape ``(Nch, Nx, Ny)``. ``initial_E``
    is the normalized physical space-charge state with shape
    ``(Nz, Nx, Ny)``.  Omitting it selects the paper/reference zero state for
    nonlinear evolution and the uniform equilibrium ``E0`` for linearized
    evolution.
    ``launch_elements`` are applied to the normalized incident beams before
    propagation and cannot be combined with an explicit ``initial_A``.
    ``scattering`` is an optional canonical physical-z phase realization;
    ``None`` preserves the historical no-scattering workflow exactly.
    """

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    solver: PRSolverOptions = PRSolverOptions()
    backend: BackendSpec = BackendSpec(
        backend="numpy",
        precision="float64",
        verbose=False,
    )
    launch_elements: tuple[ChannelLaunchElements, ...] = ()
    initial_A: Any | None = None
    initial_E: Any | None = None
    scattering: PRCanonicalScatteringSpec | None = None
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()
    material_response: "PRTransverseMaterialResponseSpec" = field(
        default_factory=_default_material_response
    )


def validate_pr_timedependent_configuration(request: PRRunRequest) -> None:
    """Validate the reduced TD response/integrator pairing."""

    request.material_response.validate()
    linearized = request.material_response.model == "linearized"
    if linearized and request.solver.integrator != PR_EXACT_MODAL_INTEGRATOR:
        raise ValueError(
            "reduced linearized TD requires integrator='exact_modal'"
        )
    if not linearized and request.solver.integrator == PR_EXACT_MODAL_INTEGRATOR:
        raise ValueError("reduced nonlinear TD does not support integrator='exact_modal'")


@dataclass(frozen=True)
class PRRunResult:
    """Physical PR state and optical products from a headless run.

    The optical fields have shape ``(Nch, Nx, Ny)``. The preserved PR states
    and the final driving-intensity stack have shape ``(Nz, Nx, Ny)``.
    """

    A_initial: Any
    A_final: Any
    E_initial: Any | None
    E_final: Any | None
    source_intensity_stack: Any | None
    power_initial: float
    power_final: float
    completed_steps: int
    time_normalized: float
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    status: str = "completed"
    requested_steps: int = 0
    checkpoint: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    retention_summary: dict[str, Any] = field(
        default_factory=lambda: {"policy": "full", "omitted_fields": []}
    )
    longitudinal_intensity_xz: Any | None = None
    longitudinal_intensity_yz: Any | None = None
    x_cut_um: float | None = None
    y_cut_um: float | None = None
    material_response_summary: dict[str, Any] = field(
        default_factory=lambda: {
            "model": "nonlinear",
            "reference_intensity": None,
            "software_evidence": "validated",
        }
    )
    intensity_preview: Any | None = None
    intensity_preview_metadata: dict[str, Any] | None = None
    td_scalar_history: tuple[dict[str, Any], ...] = ()
    td_preview_movie: Any | None = None
    td_preview_movie_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Retain compact launch directions when checkpoint provenance exists."""

        if "carrier_channels" in self.launch_summary:
            return
        request = getattr(self.checkpoint, "request", None)
        beams = getattr(request, "beams", None)
        if beams is None:
            return
        object.__setattr__(
            self,
            "launch_summary",
            {
                **self.launch_summary,
                "carrier_channels": carrier_channels_from_beams(beams),
            },
        )


__all__ = [
    "PRMaterialSpec",
    "PR_EULER_INTEGRATOR",
    "PR_EXACT_MODAL_INTEGRATOR",
    "PR_INTEGRATORS",
    "PR_MATERIAL_ID",
    "PR_SEMI_IMPLICIT_INTEGRATOR",
    "PR_TIMEDEPENDENT_WORKFLOW",
    "PRSolverOptions",
    "PRRunRequest",
    "PRRunResult",
    "validate_pr_timedependent_configuration",
]
