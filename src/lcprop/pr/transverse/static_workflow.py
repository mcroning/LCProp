"""Backend-native self-consistent workflow for full-transverse PR equilibrium."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import (
    BackendSpec,
    asnumpy,
    get_backend,
    scalar_float,
    synchronize,
)
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch, normalized_power
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.launch_configuration import (
    LaunchConfiguration,
    reject_prepared_launch_conflict,
)
from lcprop.optics.screens import ChannelLaunchElements
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_provenance,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.linearized_reference import (
    FIXED_MEAN_FIELD_ENSEMBLE,
    PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1,
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseDiscreteStaticNewtonRecord,
    PRTransverseStaticMaterialSolverOptions,
    PRTransverseStaticNewtonRecord,
    _StaticCancellationRequested,
    _solve_pr_transverse_static_intensity_host_volume,
    derivative_null_residual,
    project_production_resolved_modes,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.transport import potential_rhs, state_from_potential
from lcprop.pr.workflow import (
    _apply_canonical_scattering_after_slice,
    _canonical_scattering_phase_for_slice,
    _validate_canonical_scattering_for_grid,
    advance_pr_slice_with_midpoint_source,
)


PR_TRANSVERSE_STATIC_WORKFLOW = "pr_transverse_static"
PR_COHERENT_VISIBILITY_CONTINUATION_SCHEDULE = (0.0, 0.25, 0.5, 0.75, 1.0)
ProgressCallback = Callable[[RunProgress], None]


@dataclass(frozen=True)
class PRTransverseStaticWorkflowOptions:
    """Coupled full-volume zero-flux controls, distinct from transient controls.

    The discrete-corrector and TD-residual fields remain compatibility and
    diagnostic configuration.  They do not participate in canonical static
    material, outer-loop, or final-replay acceptance.
    """

    material_solver: PRTransverseStaticMaterialSolverOptions = field(
        default_factory=PRTransverseStaticMaterialSolverOptions
    )
    discrete_corrector: PRTransverseDiscreteStaticCorrectorOptions = field(
        default_factory=PRTransverseDiscreteStaticCorrectorOptions
    )
    max_coupled_iterations: int = 20
    equilibrium_rms_tolerance: float = 1.0e-8
    equilibrium_max_tolerance: float = 1.0e-7
    td_rhs_rms_tolerance: float = 1.0e-8
    td_rhs_max_tolerance: float = 1.0e-7
    max_backtracks: int = 12
    minimum_step_scale: float = 2.0**-12
    armijo_fraction: float = 1.0e-4
    optical_substeps: int = 1
    replay_rtol: float = 1.0e-11
    replay_atol: float = 1.0e-12
    record_iteration_history: bool = True

    def validate(self) -> None:
        self.material_solver.validate()
        self.discrete_corrector.validate()
        if int(self.max_coupled_iterations) < 1:
            raise ValueError("max_coupled_iterations must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "equilibrium_rms_tolerance",
            "equilibrium_max_tolerance",
            "td_rhs_rms_tolerance",
            "td_rhs_max_tolerance",
            "minimum_step_scale",
            "replay_rtol",
            "replay_atol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseStaticRunRequest:
    """Headless request for the separate full-transverse static workflow.

    Declarative ``launch_elements`` transform normalized incident beams before
    propagation and cannot be combined with an explicit ``initial_A``.
    """

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    transport: PRTransverseTransportProfile = PRTransverseTransportProfile()
    dielectric: PRTransverseDielectricProfile = PRTransverseDielectricProfile()
    boundary: PRTransverseBoundaryProfile = PRTransverseBoundaryProfile()
    projection: PRTransverseProjectionProfile = PRTransverseProjectionProfile()
    solver: PRTransverseStaticWorkflowOptions = field(
        default_factory=PRTransverseStaticWorkflowOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    launch_elements: tuple[ChannelLaunchElements, ...] = ()
    initial_A: Any | None = None
    initial_psi: Any | None = None
    scattering: PRCanonicalScatteringSpec | None = None
    material_response: PRTransverseMaterialResponseSpec = field(
        default_factory=PRTransverseMaterialResponseSpec
    )
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()


@dataclass(frozen=True)
class PRTransverseStaticCoupledRecord:
    """One accepted or rejected refreshed-source outer correction."""

    coupled_iteration: int
    equilibrium_rms_before: float
    equilibrium_max_before: float
    td_rhs_rms_before: float
    td_rhs_max_before: float
    equilibrium_rms_after: float
    equilibrium_max_after: float
    td_rhs_rms_after: float
    td_rhs_max_after: float
    psi_change_rms: float
    psi_change_max: float
    source_change_rms: float
    optical_field_change_rms: float
    step_scale: float
    backtracks: int
    material_newton_iterations: int
    material_pcg_iterations: int
    material_discrete_newton_iterations: int
    material_gmres_iterations: int
    accepted: bool
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseStaticMaterialIterationRecord:
    """One per-plane Newton record associated with an outer iteration."""

    coupled_iteration: int
    newton_record: PRTransverseStaticNewtonRecord


@dataclass(frozen=True)
class PRTransverseStaticDiscreteIterationRecord:
    """One discrete Newton/GMRES record associated with an outer iteration."""

    coupled_iteration: int
    newton_record: PRTransverseDiscreteStaticNewtonRecord


@dataclass(frozen=True)
class PRTransverseStaticRunResult:
    """Accepted static potential, optical fields, and replay evidence."""

    A_initial: np.ndarray
    A_final: np.ndarray
    psi_initial: np.ndarray | None
    psi_final: np.ndarray | None
    source_intensity_stack: np.ndarray | None
    equilibrium_residual_stack: np.ndarray | None
    td_rhs_residual_stack: np.ndarray | None
    power_initial: float
    power_final: float
    converged: bool
    completed_coupled_iterations: int
    iteration_records: tuple[PRTransverseStaticCoupledRecord, ...]
    material_iteration_records: tuple[
        PRTransverseStaticMaterialIterationRecord, ...
    ]
    discrete_iteration_records: tuple[
        PRTransverseStaticDiscreteIterationRecord, ...
    ]
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    resolved_profile: dict[str, Any]
    replay_diagnostics: dict[str, Any]
    diagnostics: dict[str, Any]
    timing: dict[str, float]
    status: str
    retention_summary: dict[str, Any] = field(
        default_factory=lambda: {"policy": "full", "omitted_fields": []}
    )
    longitudinal_intensity_xz: np.ndarray | None = None
    longitudinal_intensity_yz: np.ndarray | None = None
    x_cut_um: float | None = None
    y_cut_um: float | None = None


def _validate_request(request: PRTransverseStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.material_response.validate_configuration(
        material_applied_field=request.material.applied_field,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
    )
    request.solver.validate()
    request.backend.validate()
    request.optical_boundary.validate()
    LaunchConfiguration(request.beams, request.launch_elements)
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    if request.scattering is not None:
        request.scattering.validate()
    if request.backend.backend not in ("numpy", "cupy"):
        raise ValueError(
            "full-transverse static workflow requires explicit "
            "backend='numpy' or backend='cupy'"
        )
    if (
        request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
        and request.backend.backend == "numpy"
        and request.backend.precision != "float64"
    ):
        raise ValueError(
            "full-transverse static NumPy reference requires precision='float64'"
        )


def _metrics(value, *, xp) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        accumulator_dtype = (
            np.float64 if value.dtype == np.dtype(np.float64) else np.float32
        )
        work = value.astype(accumulator_dtype, copy=False)
        return (
            float(np.sqrt(np.mean(work * work, dtype=accumulator_dtype))),
            float(np.max(np.abs(work))),
        )
    accumulator_dtype = (
        xp.float64 if value.dtype == np.dtype(np.float64) else xp.float32
    )
    work = value.astype(accumulator_dtype, copy=False)
    return (
        scalar_float(xp.sqrt(xp.mean(work * work))),
        scalar_float(xp.max(xp.abs(work))),
    )


def _difference_rms(left, right, *, xp) -> float:
    difference = left - right
    if isinstance(difference, np.ndarray):
        accumulator_dtype = (
            np.float64
            if difference.real.dtype == np.dtype(np.float64)
            else np.float32
        )
        return float(
            np.sqrt(
                np.mean(np.abs(difference) ** 2, dtype=accumulator_dtype)
            )
        )
    accumulator_dtype = (
        xp.float64
        if difference.real.dtype == np.dtype(np.float64)
        else xp.float32
    )
    return scalar_float(
        xp.sqrt(xp.mean(xp.abs(difference) ** 2, dtype=accumulator_dtype))
    )


def _host_initial_potential(request, *, grid, real_dtype, xp) -> np.ndarray:
    """Return a projected host potential with one backend slice live at a time."""

    expected = (grid.Nz, grid.Nx, grid.Ny)
    dtype = np.dtype(real_dtype)
    if request.initial_psi is None:
        return np.zeros(expected, dtype=dtype)
    supplied = np.asarray(asnumpy(request.initial_psi))
    if supplied.shape != expected:
        raise ValueError(
            f"initial_psi shape {supplied.shape} does not match {expected}"
        )
    if any(not np.all(np.isfinite(plane)) for plane in supplied):
        raise ValueError("initial_psi must contain only finite values")
    return supplied.astype(dtype, copy=True)


def _project_host_volume(
    volume,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp,
    cancellation_check=None,
) -> np.ndarray:
    """Project a host longitudinal volume without a backend batch FFT."""

    source = np.asarray(volume)
    projected = np.empty_like(source)
    for plane_index in range(source.shape[0]):
        if cancellation_check is not None and cancellation_check():
            raise _StaticCancellationRequested("coupled_projection_plane")
        plane = project_production_resolved_modes(
            xp.asarray(source[plane_index]),
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        projected[plane_index] = np.asarray(asnumpy(plane))
    return projected


def _host_volume_state_validity(
    volume,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp,
    cancellation_check=None,
) -> tuple[bool, float, float, float]:
    """Validate every retained plane while keeping state scratch slice-local."""

    finite = True
    carrier_minimum = math.inf
    volume_array = np.asarray(volume)
    accumulator_dtype = (
        np.float64
        if volume_array.dtype == np.dtype(np.float64)
        else np.float32
    )
    carrier_sums = np.empty(volume_array.shape[0], dtype=accumulator_dtype)
    potential_max_abs = 0.0
    count = 0
    for plane_index, plane in enumerate(volume_array):
        if cancellation_check is not None and cancellation_check():
            raise _StaticCancellationRequested("coupled_validity_plane")
        state = state_from_potential(
            xp.asarray(plane),
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        finite = finite and bool(asnumpy(xp.all(xp.isfinite(state.psi))))
        finite = finite and bool(
            asnumpy(xp.all(xp.isfinite(state.carrier_density)))
        )
        carrier_minimum = min(
            carrier_minimum, scalar_float(xp.min(state.carrier_density))
        )
        carrier_sums[plane_index] = np.asarray(
            asnumpy(xp.sum(state.carrier_density, dtype=state.psi.dtype))
        )
        potential_max_abs = max(
            potential_max_abs, scalar_float(xp.max(xp.abs(state.psi)))
        )
        count += int(state.psi.size)
    carrier_mean = float(
        np.sum(carrier_sums, dtype=accumulator_dtype)
        / np.asarray(count, dtype=accumulator_dtype)
    )
    potential_mean = float(np.mean(volume_array, dtype=accumulator_dtype))
    dtype_epsilon = float(np.finfo(volume_array.dtype).eps)
    valid = (
        finite
        and carrier_minimum > 0.0
        and abs(carrier_mean - 1.0) <= 64.0 * dtype_epsilon
        and abs(potential_mean)
        <= 64.0 * dtype_epsilon * max(1.0, potential_max_abs)
    )
    return valid, carrier_mean, carrier_minimum, potential_mean


def _host_volume_diagnostics(
    volume,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp,
) -> dict[str, Any]:
    """Aggregate established state diagnostics from independent z planes."""

    carrier_integrals = []
    sums = {
        "curl": 0.0,
        "gauss": 0.0,
        "E_x": 0.0,
        "E_y": 0.0,
        "P_minus_one": 0.0,
    }
    maxima = {name: 0.0 for name in sums}
    potential_mean_max_abs = 0.0
    finite = True
    count = 0
    for plane in np.asarray(volume):
        state = state_from_potential(
            xp.asarray(plane),
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        plane_diagnostics = state_diagnostics(
            state,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        carrier_integrals.append(
            float(np.asarray(plane_diagnostics["carrier_integrals_per_z"]).item())
        )
        plane_count = int(plane.size)
        count += plane_count
        for name in sums:
            rms_key = f"{name}_rms"
            max_key = "P_minus_one_max_abs" if name == "P_minus_one" else (
                f"{name}_max_abs" if name in ("E_x", "E_y") else f"{name}_max"
            )
            sums[name] += float(plane_diagnostics[rms_key]) ** 2 * plane_count
            maxima[name] = max(maxima[name], float(plane_diagnostics[max_key]))
        potential_mean_max_abs = max(
            potential_mean_max_abs,
            float(plane_diagnostics["potential_mean_max_abs"]),
        )
        finite = finite and bool(plane_diagnostics["finite_material_state"])
    return {
        "carrier_integrals_per_z": np.asarray(carrier_integrals),
        "curl_rms": math.sqrt(sums["curl"] / count),
        "curl_max": maxima["curl"],
        "gauss_rms": math.sqrt(sums["gauss"] / count),
        "gauss_max": maxima["gauss"],
        "E_x_rms": math.sqrt(sums["E_x"] / count),
        "E_x_max_abs": maxima["E_x"],
        "E_y_rms": math.sqrt(sums["E_y"] / count),
        "E_y_max_abs": maxima["E_y"],
        "P_minus_one_rms": math.sqrt(sums["P_minus_one"] / count),
        "P_minus_one_max_abs": maxima["P_minus_one"],
        "potential_mean_max_abs": potential_mean_max_abs,
        "finite_material_state": finite,
    }


def _host_derivative_null_metrics(
    residual,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp,
) -> tuple[float, float]:
    """Measure the null component without constructing a backend volume."""

    volume = np.asarray(residual)
    sum_squares = 0.0
    maximum = 0.0
    for plane in volume:
        null = derivative_null_residual(
            xp.asarray(plane),
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        host = np.asarray(asnumpy(null), dtype=np.float64)
        sum_squares += float(np.sum(host * host))
        maximum = max(maximum, float(np.max(np.abs(host))))
    return math.sqrt(sum_squares / volume.size), maximum


def _residuals(
    psi,
    source,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float,
    xp,
    cancellation_check=None,
):
    potential_volume = np.asarray(psi)
    source_volume = np.asarray(source)
    if potential_volume.shape != source_volume.shape or potential_volume.ndim != 3:
        raise ValueError("psi and source must share shape (Nz, Nx, Ny)")
    equilibrium = np.empty_like(potential_volume)
    td = np.empty_like(potential_volume)
    accumulator_dtype = (
        np.float64
        if potential_volume.dtype == np.dtype(np.float64)
        else np.float32
    )
    equilibrium_sum_squares = np.asarray(0.0, dtype=accumulator_dtype)
    equilibrium_max = 0.0
    td_sum_squares = np.asarray(0.0, dtype=accumulator_dtype)
    td_max = 0.0
    count = int(potential_volume.size)
    for plane_index in range(potential_volume.shape[0]):
        if cancellation_check is not None and cancellation_check():
            raise _StaticCancellationRequested("coupled_residual_plane")
        potential_plane = xp.asarray(potential_volume[plane_index])
        source_plane = xp.asarray(source_volume[plane_index])
        equilibrium_plane = static_equilibrium_residual(
            potential_plane,
            source_plane,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        td_plane = potential_rhs(
            potential_plane,
            source_plane,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            applied_field_x=0.0,
            xp=xp,
        )
        resolved_plane = project_production_resolved_modes(
            equilibrium_plane,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=h_y,
            xp=xp,
        )
        equilibrium[plane_index] = np.asarray(asnumpy(equilibrium_plane))
        td[plane_index] = np.asarray(asnumpy(td_plane))
        resolved_host = np.asarray(asnumpy(resolved_plane), dtype=accumulator_dtype)
        td_host = np.asarray(asnumpy(td_plane), dtype=accumulator_dtype)
        equilibrium_sum_squares = np.add(
            equilibrium_sum_squares,
            np.sum(resolved_host * resolved_host, dtype=accumulator_dtype),
            dtype=accumulator_dtype,
        )
        equilibrium_max = max(
            equilibrium_max, float(np.max(np.abs(resolved_host)))
        )
        td_sum_squares = np.add(
            td_sum_squares,
            np.sum(td_host * td_host, dtype=accumulator_dtype),
            dtype=accumulator_dtype,
        )
        td_max = max(td_max, float(np.max(np.abs(td_host))))
    return equilibrium, td, (
        float(
            np.sqrt(
                equilibrium_sum_squares
                / np.asarray(count, dtype=accumulator_dtype)
            )
        ),
        equilibrium_max,
        float(
            np.sqrt(
                td_sum_squares / np.asarray(count, dtype=accumulator_dtype)
            )
        ),
        td_max,
    )


def _criteria_met(metrics, options: PRTransverseStaticWorkflowOptions) -> bool:
    equilibrium_rms, equilibrium_max, _, _ = metrics
    return (
        equilibrium_rms <= float(options.equilibrium_rms_tolerance)
        and equilibrium_max <= float(options.equilibrium_max_tolerance)
    )


@dataclass(frozen=True)
class _LinearizedVolumeResponse:
    delta_psi: np.ndarray
    delta_mean_current: np.ndarray
    mean_intensity_perturbation: np.ndarray
    plane_solves: int
    forward_fft_count: int = 1
    inverse_fft_count: int = 4


def _linearized_material_response(
    source,
    *,
    request: PRTransverseStaticRunRequest,
    dx_normalized: float,
    dy_normalized: float,
    cancellation_check=None,
):
    """Apply the commissioned frozen-intensity operator to a source batch."""

    reference_intensity = request.material_response.reference_intensity
    if reference_intensity is None:  # guarded by request validation
        raise ValueError("linearized material response requires reference_intensity")
    source_volume = np.asarray(source)
    if source_volume.ndim != 3:
        raise ValueError("production linearized source must have shape (Nz, Nx, Ny)")
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=float(reference_intensity),
        applied_field=float(request.boundary.applied_field_x),
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=float(request.transport.m_y),
        h_y=float(request.dielectric.h_y),
    )
    potential = np.empty_like(source_volume)
    mean_current = np.empty((source_volume.shape[0], 2), dtype=source_volume.dtype)
    mean_intensity = np.empty(source_volume.shape[0], dtype=source_volume.dtype)
    for plane_index, source_plane in enumerate(source_volume):
        if cancellation_check is not None and cancellation_check():
            raise _StaticCancellationRequested("linearized_material_plane")
        response = solve_pr_biased_linearized_reference(
            source_plane,
            spec=spec,
            backend=request.backend,
        )
        potential[plane_index] = np.asarray(asnumpy(response.delta_psi))
        mean_current[plane_index] = np.asarray(
            asnumpy(response.delta_mean_current)
        )
        mean_intensity[plane_index] = float(
            np.asarray(asnumpy(response.mean_intensity_perturbation))
        )
    return _LinearizedVolumeResponse(
        delta_psi=potential,
        delta_mean_current=mean_current,
        mean_intensity_perturbation=mean_intensity,
        plane_solves=source_volume.shape[0],
    )


def _contains_coherent_interference(beams: BeamStack) -> bool:
    """Return whether two or more channels contribute coherent cross terms."""

    groups = beams.coherence_groups
    return len(set(groups)) < len(groups)


def _fully_incoherent_request(
    request: PRTransverseStaticRunRequest,
) -> PRTransverseStaticRunRequest:
    """Return the same physical channels with all coherent cross terms removed."""

    channels = tuple(
        replace(channel, coherence_group=f"__pr_visibility_channel_{index}__")
        for index, channel in enumerate(request.beams.channels)
    )
    return replace(request, beams=replace(request.beams, channels=channels))


def _optical_pass_host_volume(
    A0,
    psi,
    *,
    request: PRTransverseStaticRunRequest,
    grid,
    kernel,
    peak_reference: float,
    wavelength_um: float,
    dx_normalized: float,
    dy_normalized: float,
    scattering_phase_stack=None,
    cancellation_check=None,
    cancellation_stage: str = "optical_z_march",
):
    """Propagate through host-retained material with slice-local GPU state."""

    xp = grid.xp
    potential_volume = np.asarray(psi)
    expected = (grid.Nz, grid.Nx, grid.Ny)
    if potential_volume.shape != expected:
        raise ValueError(f"psi shape {potential_volume.shape} does not match {expected}")
    source = np.empty(expected, dtype=np.dtype(grid.real_dtype))
    A = A0.copy()
    intensity_before = None
    for z_index in range(grid.Nz):
        if cancellation_check is not None and cancellation_check():
            raise _StaticCancellationRequested(cancellation_stage)
        state = state_from_potential(
            xp.asarray(potential_volume[z_index]),
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=request.dielectric.h_y,
            applied_field_x=request.boundary.applied_field_x,
            xp=xp,
        )
        active = project_active_field(
            state.E_x, state.E_y, profile=request.projection, xp=xp
        )
        A, source_plane, intensity_before = advance_pr_slice_with_midpoint_source(
            A,
            active,
            kernel=kernel,
            optical_substeps=request.solver.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=xp,
            optical_boundary=request.optical_boundary,
            boundary_grid=grid,
            _intensity_before=intensity_before,
            _return_exit_intensity=True,
        )
        source[z_index] = np.asarray(asnumpy(source_plane))
        _apply_canonical_scattering_after_slice(
            A,
            scattering=request.scattering,
            z_index=z_index,
            grid=grid,
            z_length_um=request.grid.z_length_um,
            xp=xp,
            phase=(
                None
                if scattering_phase_stack is None
                else xp.asarray(scattering_phase_stack[z_index])
            ),
        )
    return A, source


def _host_scattering_phase_stack(
    scattering,
    *,
    grid,
    z_length_um: float,
    xp,
):
    """Cache runtime-backend scattering with only one GPU plane live."""

    if scattering is None:
        return None
    phases = np.empty((grid.Nz, grid.Nx, grid.Ny), dtype=np.dtype(grid.real_dtype))
    for z_index in range(grid.Nz):
        phase = _canonical_scattering_phase_for_slice(
            scattering,
            z_index=z_index,
            grid=grid,
            z_length_um=z_length_um,
            xp=xp,
        )
        phases[z_index] = np.asarray(asnumpy(phase))
    return phases


def _optical_pass_at_visibility(
    A0,
    psi,
    *,
    visibility: float,
    request: PRTransverseStaticRunRequest,
    grid,
    kernel,
    peak_reference: float,
    wavelength_um: float,
    dx_normalized: float,
    dy_normalized: float,
    scattering_phase_stack=None,
    cancellation_check=None,
    cancellation_stage: str = "optical_z_march",
):
    """Return the exact endpoint or blended-visibility midpoint source.

    Optical propagation is channel-wise and therefore independent of the
    coherence grouping.  Only the PR-driving intensity contains coherent
    cross terms.  Intermediate continuation stages use
    ``I_incoherent + visibility * (I_coherent - I_incoherent)``; the two
    endpoints call the corresponding established optical pass exactly once.
    """

    resolved = float(visibility)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("visibility must be finite and between zero and one")
    common = {
        "grid": grid,
        "kernel": kernel,
        "peak_reference": peak_reference,
        "wavelength_um": wavelength_um,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
        "scattering_phase_stack": scattering_phase_stack,
        "cancellation_check": cancellation_check,
        "cancellation_stage": cancellation_stage,
    }
    if resolved == 1.0:
        return _optical_pass_host_volume(A0, psi, request=request, **common)

    incoherent_request = _fully_incoherent_request(request)
    A_incoherent, source_incoherent = _optical_pass_host_volume(
        A0, psi, request=incoherent_request, **common
    )
    if resolved == 0.0:
        return A_incoherent, source_incoherent

    A_coherent, source_coherent = _optical_pass_host_volume(
        A0, psi, request=request, **common
    )
    source = source_incoherent + resolved * (
        source_coherent - source_incoherent
    )
    return A_coherent, source


def _run_pr_transverse_static_at_visibility(
    request: PRTransverseStaticRunRequest,
    *,
    visibility: float,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseStaticRunResult:
    """Solve one fixed coherent-visibility continuation stage."""

    started = perf_counter()
    _validate_request(request)
    backend = get_backend(request.backend)
    xp = backend.xp
    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    _validate_canonical_scattering_for_grid(
        request.scattering, grid=grid, z_length_um=request.grid.z_length_um
    )
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=backend.complex_dtype,
        launch_elements=request.launch_elements,
        context=OpticalLaunchContext(
            grid=grid,
            n_ref=float(request.material.refractive_index),
            interaction_length_um=float(request.grid.z_length_um),
        ),
    )
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=backend.complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(
                f"initial_A shape {A0.shape} does not match {launch.A0.shape}"
            )
        if not bool(asnumpy(xp.all(xp.isfinite(A0)))):
            raise ValueError("initial_A must contain only finite values")
    k0 = request.material.characteristic_wavenumber_per_um
    dx_normalized = k0 * grid.dx_um
    dy_normalized = k0 * grid.dy_um
    psi = _project_host_volume(
        _host_initial_potential(
            request, grid=grid, real_dtype=backend.real_dtype, xp=xp
        ),
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    psi_initial = psi.copy()
    (
        initial_state_valid,
        _,
        initial_carrier_minimum,
        _,
    ) = _host_volume_state_validity(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    if not initial_state_valid:
        raise ValueError(
            "initial_psi violates the finite, positive, normalized, zero-mean "
            f"static state contract (carrier minimum {initial_carrier_minimum})"
        )
    wavelengths = tuple(float(channel.wavelength_um) for channel in request.beams.channels)
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("transverse PR workflow requires one shared wavelength")
    wavelength_um = wavelengths[0]
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    # The deterministic cache is retained on the host.  Every optical pass
    # transfers only its current phase plane, preserving the accepted cached
    # scattering trajectory without a persistent Nz-sized GPU allocation.
    scattering_phase_stack = _host_scattering_phase_stack(
        request.scattering,
        grid=grid,
        z_length_um=request.grid.z_length_um,
        xp=np,
    )
    optical_seconds = 0.0
    material_seconds = 0.0
    zero_flux_material_seconds = 0.0
    linearized_material_seconds = 0.0
    material_response_calls = 0
    latest_linearized_response = None

    def optical_pass(state, *, cancellable: bool = False):
        nonlocal optical_seconds
        synchronize(xp)
        pass_started = perf_counter()
        result = _optical_pass_at_visibility(
            A0,
            state,
            visibility=visibility,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            wavelength_um=wavelength_um,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            scattering_phase_stack=scattering_phase_stack,
            cancellation_check=(
                cancellation_token.is_cancelled
                if cancellable and cancellation_token is not None
                else None
            ),
            cancellation_stage="coupled_optical_z_march",
        )
        synchronize(xp)
        optical_seconds += perf_counter() - pass_started
        return result

    def material_residuals(state, source, *, cancellable: bool = False):
        nonlocal material_response_calls, latest_linearized_response
        if request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR:
            return _residuals(
                state,
                source,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                h_y=request.dielectric.h_y,
                cancellation_check=(
                    cancellation_token.is_cancelled
                    if cancellable and cancellation_token is not None
                    else None
                ),
                xp=xp,
            )
        response = _linearized_material_response(
            source,
            request=request,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            cancellation_check=(
                cancellation_token.is_cancelled
                if cancellable and cancellation_token is not None
                else None
            ),
        )
        material_response_calls += response.plane_solves
        latest_linearized_response = response
        residual = np.asarray(state) - np.asarray(asnumpy(response.delta_psi))
        rms, maximum = _metrics(residual, xp=np)
        return residual, np.zeros_like(residual), (rms, maximum, 0.0, 0.0)

    A_accepted, source_accepted = optical_pass(psi)
    equilibrium, td_residual, metrics = material_residuals(
        psi, source_accepted
    )
    records: list[PRTransverseStaticCoupledRecord] = []
    material_records: list[PRTransverseStaticMaterialIterationRecord] = []
    discrete_records: list[PRTransverseStaticDiscreteIterationRecord] = []
    completed_iterations = 0
    converged = _criteria_met(metrics, request.solver)
    cancelled = False
    cancellation_observed_stage = None
    termination_reason = "residual_tolerance" if converged else "maximum_coupled_iterations"

    for coupled_iteration in range(1, int(request.solver.max_coupled_iterations) + 1):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            cancellation_observed_stage = "coupled_iteration_boundary"
            termination_reason = "cancelled_at_accepted_boundary"
            break
        if converged:
            break
        synchronize(xp)
        material_started = perf_counter()
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
            try:
                material = _linearized_material_response(
                    source_accepted,
                    request=request,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    cancellation_check=(
                        cancellation_token.is_cancelled
                        if cancellation_token is not None
                        else None
                    ),
                )
            except _StaticCancellationRequested as exc:
                synchronize(xp)
                elapsed_material = perf_counter() - material_started
                material_seconds += elapsed_material
                linearized_material_seconds += elapsed_material
                cancelled = True
                cancellation_observed_stage = exc.stage
                termination_reason = "cancelled_during_material_trial"
                break
            material_response_calls += material.plane_solves
            latest_linearized_response = material
            material_proposal = np.asarray(asnumpy(material.delta_psi))
            material_converged = True
            material_status = "analytic_frozen_intensity_solve"
            material_newton = 0
            material_pcg = 0
            material_discrete_newton = 0
            material_gmres = 0
        else:
            try:
                material = _solve_pr_transverse_static_intensity_host_volume(
                    source_accepted,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    initial_psi=psi,
                    h_y=request.dielectric.h_y,
                    options=request.solver.material_solver,
                    cancellation_check=(
                        cancellation_token.is_cancelled
                        if cancellation_token is not None
                        else None
                    ),
                    xp=xp,
                )
            except _StaticCancellationRequested as exc:
                synchronize(xp)
                elapsed_material = perf_counter() - material_started
                material_seconds += elapsed_material
                zero_flux_material_seconds += elapsed_material
                cancelled = True
                cancellation_observed_stage = exc.stage
                termination_reason = "cancelled_during_material_trial"
                break
            material_proposal = material.psi
            material_converged = material.converged
            material_status = material.status
            material_newton = sum(
                summary.newton_iterations
                for summary in material.plane_summaries
            )
            material_pcg = sum(
                summary.pcg_iterations
                for summary in material.plane_summaries
            )
            material_discrete_newton = 0
            material_gmres = 0
            if request.solver.record_iteration_history:
                material_records.extend(
                    PRTransverseStaticMaterialIterationRecord(
                        coupled_iteration=coupled_iteration,
                        newton_record=record,
                    )
                    for record in material.iteration_records
                )
        synchronize(xp)
        elapsed_material = perf_counter() - material_started
        material_seconds += elapsed_material
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
            linearized_material_seconds += elapsed_material
        else:
            zero_flux_material_seconds += material.elapsed_seconds
        # The fixed-source residual volumes have served their convergence and
        # provenance purpose inside the material solve.  Do not retain them
        # through the coupled line search, which constructs refreshed residual
        # volumes of its own.
        del material
        if cancellation_token is not None and cancellation_token.is_cancelled():
            del material_proposal
            cancelled = True
            cancellation_observed_stage = "after_material_candidate"
            termination_reason = "cancelled_at_accepted_boundary"
            break
        if not material_converged:
            del material_proposal
            termination_reason = f"material_{material_status}"
            if request.solver.record_iteration_history:
                records.append(PRTransverseStaticCoupledRecord(
                    coupled_iteration=coupled_iteration,
                    equilibrium_rms_before=metrics[0],
                    equilibrium_max_before=metrics[1],
                    td_rhs_rms_before=metrics[2],
                    td_rhs_max_before=metrics[3],
                    equilibrium_rms_after=metrics[0],
                    equilibrium_max_after=metrics[1],
                    td_rhs_rms_after=metrics[2],
                    td_rhs_max_after=metrics[3],
                    psi_change_rms=0.0,
                    psi_change_max=0.0,
                    source_change_rms=0.0,
                    optical_field_change_rms=0.0,
                    step_scale=0.0,
                    backtracks=0,
                    material_newton_iterations=material_newton,
                    material_pcg_iterations=material_pcg,
                    material_discrete_newton_iterations=material_discrete_newton,
                    material_gmres_iterations=material_gmres,
                    accepted=False,
                    termination_reason=termination_reason,
                ))
            break

        direction = material_proposal - psi
        del material_proposal
        step_scale = 1.0
        accepted = False
        backtracks = 0
        candidate_metrics = metrics
        candidate_psi = psi
        candidate_A = A_accepted
        candidate_source = source_accepted
        candidate_equilibrium = equilibrium
        candidate_td = td_residual
        for backtracks in range(int(request.solver.max_backtracks) + 1):
            if cancellation_token is not None and cancellation_token.is_cancelled():
                cancelled = True
                cancellation_observed_stage = "coupled_line_search"
                termination_reason = "cancelled_during_trial"
                break
            try:
                trial_psi = _project_host_volume(
                    psi + step_scale * direction,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=request.dielectric.h_y,
                    xp=xp,
                    cancellation_check=(
                        cancellation_token.is_cancelled
                        if cancellation_token is not None
                        else None
                    ),
                )
                valid_trial, _, _, _ = _host_volume_state_validity(
                    trial_psi,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    h_y=request.dielectric.h_y,
                    xp=xp,
                    cancellation_check=(
                        cancellation_token.is_cancelled
                        if cancellation_token is not None
                        else None
                    ),
                )
            except _StaticCancellationRequested as exc:
                cancelled = True
                cancellation_observed_stage = exc.stage
                termination_reason = "cancelled_during_trial"
                break
            if valid_trial:
                try:
                    trial_A, trial_source = optical_pass(
                        trial_psi, cancellable=True
                    )
                    trial_equilibrium, trial_td, trial_metrics = material_residuals(
                        trial_psi, trial_source, cancellable=True
                    )
                except _StaticCancellationRequested as exc:
                    cancelled = True
                    cancellation_observed_stage = exc.stage
                    termination_reason = "cancelled_during_trial"
                    break
                if (
                    math.isfinite(trial_metrics[0])
                    and math.isfinite(trial_metrics[1])
                    and trial_metrics[0]
                    <= (1.0 - float(request.solver.armijo_fraction) * step_scale)
                    * metrics[0]
                    and trial_metrics[1] <= metrics[1]
                ):
                    accepted = True
                    candidate_psi = trial_psi
                    candidate_A = trial_A
                    candidate_source = trial_source
                    candidate_equilibrium = trial_equilibrium
                    candidate_td = trial_td
                    candidate_metrics = trial_metrics
                    break
            step_scale *= 0.5
            if step_scale < float(request.solver.minimum_step_scale):
                break

        if (
            accepted
            and cancellation_token is not None
            and cancellation_token.is_cancelled()
        ):
            accepted = False
            cancelled = True
            cancellation_observed_stage = "after_coupled_trial"
            termination_reason = "cancelled_during_trial"
            candidate_metrics = metrics
            candidate_psi = psi
            candidate_A = A_accepted
            candidate_source = source_accepted
            candidate_equilibrium = equilibrium
            candidate_td = td_residual

        delta = candidate_psi - psi
        delta_rms, delta_max = _metrics(delta, xp=xp)
        if request.solver.record_iteration_history:
            records.append(PRTransverseStaticCoupledRecord(
                coupled_iteration=coupled_iteration,
                equilibrium_rms_before=metrics[0],
                equilibrium_max_before=metrics[1],
                td_rhs_rms_before=metrics[2],
                td_rhs_max_before=metrics[3],
                equilibrium_rms_after=candidate_metrics[0],
                equilibrium_max_after=candidate_metrics[1],
                td_rhs_rms_after=candidate_metrics[2],
                td_rhs_max_after=candidate_metrics[3],
                psi_change_rms=delta_rms,
                psi_change_max=delta_max,
                source_change_rms=_difference_rms(
                    candidate_source, source_accepted, xp=xp
                ),
                optical_field_change_rms=_difference_rms(
                    candidate_A, A_accepted, xp=xp
                ),
                step_scale=step_scale if accepted else 0.0,
                backtracks=backtracks,
                material_newton_iterations=material_newton,
                material_pcg_iterations=material_pcg,
                material_discrete_newton_iterations=material_discrete_newton,
                material_gmres_iterations=material_gmres,
                accepted=accepted,
                termination_reason=(
                    "accepted"
                    if accepted
                    else (
                        "cancelled_during_trial"
                        if cancelled
                        else "coupled_line_search_failed"
                    )
                ),
            ))
        if not accepted:
            if not cancelled:
                termination_reason = "coupled_line_search_failed"
            break
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            cancellation_observed_stage = "before_accepting_coupled_trial"
            termination_reason = "cancelled_at_accepted_boundary"
            break
        psi = candidate_psi
        A_accepted = candidate_A
        source_accepted = candidate_source
        equilibrium = candidate_equilibrium
        td_residual = candidate_td
        metrics = candidate_metrics
        completed_iterations = coupled_iteration
        converged = _criteria_met(metrics, request.solver)
        termination_reason = (
            "residual_tolerance" if converged else "maximum_coupled_iterations"
        )
        if progress_callback is not None:
            progress_callback(RunProgress(
                workflow=PR_TRANSVERSE_STATIC_WORKFLOW,
                status="running",
                completed_units=completed_iterations,
                total_units=int(request.solver.max_coupled_iterations),
                current_coordinate=float(completed_iterations),
                coordinate_name="coupled_iteration",
                coordinate_unit="1",
                elapsed_wall_time=perf_counter() - started,
                latest_field_state={
                    "psi_current": np.asarray(asnumpy(psi)).copy()
                },
                diagnostics={
                    "material_response": request.material_response.model,
                    "material_response_rms": metrics[0],
                    "material_response_max": metrics[1],
                    "equilibrium_rms": metrics[0],
                    "equilibrium_max": metrics[1],
                    "td_rhs_rms": metrics[2],
                    "td_rhs_max": metrics[3],
                },
                message="transverse PR static outer iteration accepted",
            ))

    # Accepted residuals are recomputed independently from the replayed source;
    # releasing them first avoids overlapping two complete host residual pairs.
    del equilibrium, td_residual
    replay_A, replay_source = optical_pass(psi)
    replay_field_match = bool(asnumpy(xp.allclose(
        replay_A,
        A_accepted,
        rtol=float(request.solver.replay_rtol),
        atol=float(request.solver.replay_atol),
    )))
    replay_source_match = bool(np.allclose(
        replay_source,
        source_accepted,
        rtol=float(request.solver.replay_rtol),
        atol=float(request.solver.replay_atol),
    ))
    replay_equilibrium, replay_td_residual, replay_metrics = material_residuals(
        psi, replay_source
    )
    replay_diagnostics = {
        "field_match": bool(replay_field_match),
        "source_match": bool(replay_source_match),
        "field_relative_l2": _difference_rms(replay_A, A_accepted, xp=xp)
        / max(
            _difference_rms(replay_A, xp.zeros_like(replay_A), xp=xp),
            np.finfo(float).tiny,
        ),
        "source_relative_l2": _difference_rms(
            replay_source, source_accepted, xp=xp
        )
        / max(
            _difference_rms(replay_source, np.zeros_like(replay_source), xp=xp),
            np.finfo(float).tiny,
        ),
        "complete_independent_replay": True,
    }
    if converged and not (replay_field_match and replay_source_match):
        converged = False
        termination_reason = "replay_mismatch"
    if converged and not _criteria_met(replay_metrics, request.solver):
        converged = False
        termination_reason = (
            "final_material_response_consistency_not_met"
            if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
            else "final_zero_flux_residual_not_met"
        )
    equilibrium = replay_equilibrium
    td_residual = replay_td_residual
    metrics = replay_metrics

    (
        final_state_valid,
        final_carrier_mean,
        final_carrier_minimum,
        final_potential_mean,
    ) = _host_volume_state_validity(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    if converged and not final_state_valid:
        converged = False
        termination_reason = "final_physical_state_invalid"
    diagnostics = _host_volume_diagnostics(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    null_rms, null_max = _host_derivative_null_metrics(
        equilibrium,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    # The authoritative scalar convergence gates use the production-resolved
    # subspace. Export the same field while retaining the raw residual above
    # for the separately reported derivative-null diagnostic.
    authoritative_equilibrium = _project_host_volume(
        equilibrium,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    power_initial = normalized_power(A0, grid)
    power_final = normalized_power(replay_A, grid)
    diagnostics.update({
        "equilibrium_residual_rms": metrics[0],
        "equilibrium_residual_max": metrics[1],
        "td_rhs_residual_rms": metrics[2],
        "td_rhs_residual_max": metrics[3],
        "authoritative_static_residual": "zero_flux_equilibrium",
        "td_rhs_residual_role": "diagnostic_only",
        "physical_state_valid": final_state_valid,
        "carrier_mean": final_carrier_mean,
        "carrier_minimum": final_carrier_minimum,
        "potential_mean": final_potential_mean,
        "derivative_null_residual_rms": null_rms,
        "derivative_null_residual_max": null_max,
        "discrete_corrector": {
            "invoked": False,
            "role": "experimental_td_fixed_point_diagnostic_only",
            "continuum_newton_iterations_attempted": len(material_records),
            "continuum_newton_iterations_accepted": sum(
                int(record.newton_record.accepted) for record in material_records
            ),
            "continuum_pcg_iterations": sum(
                record.newton_record.pcg_iterations for record in material_records
            ),
            "discrete_newton_iterations_attempted": len(discrete_records),
            "discrete_newton_iterations_accepted": sum(
                int(record.newton_record.accepted) for record in discrete_records
            ),
            "gmres_iterations": sum(
                record.newton_record.gmres_iterations for record in discrete_records
            ),
            "gmres_restarts": sum(
                record.newton_record.gmres_restarts for record in discrete_records
            ),
            "backtracks": sum(
                record.newton_record.backtracks for record in discrete_records
            ),
            "authoritative_convergence": "zero_flux_equilibrium_rms_and_max",
        },
        "optical_power_relative_drift": (power_final - power_initial) / power_initial,
        "termination_reason": termination_reason,
        "cancelled": cancelled,
        "cancellation_observed_stage": cancellation_observed_stage,
        "memory_policy": {
            "backend_working_set": "one_transverse_slice",
            "retained_final_volumes": "host",
            "retained_iteration_volumes": False,
            "fft_batch_axes": "transverse_only",
        },
    })
    if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
        if latest_linearized_response is None:
            raise RuntimeError("linearized material response provenance is unavailable")
        delta_mean_current = np.asarray(
            asnumpy(latest_linearized_response.delta_mean_current)
        )
        total_mean_current = delta_mean_current.copy()
        total_mean_current[..., 0] -= (
            float(request.material_response.reference_intensity)
            * float(request.boundary.applied_field_x)
        )
        diagnostics.pop("td_rhs_residual_rms", None)
        diagnostics.pop("td_rhs_residual_max", None)
        diagnostics.pop("discrete_corrector", None)
        diagnostics.update({
            "material_response": PR_MATERIAL_RESPONSE_LINEARIZED,
            "material_response_validation": "experimental",
            "authoritative_static_residual": "material_response_consistency",
            "material_response_consistency_rms": metrics[0],
            "material_response_consistency_max": metrics[1],
            "frozen_intensity_material_equation": "analytic_fourier_solve",
            "outer_problem": "self_consistent_optical_material_fixed_point",
            "nonlinear_material_iterations": "not_applicable",
            "td_rhs_residual_role": "unavailable_for_linearized_static_response",
            "material_response_calls": material_response_calls,
            "reference_operator_fft_counts_per_call": {
                "forward": latest_linearized_response.forward_fft_count,
                "inverse": latest_linearized_response.inverse_fft_count,
            },
            "mean_current_per_plane": total_mean_current.tolist(),
            "mean_intensity_perturbation_per_plane": np.asarray(
                asnumpy(latest_linearized_response.mean_intensity_perturbation)
            ).tolist(),
        })
    else:
        diagnostics["material_response"] = PR_MATERIAL_RESPONSE_NONLINEAR
        diagnostics["material_response_validation"] = "validated"
    scattering_provenance = None
    if request.scattering is not None:
        scattering_provenance = canonical_scattering_provenance(
            request.scattering,
            z_length_um=request.grid.z_length_um,
            Nx=grid.Nx,
            Ny=grid.Ny,
            x_aperture_um=request.grid.x_aperture_um,
            y_aperture_um=request.grid.y_aperture_um,
            real_dtype=grid.real_dtype,
            xp=xp,
        )
        diagnostics["canonical_scattering"] = scattering_provenance
    physics_profile_id = (
        PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
        else PR_FULL_TRANSVERSE_PROFILE_V1
    )
    resolved_profile = {
        "physics_profile_id": physics_profile_id,
        "workflow": PR_TRANSVERSE_STATIC_WORKFLOW,
        "transport_model": "full_transverse",
        "material_response": asdict(request.material_response),
        "requested_backend": request.backend.backend,
        "precision": request.backend.precision,
        "grid_request": asdict(request.grid),
        "beam_request": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "characteristic_wavenumber_per_um": k0,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
    }
    if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
        resolved_profile.update({
            "linearized_model_id": PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1,
            "electrical_ensemble": FIXED_MEAN_FIELD_ENSEMBLE,
            "reference_intensity": float(
                request.material_response.reference_intensity
            ),
            "applied_field": float(request.boundary.applied_field_x),
            "validation_status": "experimental",
        })
    if scattering_provenance is not None:
        resolved_profile["canonical_scattering"] = scattering_provenance
    timing = {
        "total_seconds": perf_counter() - started,
        "optical_pass_seconds": optical_seconds,
        "material_solve_seconds": material_seconds,
        "zero_flux_material_seconds": zero_flux_material_seconds,
        "linearized_material_seconds": linearized_material_seconds,
        "continuum_initializer_seconds": zero_flux_material_seconds,
        "discrete_corrector_seconds": 0.0,
    }
    status = "cancelled" if cancelled else ("converged" if converged else "not_converged")
    return PRTransverseStaticRunResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(replay_A)).copy(),
        psi_initial=psi_initial,
        psi_final=psi,
        source_intensity_stack=replay_source,
        equilibrium_residual_stack=authoritative_equilibrium,
        td_rhs_residual_stack=(
            None
            if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
            else td_residual
        ),
        power_initial=power_initial,
        power_final=power_final,
        converged=converged,
        completed_coupled_iterations=completed_iterations,
        iteration_records=tuple(records),
        material_iteration_records=tuple(material_records),
        discrete_iteration_records=tuple(discrete_records),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary=backend.summary(),
        resolved_profile=resolved_profile,
        replay_diagnostics=replay_diagnostics,
        diagnostics=diagnostics,
        timing=timing,
        status=status,
    )


def _continuation_stage_record(
    visibility: float,
    result: PRTransverseStaticRunResult,
) -> dict[str, Any]:
    return {
        "visibility": float(visibility),
        "status": result.status,
        "converged": bool(result.converged),
        "termination_reason": result.diagnostics["termination_reason"],
        "coupled_iterations": int(result.completed_coupled_iterations),
        "final_equilibrium_rms": float(
            result.diagnostics["equilibrium_residual_rms"]
        ),
        "final_equilibrium_max": float(
            result.diagnostics["equilibrium_residual_max"]
        ),
        "backtracks": int(sum(record.backtracks for record in result.iteration_records)),
        "material_newton_iterations": int(sum(
            record.material_newton_iterations for record in result.iteration_records
        )),
        "material_pcg_iterations": int(sum(
            record.material_pcg_iterations for record in result.iteration_records
        )),
    }


def _run_pr_transverse_static_visibility_continuation(
    request: PRTransverseStaticRunRequest,
    *,
    direct_result: PRTransverseStaticRunResult,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseStaticRunResult:
    """Restart canonically and advance the fixed visibility schedule."""

    continuation_started = perf_counter()
    stage_request = request
    stage_records: list[dict[str, Any]] = []
    timing_keys = (
        "total_seconds",
        "optical_pass_seconds",
        "material_solve_seconds",
        "zero_flux_material_seconds",
        "linearized_material_seconds",
        "continuum_initializer_seconds",
        "discrete_corrector_seconds",
    )
    accumulated_timing = {
        key: float(direct_result.timing[key]) for key in timing_keys
    }
    latest_stage: PRTransverseStaticRunResult | None = None
    last_attempted_visibility: float | None = None
    for visibility in PR_COHERENT_VISIBILITY_CONTINUATION_SCHEDULE:
        last_attempted_visibility = visibility
        stage_result = _run_pr_transverse_static_at_visibility(
            stage_request,
            visibility=visibility,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        latest_stage = stage_result
        stage_records.append(_continuation_stage_record(visibility, stage_result))
        for key in timing_keys:
            accumulated_timing[key] += float(stage_result.timing[key])
        if not stage_result.converged:
            returned = (
                stage_result
                if stage_result.status == "cancelled"
                else direct_result
            )
            break
        stage_request = replace(request, initial_psi=stage_result.psi_final)
    else:
        assert latest_stage is not None
        returned = latest_stage

    succeeded = bool(
        latest_stage is not None
        and latest_stage.converged
        and last_attempted_visibility == 1.0
    )
    diagnostics = dict(returned.diagnostics)
    continuation_cancelled = returned.status == "cancelled"
    diagnostics.update({
        "continuation_used": True,
        "visibility_schedule": PR_COHERENT_VISIBILITY_CONTINUATION_SCHEDULE,
        "continuation_stages": tuple(stage_records),
        "direct_attempt_status": direct_result.status,
        "direct_attempt_termination_reason": direct_result.diagnostics[
            "termination_reason"
        ],
        "requested_final_visibility": 1.0,
        "last_attempted_visibility": last_attempted_visibility,
        "final_visibility": 1.0 if succeeded else None,
        "continuation_succeeded": succeeded,
        "continuation_cancelled": continuation_cancelled,
        "continuation_cancellation_visibility": (
            last_attempted_visibility if continuation_cancelled else None
        ),
        "continuation_failure_visibility": (
            None
            if succeeded or continuation_cancelled
            else last_attempted_visibility
        ),
        "returned_state_source": (
            "final_full_visibility_stage"
            if succeeded
            else (
                "cancelled_continuation_stage"
                if continuation_cancelled
                else "direct_full_visibility_failure"
            )
        ),
    })
    timing = dict(returned.timing)
    for key in (
        "optical_pass_seconds",
        "material_solve_seconds",
        "zero_flux_material_seconds",
        "linearized_material_seconds",
        "continuum_initializer_seconds",
        "discrete_corrector_seconds",
    ):
        timing[key] = accumulated_timing[key]
    timing.update({
        "total_seconds": accumulated_timing["total_seconds"],
        "direct_attempt_seconds": float(direct_result.timing["total_seconds"]),
        "visibility_continuation_seconds": perf_counter() - continuation_started,
    })
    return replace(
        returned,
        psi_initial=direct_result.psi_initial,
        diagnostics=diagnostics,
        timing=timing,
    )


def run_pr_transverse_static(
    request: PRTransverseStaticRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseStaticRunResult:
    """Solve the Profile-v1 zero-flux problem, with bounded globalization.

    The exact requested coherent problem is always attempted first.  A
    canonical restart through the internal fixed visibility schedule occurs
    only for a coherent ``coupled_line_search_failed`` result.  Successful
    direct results are returned unchanged.
    """

    direct_result = _run_pr_transverse_static_at_visibility(
        request,
        visibility=1.0,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
    )
    if (
        direct_result.converged
        or direct_result.status == "cancelled"
        or direct_result.diagnostics["termination_reason"]
        != "coupled_line_search_failed"
        or not _contains_coherent_interference(request.beams)
    ):
        return direct_result
    return _run_pr_transverse_static_visibility_continuation(
        request,
        direct_result=direct_result,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
    )


__all__ = [
    "PR_TRANSVERSE_STATIC_WORKFLOW",
    "PR_COHERENT_VISIBILITY_CONTINUATION_SCHEDULE",
    "PRTransverseStaticCoupledRecord",
    "PRTransverseStaticDiscreteIterationRecord",
    "PRTransverseStaticMaterialIterationRecord",
    "PRTransverseStaticRunRequest",
    "PRTransverseStaticRunResult",
    "PRTransverseStaticWorkflowOptions",
    "run_pr_transverse_static",
]
