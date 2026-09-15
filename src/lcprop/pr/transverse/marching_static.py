"""Backend-neutral reference for causal zero-flux PR static propagation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from time import perf_counter
from typing import Any

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
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch, normalized_power
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.splitstep import advance_prepared_response, linear_kernel
from lcprop.pr.optical_response import half_step_response_from_E
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_provenance,
)
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseStaticMaterialSolverOptions,
    PRTransverseStaticNewtonRecord,
    _physical_state_validity,
    project_production_resolved_modes,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.workflow import (
    _apply_canonical_scattering_after_slice,
    _validate_canonical_scattering_for_grid,
)


PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW = "pr_transverse_marching_static_reference"
PR_MARCHING_FLOAT64_MATERIAL_COMPLEX64_OPTICS_V1 = (
    "float64_material_complex64_optics_v1"
)
PR_MARCHING_FULL_FLOAT64_REFERENCE_V1 = (
    "float64_material_complex128_optics_reference_v1"
)


@dataclass(frozen=True)
class _MarchingPrecisionPolicy:
    policy_id: str
    material_real_dtype: Any
    material_complex_dtype: Any
    optical_real_dtype: Any
    optical_complex_dtype: Any


def _resolve_precision_policy(backend) -> _MarchingPrecisionPolicy:
    if np.dtype(backend.real_dtype) == np.dtype(np.float32):
        return _MarchingPrecisionPolicy(
            policy_id=PR_MARCHING_FLOAT64_MATERIAL_COMPLEX64_OPTICS_V1,
            material_real_dtype=backend.xp.float64,
            material_complex_dtype=backend.xp.complex128,
            optical_real_dtype=backend.xp.float32,
            optical_complex_dtype=backend.xp.complex64,
        )
    return _MarchingPrecisionPolicy(
        policy_id=PR_MARCHING_FULL_FLOAT64_REFERENCE_V1,
        material_real_dtype=backend.xp.float64,
        material_complex_dtype=backend.xp.complex128,
        optical_real_dtype=backend.xp.float64,
        optical_complex_dtype=backend.xp.complex128,
    )


@dataclass(frozen=True)
class PRTransverseMarchingStaticOptions:
    """Reference-only controls for damped local Picard iteration."""

    material_solver: PRTransverseStaticMaterialSolverOptions = field(
        default_factory=PRTransverseStaticMaterialSolverOptions
    )
    max_local_iterations: int = 20
    equilibrium_rms_tolerance: float = 1.0e-8
    equilibrium_max_tolerance: float = 1.0e-7
    max_backtracks: int = 12
    minimum_step_scale: float = 2.0**-12
    armijo_fraction: float = 1.0e-4
    optical_substeps: int = 1
    record_iteration_history: bool = True
    record_material_stacks: bool = True
    record_residual_stack: bool = True
    record_optical_audit_fields: bool = True

    def validate(self) -> None:
        self.material_solver.validate()
        if int(self.max_local_iterations) < 1:
            raise ValueError("max_local_iterations must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "equilibrium_rms_tolerance",
            "equilibrium_max_tolerance",
            "minimum_step_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRTransverseMarchingStaticRunRequest:
    """Request for the additive backend-neutral causal-marching reference.

    ``initial_psi`` is an optional two-dimensional guess for the first
    interval only.  Every later interval is warm-started from the preceding
    accepted plane; no longitudinal interpolation is performed.
    """

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    transport: PRTransverseTransportProfile = PRTransverseTransportProfile()
    dielectric: PRTransverseDielectricProfile = PRTransverseDielectricProfile()
    boundary: PRTransverseBoundaryProfile = PRTransverseBoundaryProfile()
    projection: PRTransverseProjectionProfile = PRTransverseProjectionProfile()
    solver: PRTransverseMarchingStaticOptions = field(
        default_factory=PRTransverseMarchingStaticOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    initial_A: Any | None = None
    initial_psi: Any | None = None
    scattering: PRCanonicalScatteringSpec | None = None
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()


@dataclass(frozen=True)
class PRTransverseMarchingPicardRecord:
    """One accepted or rejected refreshed-source local Picard correction."""

    interval_index: int
    local_iteration: int
    equilibrium_rms_before: float
    equilibrium_max_before: float
    equilibrium_rms_after: float
    equilibrium_max_after: float
    potential_update_rms: float
    potential_update_max: float
    source_update_rms: float
    source_update_max: float
    step_scale: float
    backtracks: int
    material_newton_iterations: int
    material_pcg_iterations: int
    accepted: bool
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseMarchingIntervalSummary:
    """Final convergence and physical-state evidence for one interval."""

    interval_index: int
    z_midpoint_um: float
    converged: bool
    local_iterations: int
    backtracks: int
    accepted_damping: float
    equilibrium_rms: float
    equilibrium_max: float
    carrier_minimum: float
    carrier_mean_error: float
    potential_gauge_error: float
    potential_update_rms: float
    potential_update_max: float
    source_update_rms: float
    source_update_max: float
    material_newton_iterations: int
    material_pcg_iterations: int
    power_relative_drift: float
    interval_seconds: float
    termination_reason: str


@dataclass(frozen=True)
class PRTransverseMarchingStaticRunResult:
    """Host-NumPy result and audit evidence from the causal reference march."""

    A_initial: np.ndarray
    A_final: np.ndarray
    incoming_field_stack: np.ndarray
    pre_scattering_exit_stack: np.ndarray
    psi_accepted: np.ndarray
    source_intensity_stack: np.ndarray
    equilibrium_residual_stack: np.ndarray
    converged: bool
    start_interval: int
    completed_intervals: int
    accepted_intervals_this_run: int
    failed_interval: int | None
    interval_summaries: tuple[PRTransverseMarchingIntervalSummary, ...]
    iteration_records: tuple[PRTransverseMarchingPicardRecord, ...]
    power_initial: float
    power_final: float
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    resolved_profile: dict[str, Any]
    diagnostics: dict[str, Any]
    timing: dict[str, float]
    status: str
    restart_state: PRTransverseMarchingRestartState | None = None
    failure_state: PRTransverseMarchingFailureState | None = None


@dataclass(frozen=True)
class PRTransverseMarchingRestartState:
    """Host checkpoint sufficient to resume at one causal interval boundary."""

    request_fingerprint: str
    precision_policy_id: str
    next_interval_index: int
    A_incoming: np.ndarray
    previous_psi: np.ndarray | None
    entrance_intensity: np.ndarray
    previous_material_source: np.ndarray | None
    scattering_provenance: dict[str, Any] | None


@dataclass(frozen=True)
class PRTransverseMarchingFailureState:
    """Host snapshot of the exact failed local material/optical problem."""

    request_fingerprint: str
    precision_policy_id: str
    interval_index: int
    z_midpoint_um: float
    A_incoming: np.ndarray
    previous_psi: np.ndarray | None
    warm_start_psi: np.ndarray
    trial_psi: np.ndarray
    midpoint_source: np.ndarray
    pre_scattering_output: np.ndarray
    equilibrium_residual: np.ndarray
    iteration_records: tuple[PRTransverseMarchingPicardRecord, ...]
    material_iteration_records: tuple[PRTransverseStaticNewtonRecord, ...]
    scattering_provenance: dict[str, Any] | None


@dataclass(frozen=True)
class _LocalEvaluation:
    psi: Any
    A_pre_scattering: Any
    source: Any
    exit_intensity: Any
    residual: Any
    equilibrium_rms: float
    equilibrium_max: float
    physical_state_valid: bool
    carrier_mean: float
    carrier_minimum: float
    potential_mean: float


def _metrics(value, *, xp) -> tuple[float, float]:
    accumulator_dtype = (
        xp.float64 if value.dtype == np.dtype(np.float64) else xp.float32
    )
    work = value.astype(accumulator_dtype, copy=False)
    return (
        scalar_float(xp.sqrt(xp.mean(work * work))),
        scalar_float(xp.max(xp.abs(work))),
    )


def _difference_metrics(left, right, *, xp) -> tuple[float, float]:
    difference = left - right
    accumulator_dtype = (
        xp.float64 if difference.real.dtype == np.dtype(np.float64) else xp.float32
    )
    return (
        scalar_float(
            xp.sqrt(xp.mean(xp.abs(difference) ** 2, dtype=accumulator_dtype))
        ),
        scalar_float(xp.max(xp.abs(difference))),
    )


def _request_fingerprint(
    request: PRTransverseMarchingStaticRunRequest,
    *,
    precision_policy_id: str,
) -> str:
    def array_identity(value):
        if value is None:
            return None
        array = np.ascontiguousarray(asnumpy(value))
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        }

    payload = {
        "grid": asdict(request.grid),
        "beams": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "precision_policy_id": precision_policy_id,
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
        "initial_A": array_identity(request.initial_A),
        "initial_psi": array_identity(request.initial_psi),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _criteria_met(evaluation, options) -> bool:
    return (
        evaluation.physical_state_valid
        and evaluation.equilibrium_rms
        <= float(options.equilibrium_rms_tolerance)
        and evaluation.equilibrium_max
        <= float(options.equilibrium_max_tolerance)
    )


def _validate_request(request: PRTransverseMarchingStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.solver.validate()
    request.backend.validate()
    request.optical_boundary.validate()
    if request.scattering is not None:
        request.scattering.validate()
    if request.backend.backend not in ("numpy", "cupy"):
        raise ValueError(
            "marching static reference requires explicit backend='numpy' or 'cupy'"
        )
    if request.boundary.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
        raise ValueError(
            "marching static Profile v1 does not support the periodic biased "
            "electrical profile"
        )
    if float(request.material.applied_field) != 0.0:
        raise ValueError("Profile v1 requires material.applied_field=0")


def _advance_mixed_precision_slice(
    A_in,
    active_field,
    *,
    request,
    grid,
    kernel,
    peak_reference,
    intensity_before,
    wavelength_um,
    optical_complex_dtype,
    material_real_dtype,
    xp,
):
    """Advance one slice with the explicit Stage-22G K3 precision boundary."""

    resolved_substeps = int(request.solver.optical_substeps)
    half_response = half_step_response_from_E(
        active_field,
        dz_substep_um=grid.dz_um / resolved_substeps,
        wavelength_um=wavelength_um,
        interaction_length_um=request.grid.z_length_um,
        gain_length_product=request.material.gain_length_product,
        xp=xp,
    ).astype(optical_complex_dtype, copy=False)
    A_pre = A_in.copy()
    advance_prepared_response(
        A_pre,
        kernel=kernel,
        half_step_response=half_response,
        Nsub=resolved_substeps,
        boundary=request.optical_boundary,
        boundary_grid=grid,
        propagation_distance_um=grid.dz_um,
        xp=xp,
    )
    exit_intensity = pr_driving_intensity(
        A_pre,
        peak_intensity_reference=peak_reference,
        background_intensity=request.material.background_intensity,
        coherence_groups=request.beams.coherence_groups,
        xp=xp,
    )
    optical_source = 0.5 * (intensity_before + exit_intensity)
    material_source = optical_source.astype(material_real_dtype, copy=False)
    return A_pre, material_source, exit_intensity


def _evaluate_local_state(
    A_in,
    psi,
    *,
    request,
    grid,
    kernel,
    peak_reference,
    intensity_before,
    wavelength_um,
    dx_normalized,
    dy_normalized,
    material_real_dtype,
    optical_complex_dtype,
    xp,
) -> _LocalEvaluation:
    resolved = project_production_resolved_modes(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    state = state_from_potential(
        resolved,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
        xp=xp,
    )
    active = project_active_field(
        state.E_x, state.E_y, profile=request.projection, xp=xp
    )
    A_pre, source, exit_intensity = _advance_mixed_precision_slice(
        A_in,
        active,
        request=request,
        grid=grid,
        kernel=kernel,
        peak_reference=peak_reference,
        intensity_before=intensity_before,
        wavelength_um=wavelength_um,
        optical_complex_dtype=optical_complex_dtype,
        material_real_dtype=material_real_dtype,
        xp=xp,
    )
    residual = static_equilibrium_residual(
        resolved,
        source,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    resolved_residual = project_production_resolved_modes(
        residual,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    residual_rms, residual_max = _metrics(resolved_residual, xp=xp)
    valid, carrier_mean, carrier_minimum, potential_mean = (
        _physical_state_validity(state, xp=xp)
    )
    return _LocalEvaluation(
        psi=resolved,
        A_pre_scattering=A_pre,
        source=source,
        exit_intensity=exit_intensity,
        residual=residual,
        equilibrium_rms=residual_rms,
        equilibrium_max=residual_max,
        physical_state_valid=valid,
        carrier_mean=carrier_mean,
        carrier_minimum=carrier_minimum,
        potential_mean=potential_mean,
    )


def run_pr_transverse_static_marching(
    request: PRTransverseMarchingStaticRunRequest,
    *,
    restart_state: PRTransverseMarchingRestartState | None = None,
    stop_after_accepted_intervals: int | None = None,
) -> PRTransverseMarchingStaticRunResult:
    """March through locally implicit intervals and return host NumPy arrays."""

    started = perf_counter()
    _validate_request(request)
    backend = get_backend(request.backend)
    xp = backend.xp
    precision = _resolve_precision_policy(backend)
    grid = make_grid(
        request.grid, xp=xp, real_dtype=precision.optical_real_dtype
    )
    _validate_canonical_scattering_for_grid(
        request.scattering, grid=grid, z_length_um=request.grid.z_length_um
    )
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=precision.optical_complex_dtype,
        context=OpticalLaunchContext(
            grid=grid,
            n_ref=float(request.material.refractive_index),
            interaction_length_um=float(request.grid.z_length_um),
        ),
    )
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(
            request.initial_A, dtype=precision.optical_complex_dtype
        ).copy()
        valid_A = A0.shape == launch.A0.shape and bool(
            asnumpy(xp.all(xp.isfinite(A0)))
        )
        if not valid_A:
            raise ValueError("initial_A must be finite and match the launch shape")
    if request.initial_psi is None:
        first_guess = xp.zeros(
            (grid.Nx, grid.Ny), dtype=precision.material_real_dtype
        )
    else:
        first_guess = xp.asarray(
            request.initial_psi, dtype=precision.material_real_dtype
        ).copy()
        if first_guess.shape != (grid.Nx, grid.Ny):
            raise ValueError("initial_psi must have shape (Nx, Ny)")
        if not bool(asnumpy(xp.all(xp.isfinite(first_guess)))):
            raise ValueError("initial_psi must contain only finite values")

    wavelengths = tuple(
        float(channel.wavelength_um) for channel in request.beams.channels
    )
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("transverse PR workflow requires one shared wavelength")
    wavelength_um = wavelengths[0]
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    ).astype(precision.optical_complex_dtype, copy=False)
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    request_fingerprint = _request_fingerprint(
        request, precision_policy_id=precision.policy_id
    )
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

    if restart_state is None:
        start_interval = 0
        A_in = A0.copy()
        previous_psi = None
        previous_material_source = None
        intensity_before = pr_driving_intensity(
            A_in,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=xp,
        )
    else:
        if restart_state.request_fingerprint != request_fingerprint:
            raise ValueError("restart state does not match the marching request")
        if restart_state.precision_policy_id != precision.policy_id:
            raise ValueError(
                "restart precision policy does not match the marching request"
            )
        if restart_state.scattering_provenance != scattering_provenance:
            raise ValueError(
                "restart scattering provenance does not match the marching request"
            )
        start_interval = int(restart_state.next_interval_index)
        if not 0 <= start_interval < grid.Nz:
            raise ValueError("restart interval must lie inside the requested grid")
        if np.asarray(restart_state.A_incoming).dtype != np.dtype(
            precision.optical_complex_dtype
        ):
            raise ValueError("restart optical field has incompatible dtype")
        A_in = xp.asarray(restart_state.A_incoming).copy()
        previous_psi = (
            None
            if restart_state.previous_psi is None
            else xp.asarray(
                restart_state.previous_psi,
                dtype=precision.material_real_dtype,
            ).copy()
        )
        if (
            restart_state.previous_psi is not None
            and np.asarray(restart_state.previous_psi).dtype
            != np.dtype(precision.material_real_dtype)
        ):
            raise ValueError("restart material potential must remain float64")
        if np.asarray(restart_state.entrance_intensity).dtype != np.dtype(
            precision.optical_real_dtype
        ):
            raise ValueError("restart entrance intensity has incompatible dtype")
        intensity_before = xp.asarray(
            restart_state.entrance_intensity,
            dtype=precision.optical_real_dtype,
        ).copy()
        if (
            restart_state.previous_material_source is not None
            and np.asarray(restart_state.previous_material_source).dtype
            != np.dtype(precision.material_real_dtype)
        ):
            raise ValueError("restart material source must remain float64")
        previous_material_source = (
            None
            if restart_state.previous_material_source is None
            else xp.asarray(
                restart_state.previous_material_source,
                dtype=precision.material_real_dtype,
            ).copy()
        )
    if stop_after_accepted_intervals is not None:
        stop_index = int(stop_after_accepted_intervals)
        if not start_interval < stop_index <= grid.Nz:
            raise ValueError(
                "stop_after_accepted_intervals must be after the start and within Nz"
            )
    else:
        stop_index = None

    k0 = request.material.characteristic_wavenumber_per_um
    dx_normalized = k0 * grid.dx_um
    dy_normalized = k0 * grid.dy_um
    capacity = grid.Nz - start_interval
    material_shape = (capacity, grid.Nx, grid.Ny)
    field_shape = (
        capacity,
        len(request.beams.channels),
        grid.Nx,
        grid.Ny,
    )
    psi_buffer = (
        xp.empty(material_shape, dtype=precision.material_real_dtype)
        if request.solver.record_material_stacks
        else None
    )
    source_buffer = (
        xp.empty(material_shape, dtype=precision.material_real_dtype)
        if request.solver.record_material_stacks
        else None
    )
    residual_buffer = (
        xp.empty(material_shape, dtype=precision.material_real_dtype)
        if request.solver.record_residual_stack
        else None
    )
    incoming_buffer = (
        xp.empty(
            (capacity + 1, *field_shape[1:]),
            dtype=precision.optical_complex_dtype,
        )
        if request.solver.record_optical_audit_fields
        else None
    )
    pre_scattering_buffer = (
        xp.empty(field_shape, dtype=precision.optical_complex_dtype)
        if request.solver.record_optical_audit_fields
        else None
    )
    if incoming_buffer is not None:
        incoming_buffer[0] = A_in

    summaries: list[PRTransverseMarchingIntervalSummary] = []
    records: list[PRTransverseMarchingPicardRecord] = []
    failed_interval: int | None = None
    failure_evaluation: _LocalEvaluation | None = None
    failure_warm_start = None
    failure_material_records: tuple[PRTransverseStaticNewtonRecord, ...] = ()
    accepted_count = 0
    optical_seconds = 0.0
    material_seconds = 0.0
    stopped = False
    power_initial = normalized_power(A0, grid)

    def evaluate(A_value, psi_value):
        nonlocal optical_seconds
        synchronize(xp)
        evaluation_started = perf_counter()
        value = _evaluate_local_state(
            A_value,
            psi_value,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            intensity_before=intensity_before,
            wavelength_um=wavelength_um,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            material_real_dtype=precision.material_real_dtype,
            optical_complex_dtype=precision.optical_complex_dtype,
            xp=xp,
        )
        synchronize(xp)
        optical_seconds += perf_counter() - evaluation_started
        return value

    for interval_index in range(start_interval, grid.Nz):
        interval_started = perf_counter()
        warm_start = (
            first_guess.copy()
            if interval_index == 0 and previous_psi is None
            else previous_psi.copy()
        )
        psi = warm_start.copy()
        evaluation = None
        latest_trial = None
        last_update = (0.0, 0.0)
        last_source_update = (0.0, 0.0)
        total_backtracks = 0
        accepted_damping = 1.0
        material_newton = 0
        material_pcg = 0
        interval_material_records: list[PRTransverseStaticNewtonRecord] = []
        local_iterations = 0
        termination_reason = "maximum_local_iterations"

        for local_iteration in range(1, int(request.solver.max_local_iterations) + 1):
            local_iterations = local_iteration - 1
            if evaluation is None:
                evaluation = evaluate(A_in, psi)
                psi = evaluation.psi
            if _criteria_met(evaluation, request.solver):
                termination_reason = "residual_tolerance"
                break

            synchronize(xp)
            material_started = perf_counter()
            material = solve_pr_transverse_static_intensity(
                evaluation.source,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                initial_psi=psi,
                h_y=request.dielectric.h_y,
                options=request.solver.material_solver,
                xp=xp,
            )
            synchronize(xp)
            material_seconds += perf_counter() - material_started
            material_summary = material.plane_summaries[0]
            interval_material_records.extend(material.iteration_records)
            material_newton += material_summary.newton_iterations
            material_pcg += material_summary.pcg_iterations
            if not material.converged:
                termination_reason = f"material_{material.status}"
                break

            direction = material.psi - psi
            step_scale = 1.0
            trial_accepted = False
            candidate = evaluation
            backtracks = 0
            for backtracks in range(int(request.solver.max_backtracks) + 1):
                trial = evaluate(A_in, psi + step_scale * direction)
                latest_trial = trial
                target_rms = (
                    1.0 - float(request.solver.armijo_fraction) * step_scale
                ) * evaluation.equilibrium_rms
                if trial.physical_state_valid and (
                    _criteria_met(trial, request.solver)
                    or (
                        math.isfinite(trial.equilibrium_rms)
                        and math.isfinite(trial.equilibrium_max)
                        and trial.equilibrium_rms <= target_rms
                        and trial.equilibrium_max <= evaluation.equilibrium_max
                    )
                ):
                    trial_accepted = True
                    candidate = trial
                    break
                step_scale *= 0.5
                if step_scale < float(request.solver.minimum_step_scale):
                    break

            total_backtracks += backtracks
            potential_update = _difference_metrics(candidate.psi, psi, xp=xp)
            source_update = _difference_metrics(
                candidate.source, evaluation.source, xp=xp
            )
            if request.solver.record_iteration_history:
                records.append(PRTransverseMarchingPicardRecord(
                    interval_index=interval_index,
                    local_iteration=local_iteration,
                    equilibrium_rms_before=evaluation.equilibrium_rms,
                    equilibrium_max_before=evaluation.equilibrium_max,
                    equilibrium_rms_after=candidate.equilibrium_rms,
                    equilibrium_max_after=candidate.equilibrium_max,
                    potential_update_rms=potential_update[0],
                    potential_update_max=potential_update[1],
                    source_update_rms=source_update[0],
                    source_update_max=source_update[1],
                    step_scale=step_scale if trial_accepted else 0.0,
                    backtracks=backtracks,
                    material_newton_iterations=material_summary.newton_iterations,
                    material_pcg_iterations=material_summary.pcg_iterations,
                    accepted=trial_accepted,
                    termination_reason=(
                        "accepted" if trial_accepted else "local_line_search_failed"
                    ),
                ))
            if not trial_accepted:
                termination_reason = "local_line_search_failed"
                break
            psi = candidate.psi
            evaluation = candidate
            last_update = potential_update
            last_source_update = source_update
            accepted_damping = step_scale
            local_iterations = local_iteration
            if _criteria_met(evaluation, request.solver):
                termination_reason = "residual_tolerance"
                break

        assert evaluation is not None
        converged_interval = (
            termination_reason == "residual_tolerance"
            and _criteria_met(evaluation, request.solver)
        )
        current_power = normalized_power(
            evaluation.A_pre_scattering if converged_interval else A_in, grid
        )
        summaries.append(PRTransverseMarchingIntervalSummary(
            interval_index=interval_index,
            z_midpoint_um=(interval_index + 0.5) * grid.dz_um,
            converged=converged_interval,
            local_iterations=local_iterations,
            backtracks=total_backtracks,
            accepted_damping=accepted_damping,
            equilibrium_rms=evaluation.equilibrium_rms,
            equilibrium_max=evaluation.equilibrium_max,
            carrier_minimum=evaluation.carrier_minimum,
            carrier_mean_error=abs(evaluation.carrier_mean - 1.0),
            potential_gauge_error=abs(evaluation.potential_mean),
            potential_update_rms=last_update[0],
            potential_update_max=last_update[1],
            source_update_rms=last_source_update[0],
            source_update_max=last_source_update[1],
            material_newton_iterations=material_newton,
            material_pcg_iterations=material_pcg,
            power_relative_drift=(current_power - power_initial) / power_initial,
            interval_seconds=perf_counter() - interval_started,
            termination_reason=termination_reason,
        ))
        if not converged_interval:
            failed_interval = interval_index
            failure_evaluation = latest_trial or evaluation
            failure_warm_start = warm_start
            failure_material_records = tuple(interval_material_records)
            break

        if psi_buffer is not None:
            psi_buffer[accepted_count] = evaluation.psi
            source_buffer[accepted_count] = evaluation.source
        if residual_buffer is not None:
            residual_buffer[accepted_count] = evaluation.residual
        if pre_scattering_buffer is not None:
            pre_scattering_buffer[accepted_count] = evaluation.A_pre_scattering
        previous_psi = evaluation.psi.copy()
        previous_material_source = evaluation.source.copy()
        A_in = evaluation.A_pre_scattering.copy()
        _apply_canonical_scattering_after_slice(
            A_in,
            scattering=request.scattering,
            z_index=interval_index,
            grid=grid,
            z_length_um=request.grid.z_length_um,
            xp=xp,
        )
        accepted_count += 1
        if incoming_buffer is not None:
            incoming_buffer[accepted_count] = A_in
        # Canonical scattering is phase-only, so the pre-scattering exit
        # intensity is exactly the next interval's entrance intensity.
        intensity_before = evaluation.exit_intensity
        if stop_index is not None and interval_index + 1 == stop_index:
            stopped = True
            break

    completed = start_interval + accepted_count
    converged = completed == grid.Nz and failed_interval is None
    status = (
        "converged"
        if converged
        else ("stopped_at_accepted_boundary" if stopped else "not_converged")
    )
    power_final = normalized_power(A_in, grid)

    def host_stack(buffer, count, *, dtype, trailing_shape):
        if buffer is None or count == 0:
            return np.empty((0, *trailing_shape), dtype=dtype)
        return np.asarray(asnumpy(buffer[:count])).copy()

    psi_stack = host_stack(
        psi_buffer,
        accepted_count,
        dtype=np.dtype(precision.material_real_dtype),
        trailing_shape=(grid.Nx, grid.Ny),
    )
    source_stack = host_stack(
        source_buffer,
        accepted_count,
        dtype=np.dtype(precision.material_real_dtype),
        trailing_shape=(grid.Nx, grid.Ny),
    )
    residual_stack = host_stack(
        residual_buffer,
        accepted_count,
        dtype=np.dtype(precision.material_real_dtype),
        trailing_shape=(grid.Nx, grid.Ny),
    )
    pre_stack = host_stack(
        pre_scattering_buffer,
        accepted_count,
        dtype=np.dtype(precision.optical_complex_dtype),
        trailing_shape=field_shape[1:],
    )
    incoming_stack = host_stack(
        incoming_buffer,
        accepted_count + 1 if incoming_buffer is not None else 0,
        dtype=np.dtype(precision.optical_complex_dtype),
        trailing_shape=field_shape[1:],
    )

    restart_output = None
    if not converged:
        restart_output = PRTransverseMarchingRestartState(
            request_fingerprint=request_fingerprint,
            precision_policy_id=precision.policy_id,
            next_interval_index=(
                failed_interval if failed_interval is not None else completed
            ),
            A_incoming=np.asarray(asnumpy(A_in)).copy(),
            previous_psi=(
                None
                if previous_psi is None
                else np.asarray(asnumpy(previous_psi)).copy()
            ),
            entrance_intensity=np.asarray(asnumpy(intensity_before)).copy(),
            previous_material_source=(
                None
                if previous_material_source is None
                else np.asarray(asnumpy(previous_material_source)).copy()
            ),
            scattering_provenance=scattering_provenance,
        )
    failure_output = None
    if failure_evaluation is not None and failed_interval is not None:
        failure_records = tuple(
            record for record in records if record.interval_index == failed_interval
        )
        failure_output = PRTransverseMarchingFailureState(
            request_fingerprint=request_fingerprint,
            precision_policy_id=precision.policy_id,
            interval_index=failed_interval,
            z_midpoint_um=(failed_interval + 0.5) * grid.dz_um,
            A_incoming=np.asarray(asnumpy(A_in)).copy(),
            previous_psi=(
                None
                if previous_psi is None
                else np.asarray(asnumpy(previous_psi)).copy()
            ),
            warm_start_psi=np.asarray(asnumpy(failure_warm_start)).copy(),
            trial_psi=np.asarray(asnumpy(failure_evaluation.psi)).copy(),
            midpoint_source=np.asarray(asnumpy(failure_evaluation.source)).copy(),
            pre_scattering_output=np.asarray(
                asnumpy(failure_evaluation.A_pre_scattering)
            ).copy(),
            equilibrium_residual=np.asarray(
                asnumpy(failure_evaluation.residual)
            ).copy(),
            iteration_records=failure_records,
            material_iteration_records=failure_material_records,
            scattering_provenance=scattering_provenance,
        )

    resolved_profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
        "workflow": PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW,
        "grid_request": asdict(request.grid),
        "beam_request": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "state_location": "interval_midpoint_piecewise_constant",
        "result_array_boundary": "host_numpy",
        "request_fingerprint": request_fingerprint,
        "precision_policy": {
            "policy_id": precision.policy_id,
            "material_real_dtype": str(np.dtype(precision.material_real_dtype)),
            "material_complex_dtype": str(
                np.dtype(precision.material_complex_dtype)
            ),
            "optical_real_dtype": str(np.dtype(precision.optical_real_dtype)),
            "optical_complex_dtype": str(
                np.dtype(precision.optical_complex_dtype)
            ),
            "material_to_optics": (
                "complex128_half_screen_explicitly_cast_to_optical_complex"
            ),
            "optics_to_material": "midpoint_source_explicitly_promoted_to_float64",
        },
        "characteristic_wavenumber_per_um": k0,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
    }
    if scattering_provenance is not None:
        resolved_profile["canonical_scattering"] = scattering_provenance
    diagnostics = {
        "termination_reason": (
            "all_intervals_converged"
            if converged
            else (
                "stopped_at_accepted_boundary"
                if stopped
                else summaries[-1].termination_reason
            )
        ),
        "failed_interval": failed_interval,
        "accepted_upstream_intervals_are_final": True,
        "authoritative_static_residual": "zero_flux_equilibrium",
        "midpoint_source": "arithmetic_entrance_pre_scattering_exit",
        "scattering_placement": "after_local_acceptance",
        "precision_policy_id": precision.policy_id,
        "optical_power_relative_drift": (
            (power_final - power_initial) / power_initial
        ),
    }
    if scattering_provenance is not None:
        diagnostics["canonical_scattering"] = scattering_provenance
    synchronize(xp)
    elapsed = perf_counter() - started
    return PRTransverseMarchingStaticRunResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(A_in)).copy(),
        incoming_field_stack=incoming_stack,
        pre_scattering_exit_stack=pre_stack,
        psi_accepted=psi_stack,
        source_intensity_stack=source_stack,
        equilibrium_residual_stack=residual_stack,
        converged=converged,
        start_interval=start_interval,
        completed_intervals=completed,
        accepted_intervals_this_run=accepted_count,
        failed_interval=failed_interval,
        interval_summaries=tuple(summaries),
        iteration_records=tuple(records),
        power_initial=power_initial,
        power_final=power_final,
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary={
            **backend.summary(),
            "precision_policy_id": precision.policy_id,
            "material_real_dtype": str(np.dtype(precision.material_real_dtype)),
            "material_complex_dtype": str(
                np.dtype(precision.material_complex_dtype)
            ),
            "optical_real_dtype": str(np.dtype(precision.optical_real_dtype)),
            "optical_complex_dtype": str(
                np.dtype(precision.optical_complex_dtype)
            ),
        },
        resolved_profile=resolved_profile,
        diagnostics=diagnostics,
        timing={
            "total_seconds": elapsed,
            "optical_slice_seconds": optical_seconds,
            "material_solve_seconds": material_seconds,
        },
        status=status,
        restart_state=restart_output,
        failure_state=failure_output,
    )


__all__ = [
    "PR_MARCHING_FLOAT64_MATERIAL_COMPLEX64_OPTICS_V1",
    "PR_MARCHING_FULL_FLOAT64_REFERENCE_V1",
    "PR_TRANSVERSE_MARCHING_STATIC_WORKFLOW",
    "PRTransverseMarchingIntervalSummary",
    "PRTransverseMarchingFailureState",
    "PRTransverseMarchingPicardRecord",
    "PRTransverseMarchingRestartState",
    "PRTransverseMarchingStaticOptions",
    "PRTransverseMarchingStaticRunRequest",
    "PRTransverseMarchingStaticRunResult",
    "run_pr_transverse_static_marching",
]
