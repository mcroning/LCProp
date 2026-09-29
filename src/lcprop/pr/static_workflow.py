"""Self-consistent static photorefractive workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy, get_backend
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
from lcprop.optics.splitstep import scalar_angular_spectrum_kernel
from lcprop.pr.carrier_power import carrier_channels_from_beams
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.reduced_linearized import (
    PR_REDUCED_LINEARIZED_RESPONSE_V1,
    PRReducedLinearizedSpec,
    reduced_linearized_residual,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import (
    PRStaticSolverOptions,
    solve_pr_static_intensity_batched,
)
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec, canonical_scattering_provenance
from lcprop.pr.workflow import (
    advance_pr_slice_with_midpoint_source,
    _apply_canonical_scattering_after_slice,
    _canonical_scattering_phase_for_slice,
    _validate_canonical_scattering_for_grid,
)


PR_STATIC_WORKFLOW = "pr_static"


ProgressCallback = Callable[[RunProgress], None]


@dataclass(frozen=True)
class PRStaticWorkflowOptions:
    """Controls for the coupled static workflow.

    ``None`` selects documented precision-aware defaults for the material,
    coupled-residual, and replay tolerances. Explicit values are preserved
    exactly.
    """

    material_solver: PRStaticSolverOptions | None = None
    max_coupled_passes: int = 20
    residual_rms_tolerance: float | None = None
    residual_max_tolerance: float | None = None
    max_backtracks: int = 16
    minimum_step_scale: float = 2.0**-16
    armijo_fraction: float = 1e-4
    optical_substeps: int = 1
    replay_rtol: float | None = None
    replay_atol: float | None = None
    record_iteration_history: bool = True

    def validate(self) -> None:
        if self.material_solver is not None:
            self.material_solver.validate()
        if int(self.max_coupled_passes) < 1:
            raise ValueError("max_coupled_passes must be at least one")
        if int(self.max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if int(self.optical_substeps) < 1:
            raise ValueError("optical_substeps must be at least one")
        for name in (
            "residual_rms_tolerance",
            "residual_max_tolerance",
            "minimum_step_scale",
            "replay_rtol",
            "replay_atol",
        ):
            supplied = getattr(self, name)
            if supplied is None:
                continue
            value = float(supplied)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        fraction = float(self.armijo_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("armijo_fraction must be between zero and one")


@dataclass(frozen=True)
class PRStaticRunRequest:
    """Request for a headless self-consistent static PR calculation.

    Declarative ``launch_elements`` transform normalized incident beams before
    propagation and cannot be combined with an explicit ``initial_A``.
    """

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    solver: PRStaticWorkflowOptions = field(
        default_factory=PRStaticWorkflowOptions
    )
    backend: BackendSpec = BackendSpec(
        backend="numpy",
        precision="float64",
        verbose=False,
    )
    launch_elements: tuple[ChannelLaunchElements, ...] = ()
    initial_A: Any | None = None
    initial_E: Any | None = None
    material_response: PRTransverseMaterialResponseSpec = field(
        default_factory=PRTransverseMaterialResponseSpec
    )
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()
    scattering: PRCanonicalScatteringSpec | None = None


@dataclass(frozen=True)
class PRCoupledStaticIterationRecord:
    """One coupled material/optical correction at a longitudinal slice."""

    z_index: int
    coupled_pass: int
    residual_before_rms: float
    residual_before_max: float
    residual_after_rms: float
    residual_after_max: float
    delta_E_rms: float
    delta_E_max: float
    step_scale: float
    material_status: str
    accepted: bool


@dataclass(frozen=True)
class PRCoupledStaticSliceSummary:
    """Final convergence evidence for one longitudinal slice."""

    z_index: int
    z_um: float
    coupled_passes: int
    final_residual_rms: float
    final_residual_max: float
    final_delta_E_rms: float
    final_delta_E_max: float
    converged: bool
    termination_reason: str


@dataclass(frozen=True)
class PRStaticRunResult:
    """Static PR state, optical fields, and strict replay diagnostics."""

    A_initial: np.ndarray
    A_final: np.ndarray
    E_initial: np.ndarray | None
    E_final: np.ndarray | None
    source_intensity_stack: np.ndarray | None
    residual_stack: np.ndarray | None
    power_initial: float
    power_final: float
    converged: bool
    completed_slices: int
    iteration_records: tuple[PRCoupledStaticIterationRecord, ...]
    slice_summaries: tuple[PRCoupledStaticSliceSummary, ...]
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    backend_summary: dict[str, Any]
    tolerance_provenance: dict[str, Any]
    replay_diagnostics: dict[str, Any]
    status: str
    retention_summary: dict[str, Any] = field(
        default_factory=lambda: {"policy": "full", "omitted_fields": []}
    )
    material_response_summary: dict[str, Any] = field(
        default_factory=lambda: {
            "model": PR_MATERIAL_RESPONSE_NONLINEAR,
            "validation_status": "validated",
        }
    )
    longitudinal_intensity_xz: np.ndarray | None = None
    longitudinal_intensity_yz: np.ndarray | None = None
    x_cut_um: float | None = None
    y_cut_um: float | None = None
    intensity_preview: np.ndarray | None = None
    intensity_preview_metadata: dict[str, Any] | None = None


def _backend_scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


@dataclass(frozen=True)
class _ResolvedStaticTolerances:
    precision: str
    material_solver: PRStaticSolverOptions
    material_source: str
    residual_rms_tolerance: float
    residual_rms_source: str
    residual_max_tolerance: float
    residual_max_source: str
    replay_rtol: float
    replay_rtol_source: str
    replay_atol: float
    replay_atol_source: str

    def provenance(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "material_solver": {
                "source": self.material_source,
                "residual_rms_tolerance": float(
                    self.material_solver.residual_rms_tolerance
                ),
                "residual_max_tolerance": float(
                    self.material_solver.residual_max_tolerance
                ),
            },
            "coupled_residual_rms_tolerance": {
                "value": self.residual_rms_tolerance,
                "source": self.residual_rms_source,
            },
            "coupled_residual_max_tolerance": {
                "value": self.residual_max_tolerance,
                "source": self.residual_max_source,
            },
            "replay_rtol": {
                "value": self.replay_rtol,
                "source": self.replay_rtol_source,
            },
            "replay_atol": {
                "value": self.replay_atol,
                "source": self.replay_atol_source,
            },
        }


def _resolved_value(value, default: float) -> tuple[float, str]:
    if value is None:
        return float(default), "precision_default"
    return float(value), "explicit_override"


def _resolve_static_tolerances(
    options: PRStaticWorkflowOptions,
    *,
    real_dtype: Any,
    linearized: bool = False,
) -> _ResolvedStaticTolerances:
    precision = str(np.dtype(real_dtype))
    if precision == "float64":
        material_rms = 1e-10
        material_max = 1e-9
        coupled_rms = 1e-8
        coupled_max = 1e-7
        replay_rtol = 1e-11
        replay_atol = 1e-12
    elif precision == "float32":
        material_rms = 2e-6
        material_max = 1e-5
        coupled_rms = 5e-6 if linearized else 2e-6
        coupled_max = 2e-5 if linearized else 1e-5
        replay_rtol = 2e-6
        replay_atol = 2e-7
    else:  # pragma: no cover - backend precision validation owns this case.
        raise ValueError(f"unsupported PR static precision {precision!r}")

    if options.material_solver is None:
        material_solver = PRStaticSolverOptions(
            residual_rms_tolerance=material_rms,
            residual_max_tolerance=material_max,
        )
        material_source = "precision_default"
    else:
        material_solver = options.material_solver
        material_source = "explicit_PRStaticSolverOptions"

    resolved_rms, resolved_rms_source = _resolved_value(
        options.residual_rms_tolerance,
        coupled_rms,
    )
    resolved_max, resolved_max_source = _resolved_value(
        options.residual_max_tolerance,
        coupled_max,
    )
    resolved_replay_rtol, replay_rtol_source = _resolved_value(
        options.replay_rtol,
        replay_rtol,
    )
    resolved_replay_atol, replay_atol_source = _resolved_value(
        options.replay_atol,
        replay_atol,
    )
    return _ResolvedStaticTolerances(
        precision=precision,
        material_solver=material_solver,
        material_source=material_source,
        residual_rms_tolerance=resolved_rms,
        residual_rms_source=resolved_rms_source,
        residual_max_tolerance=resolved_max,
        residual_max_source=resolved_max_source,
        replay_rtol=resolved_replay_rtol,
        replay_rtol_source=replay_rtol_source,
        replay_atol=resolved_replay_atol,
        replay_atol_source=replay_atol_source,
    )


def _array_has_true(value, *, xp: Any) -> bool:
    reduced = xp.any(value)
    return bool(reduced.item() if hasattr(reduced, "item") else reduced)


def _residual_metrics(residual, *, xp: Any) -> tuple[float, float]:
    return (
        _backend_scalar(xp.sqrt(xp.mean(residual * residual))),
        _backend_scalar(xp.max(xp.abs(residual))),
    )


# Diagnostic work arrays are capped at 2 MiB per float64 chunk, independently
# of Nz/Nx/Ny. Slicing (including noncontiguous inputs) never flattens/copies a
# complete volume. Physics/trial reductions intentionally retain their old path.
_COMPLETION_CHUNK_ELEMENTS = 262144


def _completion_slices(shape):
    def split(bounds):
        sizes = [stop - start for start, stop in bounds]
        if math.prod(sizes) <= _COMPLETION_CHUNK_ELEMENTS:
            yield tuple(slice(start, stop) for start, stop in bounds)
            return
        axis = next(i for i, size in enumerate(sizes) if size > 1)
        start, stop = bounds[axis]
        mid = (start + stop) // 2
        for interval in ((start, mid), (mid, stop)):
            child = list(bounds)
            child[axis] = interval
            yield from split(child)
    yield from split([(0, int(size)) for size in shape])


# Finite binary64 squares occupy bits 0..4195 in units of 2**-2148.
# 264 base-2**16 bins include the final shifted digit (no sample export).
_EXACT_SQUARE_BINS = 264


def _exact_square_summary(chunk, *, xp):
    """Exact sum-of-squares digits, reduced on the array's backend.

    Each weighted bincount sums nonnegative 16-bit integers. With <=2**18
    samples, each bin sum is <2**34; summing the fourteen histograms stays
    <2**38, hence every float64 histogram addition is exact. Integer products
    of the four 16-bit significand limbs fit uint64 before carry propagation.
    """
    bits = xp.asarray(chunk, dtype=xp.float64).view(xp.uint64)
    exponent = (bits >> 52) & 2047
    mantissa = (bits & ((1 << 52) - 1)) | ((exponent != 0).astype(xp.uint64) << 52)
    # x = mantissa * 2**-1074 for subnormals; otherwise multiply by
    # 2**(exponent-1). Squaring doubles that nonnegative shift.
    shift = 2 * xp.maximum(exponent.astype(xp.int64) - 1, 0)
    bins = shift // 16
    remainder = (shift % 16).astype(xp.uint64)
    limbs = [(mantissa >> (16 * i)) & 65535 for i in range(4)]
    summary = xp.zeros(_EXACT_SQUARE_BINS, dtype=xp.float64)
    carry = xp.zeros_like(mantissa)
    for digit in range(7):
        coefficient = carry.copy()
        for i in range(max(0, digit - 3), min(3, digit) + 1):
            coefficient += limbs[i] * limbs[digit - i]
        carry = coefficient >> 16
        shifted = (coefficient & 65535) << remainder
        # ravel may copy a strided CHUNK, never the residual volume.
        for offset, weights in ((0, shifted & 65535), (1, shifted >> 16)):
            summary += xp.bincount(
                (bins + digit + offset).ravel(),
                weights=weights.ravel().astype(xp.float64),
                minlength=_EXACT_SQUARE_BINS,
            )
    return summary


def _exact_rms_within(residual, tolerance, *, xp):
    numerator, denominator = float(tolerance).as_integer_ratio()
    limit = (numerator * numerator * int(residual.size)) << (
        2148 - 2 * (denominator.bit_length() - 1)
    )
    total = 0
    for index in _completion_slices(residual.shape):
        summary = np.asarray(asnumpy(_exact_square_summary(residual[index], xp=xp)))
        # Fixed 264-bin Python work per chunk, independent of sample count.
        total += sum(int(value) << (16 * i) for i, value in enumerate(summary))
        if total > limit:
            return False
    return total <= limit


def _completion_residual_metrics(residual, *, xp, rms_tolerance):
    """Scaled float64 RMS with exact backend-reduced boundary decisions.

    Chunk sums use backend float64 reduction; math.fsum combines scalar sums.
    A conservative linear-summation forward-error envelope identifies boundary
    cases, resolved by exact binary64 square histograms, never sample export.
    """
    maximum = 0.0
    for index in _completion_slices(residual.shape):
        maximum = float(np.maximum(maximum, _backend_scalar(
            xp.max(xp.abs(residual[index])))))
    if not math.isfinite(maximum):
        return maximum, maximum, False
    if maximum == 0.0:
        return 0.0, 0.0, 0.0 <= rms_tolerance

    def sums():
        for index in _completion_slices(residual.shape):
            scaled = xp.asarray(residual[index], dtype=xp.float64) / maximum
            yield _backend_scalar(xp.sum(scaled * scaled, dtype=xp.float64))
    rms = maximum * math.sqrt(min(1.0, math.fsum(sums()) / residual.size))
    within = rms <= rms_tolerance
    # Division, square, sum, final division/sqrt: deliberately conservative.
    error = (8 * (_COMPLETION_CHUNK_ELEMENTS + 8)
             * np.finfo(np.float64).eps * max(rms, rms_tolerance))
    if abs(rms - rms_tolerance) <= error:
        within = _exact_rms_within(residual, rms_tolerance, xp=xp)
    return rms, maximum, within


def _completion_comparison(actual, expected, *, xp, rtol, atol):
    """Preserve elementwise isclose semantics, including NaN/Inf behavior."""
    maximum = 0.0
    consistent = True
    for index in _completion_slices(actual.shape):
        a, b = actual[index], expected[index]
        maximum = float(np.maximum(maximum, _backend_scalar(
            xp.max(xp.abs(a - b)))))
        close = bool(_backend_scalar(xp.all(xp.isclose(a, b, rtol=rtol, atol=atol))))
        consistent = consistent and close
    return maximum, consistent


def _owned_host_result(value):
    # CuPy conversion already owns its new host allocation. NumPy still needs
    # a copy so the returned scientific arrays do not alias solver inputs.
    if isinstance(value, np.ndarray):
        return value.copy()
    return np.asarray(asnumpy(value))


def _update_metrics(new_state, old_state, *, xp: Any) -> tuple[float, float]:
    delta = new_state - old_state
    return (
        _backend_scalar(xp.sqrt(xp.mean(delta * delta))),
        _backend_scalar(xp.max(xp.abs(delta))),
    )


def _criteria_met(
    residual_rms: float,
    residual_max: float,
    tolerances: _ResolvedStaticTolerances,
) -> bool:
    return (
        residual_rms <= tolerances.residual_rms_tolerance
        and residual_max <= tolerances.residual_max_tolerance
    )


def run_pr_static(
    request: PRStaticRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRStaticRunResult:
    """Solve the self-consistent static PR problem by a local coupled z-march.

    At each slice, the accepted incoming optical field remains fixed while a
    prescribed-intensity solve supplies a block material correction (Newton
    for nonlinear response, direct Fourier response for linearized response).
    Every trial correction is checked against a freshly propagated midpoint
    intensity. A final independent replay verifies the assembled state.
    """

    started_at = perf_counter()
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.material_response.validate()
    request.solver.validate()
    request.backend.validate()
    request.optical_boundary.validate()
    _validate_canonical_scattering_for_grid(
        request.scattering, grid=request.grid, z_length_um=request.grid.z_length_um
    )
    LaunchConfiguration(request.beams, request.launch_elements)
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    wavelengths = tuple(
        float(channel.wavelength_um) for channel in request.beams.channels
    )
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("minimal PR workflow requires one shared wavelength")

    linearized = (
        request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
    )
    backend = get_backend(request.backend)
    xp = backend.xp
    tolerances = _resolve_static_tolerances(
        request.solver,
        real_dtype=backend.real_dtype,
        linearized=linearized,
    )
    grid = make_grid(request.grid, xp=backend.xp, real_dtype=backend.real_dtype)
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
    dx_normalized = (
        request.material.characteristic_wavenumber_per_um * grid.dx_um
    )
    linearized_spec = None
    if linearized:
        linearized_spec = PRReducedLinearizedSpec(
            reference_intensity=float(
                request.material_response.reference_intensity
            ),
            applied_field=float(request.material.applied_field),
            background_intensity=float(request.material.background_intensity),
            dx_normalized=dx_normalized,
        )
        linearized_spec.validate()
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=backend.complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(
                f"initial_A shape {A0.shape} does not match {launch.A0.shape}"
            )

    E_shape = (grid.Nz, grid.Nx, grid.Ny)
    if request.initial_E is None:
        E_initial = xp.zeros(E_shape, dtype=backend.real_dtype)
        if linearized:
            E_initial.fill(linearized_spec.equilibrium_field)
    else:
        E_initial = xp.asarray(
            request.initial_E,
            dtype=backend.real_dtype,
        ).copy()
        if E_initial.shape != E_shape:
            raise ValueError(
                f"initial_E shape {E_initial.shape} does not match {E_shape}"
            )
        if _array_has_true(~xp.isfinite(E_initial), xp=xp):
            raise ValueError("initial_E must contain only finite values")

    wavelength_um = wavelengths[0]
    kernel = scalar_angular_spectrum_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
        complex_dtype=backend.complex_dtype,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    groups = request.beams.coherence_groups
    E_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    source_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    residual_stack = xp.empty(E_shape, dtype=backend.real_dtype)
    A = A0.copy()
    records: list[PRCoupledStaticIterationRecord] = []
    summaries: list[PRCoupledStaticSliceSummary] = []
    completed_slices = 0
    cancelled = False
    material_response_calls = 0

    # One phase plane only: reuse within all trials of a slice; regenerate
    # deterministically by coordinate for the final replay, never by trial count.
    cached_z_index = None
    cached_phase = None

    def advance_slice(A_in, state, z_index):
        nonlocal cached_z_index, cached_phase
        if request.scattering is not None and cached_z_index != z_index:
            cached_phase = None
            cached_phase = _canonical_scattering_phase_for_slice(
                request.scattering, z_index=z_index, grid=grid,
                z_length_um=request.grid.z_length_um, xp=xp,
            )
            cached_z_index = z_index
        A_out, source = advance_pr_slice_with_midpoint_source(
            A_in,
            state,
            kernel=kernel,
            optical_substeps=request.solver.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=groups,
            xp=xp,
            optical_boundary=request.optical_boundary,
            boundary_grid=grid,
        )
        _apply_canonical_scattering_after_slice(
            A_out, scattering=request.scattering, z_index=z_index, grid=grid,
            z_length_um=request.grid.z_length_um, xp=xp, phase=cached_phase,
        )
        return A_out, source

    def residual_at(state, intensity):
        if linearized:
            return reduced_linearized_residual(
                state,
                intensity,
                spec=linearized_spec,
                xp=xp,
            )
        return hopping_rhs(
            state,
            intensity,
            applied_field=request.material.applied_field,
            background_intensity=request.material.background_intensity,
            dx_normalized=dx_normalized,
            xp=xp,
        )

    def report_completion_phase(
        phase: str,
        *,
        completed_units: int,
        total_units: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            RunProgress(
                workflow=PR_STATIC_WORKFLOW,
                status="running",
                completed_units=int(completed_units),
                total_units=int(total_units),
                current_coordinate=float(completed_units),
                coordinate_name="replay_slice",
                coordinate_unit="1",
                elapsed_wall_time=perf_counter() - started_at,
                message=message,
                diagnostics={"phase": phase},
            )
        )

    for k in range(grid.Nz):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            cancelled = True
            break

        A_slice_in = A.copy()
        if request.initial_E is not None:
            state = E_initial[k].copy()
        elif k == 0:
            state = E_initial[0].copy()
        else:
            state = E_stack[k - 1].copy()

        A_trial, intensity = advance_slice(A_slice_in, state, k)
        residual = residual_at(state, intensity)
        residual_rms, residual_max = _residual_metrics(residual, xp=xp)
        converged = _criteria_met(residual_rms, residual_max, tolerances)
        termination_reason = (
            "residual_tolerance" if converged else "maximum_coupled_passes"
        )
        coupled_passes = 0
        final_delta_rms = 0.0
        final_delta_max = 0.0

        for coupled_pass in range(1, int(request.solver.max_coupled_passes) + 1):
            if converged:
                break
            coupled_passes = coupled_pass
            if linearized:
                material_result = solve_pr_reduced_linearized_intensity(
                    intensity,
                    spec=linearized_spec,
                    backend=request.backend,
                )
                material_response_calls += 1
                material_converged = True
                material_status = "analytic_fourier_solve"
                material_state = material_result.E
            else:
                material_result = solve_pr_static_intensity_batched(
                    intensity,
                    applied_field=request.material.applied_field,
                    background_intensity=request.material.background_intensity,
                    dx_normalized=dx_normalized,
                    initial_E=state,
                    options=tolerances.material_solver,
                    xp=xp,
                )
                material_converged = material_result.converged
                material_status = material_result.status
                material_state = material_result.E
            if not material_converged:
                termination_reason = f"material_{material_status}"
                if request.solver.record_iteration_history:
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index=k,
                            coupled_pass=coupled_pass,
                            residual_before_rms=residual_rms,
                            residual_before_max=residual_max,
                            residual_after_rms=residual_rms,
                            residual_after_max=residual_max,
                            delta_E_rms=0.0,
                            delta_E_max=0.0,
                            step_scale=0.0,
                            material_status=material_status,
                            accepted=False,
                        )
                    )
                break

            direction = material_state - state
            merit = 0.5 * residual_rms * residual_rms
            step_scale = 1.0
            accepted = False
            candidate = state
            candidate_A = A_trial
            candidate_intensity = intensity
            candidate_residual = residual
            candidate_rms = residual_rms
            candidate_max = residual_max
            for _ in range(int(request.solver.max_backtracks) + 1):
                trial_state = state + step_scale * direction
                trial_A, trial_intensity = advance_slice(
                    A_slice_in, trial_state, k
                )
                trial_residual = residual_at(trial_state, trial_intensity)
                trial_rms, trial_max = _residual_metrics(
                    trial_residual,
                    xp=xp,
                )
                trial_merit = 0.5 * trial_rms * trial_rms
                if (
                    not _array_has_true(~xp.isfinite(trial_state), xp=xp)
                    and not _array_has_true(~xp.isfinite(trial_A), xp=xp)
                    and math.isfinite(trial_merit)
                    and trial_merit
                    <= (1.0 - request.solver.armijo_fraction * step_scale)
                    * merit
                ):
                    accepted = True
                    candidate = trial_state
                    candidate_A = trial_A
                    candidate_intensity = trial_intensity
                    candidate_residual = trial_residual
                    candidate_rms = trial_rms
                    candidate_max = trial_max
                    break
                step_scale *= 0.5
                if step_scale < request.solver.minimum_step_scale:
                    break

            if not accepted:
                termination_reason = "coupled_line_search_failed"
                if request.solver.record_iteration_history:
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index=k,
                            coupled_pass=coupled_pass,
                            residual_before_rms=residual_rms,
                            residual_before_max=residual_max,
                            residual_after_rms=residual_rms,
                            residual_after_max=residual_max,
                            delta_E_rms=0.0,
                            delta_E_max=0.0,
                            step_scale=0.0,
                            material_status=material_status,
                            accepted=False,
                        )
                    )
                break

            final_delta_rms, final_delta_max = _update_metrics(
                candidate,
                state,
                xp=xp,
            )
            if request.solver.record_iteration_history:
                records.append(
                    PRCoupledStaticIterationRecord(
                        z_index=k,
                        coupled_pass=coupled_pass,
                        residual_before_rms=residual_rms,
                        residual_before_max=residual_max,
                        residual_after_rms=candidate_rms,
                        residual_after_max=candidate_max,
                        delta_E_rms=final_delta_rms,
                        delta_E_max=final_delta_max,
                        step_scale=step_scale,
                        material_status=material_status,
                        accepted=True,
                    )
                )
            state = candidate
            A_trial = candidate_A
            intensity = candidate_intensity
            residual = candidate_residual
            residual_rms = candidate_rms
            residual_max = candidate_max
            converged = _criteria_met(
                residual_rms, residual_max, tolerances
            )
            if converged:
                termination_reason = "residual_tolerance"
                break

        E_stack[k] = state
        source_stack[k] = intensity
        residual_stack[k] = residual
        A = A_trial
        summaries.append(
            PRCoupledStaticSliceSummary(
                z_index=k,
                z_um=float(k * grid.dz_um),
                coupled_passes=coupled_passes,
                final_residual_rms=residual_rms,
                final_residual_max=residual_max,
                final_delta_E_rms=final_delta_rms,
                final_delta_E_max=final_delta_max,
                converged=converged,
                termination_reason=termination_reason,
            )
        )
        completed_slices = k + 1
        if progress_callback is not None:
            progress_callback(
                RunProgress(
                    workflow=PR_STATIC_WORKFLOW,
                    status="running",
                    completed_units=completed_slices,
                    total_units=grid.Nz,
                    current_coordinate=float(completed_slices * grid.dz_um),
                    coordinate_name="z",
                    coordinate_unit="um",
                    elapsed_wall_time=perf_counter() - started_at,
                    latest_field_state={
                        "A_initial": np.asarray(asnumpy(A0)).copy(),
                        "A_current": np.asarray(asnumpy(A)).copy(),
                        "E_current": np.asarray(asnumpy(state)).copy(),
                        "source_intensity_current": np.asarray(
                            asnumpy(intensity)
                        ).copy(),
                        "residual_current": np.asarray(
                            asnumpy(residual)
                        ).copy(),
                        "completed_slices": completed_slices,
                        "grid_summary": grid.summary(),
                        "launch_summary": launch.summary(),
                    },
                    checkpoint_available=False,
                    message="PR static z slice completed",
                    diagnostics={
                        "phase": "solve",
                        "converged": converged,
                        "termination_reason": termination_reason,
                        "residual_rms": residual_rms,
                        "residual_max": residual_max,
                    },
                )
            )

    sequential_A = A.copy()
    replay_A = A0.copy()
    completed_E = E_stack[:completed_slices]
    completed_source = source_stack[:completed_slices]
    completed_residual = residual_stack[:completed_slices]
    replay_source = xp.empty_like(completed_source)
    replay_residual = xp.empty_like(completed_residual)
    report_completion_phase(
        "replay",
        completed_units=0,
        total_units=completed_slices,
        message=(
            "Validating static solution: independent replay "
            f"0/{completed_slices}"
        ),
    )
    replay_progress_cadence = max(1, math.ceil(completed_slices / 100))
    for k in range(completed_slices):
        replay_A, replay_source[k] = advance_slice(replay_A, E_stack[k], k)
        replay_residual[k] = residual_at(E_stack[k], replay_source[k])
        replay_completed = k + 1
        if (
            replay_completed % replay_progress_cadence == 0
            or replay_completed == completed_slices
        ):
            report_completion_phase(
                "replay",
                completed_units=replay_completed,
                total_units=completed_slices,
                message=(
                    "Validating static solution: independent replay "
                    f"{replay_completed}/{completed_slices}"
                ),
            )

    report_completion_phase(
        "replay_validation",
        completed_units=completed_slices,
        total_units=completed_slices,
        message="Validating replay consistency...",
    )
    if completed_slices:
        replay_rms, replay_max, replay_rms_converged = _completion_residual_metrics(
            replay_residual, xp=xp,
            rms_tolerance=tolerances.residual_rms_tolerance,
        )
    else:
        replay_rms, replay_max, replay_rms_converged = 0.0, 0.0, True
    comparison_options = dict(
        xp=xp, rtol=tolerances.replay_rtol, atol=tolerances.replay_atol,
    )
    field_max_abs, field_consistent = _completion_comparison(
        replay_A, sequential_A, **comparison_options,
    )
    if completed_slices:
        source_max_abs, source_consistent = _completion_comparison(
            replay_source, completed_source, **comparison_options,
        )
        residual_max_abs, residual_consistent = _completion_comparison(
            replay_residual, completed_residual, **comparison_options,
        )
    else:
        source_max_abs, residual_max_abs = 0.0, 0.0
        source_consistent, residual_consistent = True, True
    replay_converged = bool(
        completed_slices == grid.Nz
        and replay_rms_converged
        and replay_max <= tolerances.residual_max_tolerance
    )
    converged = bool(
        not cancelled
        and completed_slices == grid.Nz
        and all(summary.converged for summary in summaries)
        and replay_converged
        and field_consistent
        and source_consistent
        and residual_consistent
    )

    report_completion_phase(
        "scientific_result",
        completed_units=completed_slices,
        total_units=completed_slices,
        message="Preparing scientific results...",
    )

    return PRStaticRunResult(
        A_initial=_owned_host_result(A0),
        A_final=_owned_host_result(replay_A),
        E_initial=_owned_host_result(E_initial[:completed_slices]),
        E_final=_owned_host_result(completed_E),
        source_intensity_stack=_owned_host_result(replay_source),
        residual_stack=_owned_host_result(replay_residual),
        power_initial=normalized_power(A0, grid),
        power_final=normalized_power(replay_A, grid),
        converged=converged,
        completed_slices=completed_slices,
        iteration_records=tuple(records),
        slice_summaries=tuple(summaries),
        grid_summary=grid.summary(),
        launch_summary={
            **launch.summary(),
            "refractive_index": float(request.material.refractive_index),
            "carrier_channels": carrier_channels_from_beams(request.beams),
        },
        backend_summary=backend.summary(),
        tolerance_provenance=(
            {
                **tolerances.provenance(),
                "material_solver": {
                    "source": "not_applicable_direct_linearized_solve"
                },
            }
            if linearized
            else tolerances.provenance()
        ),
        replay_diagnostics={
            **({"canonical_scattering": canonical_scattering_provenance(
                request.scattering, z_length_um=request.grid.z_length_um,
                Nx=grid.Nx, Ny=grid.Ny,
                x_aperture_um=request.grid.x_aperture_um,
                y_aperture_um=request.grid.y_aperture_um,
                real_dtype=grid.real_dtype, xp=xp,
            )} if request.scattering is not None else {}),
            "performed_slices": completed_slices,
            "requested_slices": grid.Nz,
            "residual_rms": replay_rms,
            "residual_max": replay_max,
            "field_max_abs_difference": field_max_abs,
            "source_max_abs_difference": source_max_abs,
            "residual_max_abs_difference": residual_max_abs,
            "field_consistent": field_consistent,
            "source_consistent": source_consistent,
            "residual_consistent": residual_consistent,
            "residual_converged": replay_converged,
        },
        status=(
            "cancelled"
            if cancelled
            else ("converged" if converged else "not_converged")
        ),
        material_response_summary=(
            {
                "model": PR_MATERIAL_RESPONSE_LINEARIZED,
                "validation_status": "experimental",
                "operator": PR_REDUCED_LINEARIZED_RESPONSE_V1,
                "reference_intensity": float(
                    request.material_response.reference_intensity
                ),
                "background_intensity": float(
                    request.material.background_intensity
                ),
                "applied_field": float(request.material.applied_field),
                "equilibrium_field": float(linearized_spec.equilibrium_field),
                "material_response_calls": int(material_response_calls),
                "solver": "analytic_centered_difference_fourier",
            }
            if linearized
            else {
                "model": PR_MATERIAL_RESPONSE_NONLINEAR,
                "validation_status": "validated",
                "background_intensity": float(
                    request.material.background_intensity
                ),
            }
        ),
    )


__all__ = [
    "PRCoupledStaticIterationRecord",
    "PRCoupledStaticSliceSummary",
    "PRStaticRunRequest",
    "PRStaticRunResult",
    "PRStaticWorkflowOptions",
    "PR_STATIC_WORKFLOW",
    "run_pr_static",
]
