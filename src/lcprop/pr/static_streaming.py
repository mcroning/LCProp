"""PR-owned streaming static workflows and paper-code reference paths.

The production streaming path solves the same complete nonlinear residual and
uses the same midpoint-source Strang optical coupling as ``run_pr_static``.
It retains no ``(Nz, Nx, Ny)`` material volume.  Deterministic replay means a
second, independent streaming solve from the original optical launch and zero
material seed; no material state from the first pass is reused.

Two deliberately separate reference paths preserve the numerical choices in
the trusted PRProp3D static loop: spectral material derivatives, a full linear
hop followed by a full material screen (Lie ordering), and optional Tukey
apodization.  One uses the paper's linearized steady equation and the other
uses LCProp's complete nonlinear steady residual.  Neither is the production
solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy, get_backend
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid, round_nz
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.optical_response import half_step_response_from_E
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_phase_increment,
    canonical_scattering_provenance,
    canonical_slab_range,
)
from lcprop.pr.source import (
    channel_peak_intensity_reference,
    pr_driving_intensity,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import solve_pr_static_intensity_batched
from lcprop.pr.static_workflow import (
    PRCoupledStaticIterationRecord,
    PRCoupledStaticSliceSummary,
    PRStaticWorkflowOptions,
    _criteria_met,
    _resolve_static_tolerances,
)
from lcprop.pr.workflow import advance_pr_slice_with_midpoint_source


PR_STREAMING_PRODUCTION = "production_nonlinear_strang"
PR_STREAMING_FULL_NONLINEAR_REFERENCE = "full_nonlinear_lie_reference"
PR_STREAMING_LEGACY_LINEARIZED_REFERENCE = "legacy_linearized_spectral_lie"
PR_STREAMING_MODES = (
    PR_STREAMING_PRODUCTION,
    PR_STREAMING_FULL_NONLINEAR_REFERENCE,
    PR_STREAMING_LEGACY_LINEARIZED_REFERENCE,
)


@dataclass(frozen=True)
class PRStreamingStaticOptions:
    """Controls for a bounded-memory static z march.

    ``tukey_alpha=0`` disables apodization.  A positive value applies the
    trusted implementation's square-root two-dimensional Tukey window.  The
    legacy reference modes additionally apply its FFT-shifted window during
    the linear hop.  Production applies only the real-space window after the
    complete Strang slice so its response ordering remains symmetric.
    """

    coupled: PRStaticWorkflowOptions = field(default_factory=PRStaticWorkflowOptions)
    mode: str = PR_STREAMING_PRODUCTION
    tukey_alpha: float = 0.0
    capture_y_indices: tuple[int, ...] = ()
    capture_x_indices: tuple[int, ...] = ()
    deterministic_replay: bool = True
    volume_noise_epsilon: float = 0.0
    volume_noise_correlation_um: float = 0.4
    volume_noise_seed: int | None = None
    volume_noise_seeds: tuple[int, ...] | None = None
    partition_independent_scattering: PRCanonicalScatteringSpec | None = None

    def validate(self) -> None:
        self.coupled.validate()
        if self.mode not in PR_STREAMING_MODES:
            raise ValueError("mode must be one of " + ", ".join(PR_STREAMING_MODES))
        alpha = float(self.tukey_alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("tukey_alpha must be between zero and one")
        epsilon = float(self.volume_noise_epsilon)
        correlation = float(self.volume_noise_correlation_um)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("volume_noise_epsilon must be finite and nonnegative")
        if not math.isfinite(correlation) or correlation <= 0.0:
            raise ValueError("volume_noise_correlation_um must be finite and positive")
        if self.volume_noise_seed is not None and self.volume_noise_seeds is not None:
            raise ValueError(
                "volume_noise_seed and volume_noise_seeds are mutually exclusive"
            )
        if (
            epsilon > 0.0
            and self.volume_noise_seed is None
            and self.volume_noise_seeds is None
        ):
            raise ValueError(
                "volume_noise_seed or volume_noise_seeds is required for "
                "deterministic volume noise"
            )
        if self.volume_noise_seed is not None and int(self.volume_noise_seed) < 0:
            raise ValueError("volume_noise_seed must be nonnegative")
        if self.volume_noise_seeds is not None:
            for seed in self.volume_noise_seeds:
                if int(seed) != seed or not 0 <= int(seed) <= 2**32 - 1:
                    raise ValueError(
                        "volume_noise_seeds must contain unsigned 32-bit integers"
                    )
        canonical = self.partition_independent_scattering
        if canonical is not None:
            canonical.validate()
            if (
                epsilon != 0.0
                or self.volume_noise_seed is not None
                or self.volume_noise_seeds is not None
            ):
                raise ValueError(
                    "partition_independent_scattering and legacy volume-noise "
                    "controls are mutually exclusive"
                )
        for name in ("capture_y_indices", "capture_x_indices"):
            if any(
                int(index) != index or int(index) < 0 for index in getattr(self, name)
            ):
                raise ValueError(f"{name} must contain nonnegative integers")


@dataclass(frozen=True)
class PRStreamingStaticRequest:
    """Request for a PR static calculation without a retained E volume."""

    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    solver: PRStreamingStaticOptions = field(default_factory=PRStreamingStaticOptions)
    backend: BackendSpec = BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    initial_A: Any | None = None


@dataclass(frozen=True)
class PRStreamingPassResult:
    """Bounded-memory evidence from one independent streaming pass."""

    slice_summaries: tuple[PRCoupledStaticSliceSummary, ...]
    iteration_records: tuple[PRCoupledStaticIterationRecord, ...]
    state_moments: np.ndarray
    source_moments: np.ndarray
    residual_moments: np.ndarray
    full_nonlinear_residual_moments: np.ndarray
    E_xz_at_y: dict[int, np.ndarray]
    E_yz_at_x: dict[int, np.ndarray]
    source_xz_at_y: dict[int, np.ndarray]
    source_yz_at_x: dict[int, np.ndarray]
    converged: bool


@dataclass(frozen=True)
class PRStreamingStaticResult:
    """Host-boundary result and deterministic recomputation diagnostics."""

    A_initial: np.ndarray
    A_final: np.ndarray
    first_pass: PRStreamingPassResult
    replay_pass: PRStreamingPassResult | None
    power_initial: float
    power_final: float
    completed_slices: int
    backend_summary: dict[str, Any]
    tolerance_provenance: dict[str, Any]
    memory_policy: dict[str, Any]
    replay_diagnostics: dict[str, Any]
    status: str


def _backend_scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _residual_metrics(residual, *, xp: Any) -> tuple[float, float]:
    return (
        _backend_scalar(xp.sqrt(xp.mean(residual * residual))),
        _backend_scalar(xp.max(xp.abs(residual))),
    )


def _update_metrics(new_state, old_state, *, xp: Any) -> tuple[float, float]:
    difference = new_state - old_state
    return (
        _backend_scalar(xp.sqrt(xp.mean(difference * difference))),
        _backend_scalar(xp.max(xp.abs(difference))),
    )


def _moments(value, *, xp: Any) -> tuple[float, float, float, float]:
    return (
        _backend_scalar(xp.sum(value)),
        _backend_scalar(xp.sum(value * value)),
        _backend_scalar(xp.min(value)),
        _backend_scalar(xp.max(value)),
    )


def _tukey(length: int, alpha: float, *, xp: Any, dtype: Any):
    if alpha <= 0.0:
        return xp.ones(length, dtype=dtype)
    if alpha >= 1.0:
        values = np.hanning(length + 1)[:-1]
    else:
        index = np.arange(length, dtype=np.float64)
        fraction = index / float(length)
        values = np.ones(length, dtype=np.float64)
        leading = fraction < alpha / 2.0
        trailing = fraction >= 1.0 - alpha / 2.0
        values[leading] = 0.5 * (
            1.0 + np.cos(2.0 * np.pi / alpha * (fraction[leading] - alpha / 2.0))
        )
        values[trailing] = 0.5 * (
            1.0 + np.cos(2.0 * np.pi / alpha * (fraction[trailing] - 1.0 + alpha / 2.0))
        )
    return xp.asarray(values, dtype=dtype)


def trusted_tukey_window(
    Nx: int,
    Ny: int,
    *,
    alpha: float,
    xp: Any,
    real_dtype: Any,
):
    """Return PRProp3D's ``sqrt(outer(tukey_x, tukey_y))`` window."""

    wx = _tukey(int(Nx), float(alpha), xp=xp, dtype=real_dtype)
    wy = _tukey(int(Ny), float(alpha), xp=xp, dtype=real_dtype)
    return xp.sqrt(wx[:, None] * wy[None, :])


def solve_legacy_linearized_static_intensity_spectral(
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
):
    """Solve paper Equation (5) with trusted PRProp3D spectral conventions.

    The supplied intensity already includes dark and optional uniform
    background.  With normalized Fourier wavenumber ``k``, this evaluates

    ``FFT(E) = FFT((E_app*I_b + I_x)/I)/(1 + i*E_app*k + k**2)``.
    """

    if intensity.ndim != 2:
        raise ValueError("intensity must have shape (Nx, Ny)")
    dx = float(dx_normalized)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    if _backend_scalar(xp.min(intensity)) <= 0.0:
        raise ValueError("intensity including dark/background must be positive")
    k = 2.0 * math.pi * xp.fft.fftfreq(intensity.shape[0], d=dx)[:, None]
    intensity_ft = xp.fft.fft(intensity, axis=-2)
    intensity_x = xp.fft.ifft(1j * k * intensity_ft, axis=-2).real
    rhs = (float(applied_field) * float(background_intensity) + intensity_x) / intensity
    denominator = 1.0 + 1j * float(applied_field) * k + k * k
    return xp.fft.ifft(xp.fft.fft(rhs, axis=-2) / denominator, axis=-2).real


def legacy_linearized_static_residual_spectral(
    E,
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
):
    """Return the discrete spectral residual of paper Equation (5)."""

    if E.shape != intensity.shape or E.ndim != 2:
        raise ValueError("E and intensity must have identical shape (Nx, Ny)")
    k = 2.0 * math.pi * xp.fft.fftfreq(E.shape[0], d=float(dx_normalized))[:, None]
    E_ft = xp.fft.fft(E, axis=-2)
    intensity_ft = xp.fft.fft(intensity, axis=-2)
    E_x = xp.fft.ifft(1j * k * E_ft, axis=-2).real
    E_xx = xp.fft.ifft(-(k * k) * E_ft, axis=-2).real
    intensity_x = xp.fft.ifft(1j * k * intensity_ft, axis=-2).real
    rhs = (float(applied_field) * float(background_intensity) + intensity_x) / intensity
    return E + float(applied_field) * E_x - E_xx - rhs


def _legacy_angular_spectrum_kernel(
    grid, *, wavelength_um: float, n_ref: float, xp: Any
):
    argument = 1.0 - (float(wavelength_um) / float(n_ref)) ** 2 * grid.fxy2_um
    return xp.where(
        argument >= 0.0,
        xp.exp(
            2.0j
            * math.pi
            * float(n_ref)
            * float(grid.dz_um)
            / float(wavelength_um)
            * xp.sqrt(xp.maximum(argument, 0.0))
        ),
        0.0,
    )


def _volume_noise_phase(
    z_index: int,
    *,
    request: PRStreamingStaticRequest,
    grid,
    xp: Any,
):
    canonical = request.solver.partition_independent_scattering
    if canonical is not None:
        return canonical_scattering_phase_increment(
            canonical,
            z_start_um=float(z_index) * float(grid.dz_um),
            dz_um=float(grid.dz_um),
            z_length_um=float(request.grid.z_length_um),
            Nx=grid.Nx,
            Ny=grid.Ny,
            x_aperture_um=float(request.grid.x_aperture_um),
            y_aperture_um=float(request.grid.y_aperture_um),
            real_dtype=grid.real_dtype,
            xp=xp,
        )
    epsilon = float(request.solver.volume_noise_epsilon)
    if epsilon == 0.0:
        return None
    seed = volume_noise_seed_for_slice(
        request.solver,
        z_index=z_index,
        longitudinal_slices=grid.Nz,
    )
    scale = math.sqrt(epsilon / grid.Nz * 4.0 * math.pi)
    if getattr(xp, "__name__", "") == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter  # type: ignore

        raw = (
            xp.random.RandomState(seed)
            .normal(0.0, scale, size=(grid.Nx, grid.Ny))
            .astype(grid.real_dtype)
        )
    else:
        from scipy.ndimage import gaussian_filter

        raw = (
            np.random.RandomState(seed)
            .normal(0.0, scale, size=(grid.Nx, grid.Ny))
            .astype(grid.real_dtype)
        )
    sigma_x = (
        request.solver.volume_noise_correlation_um
        * grid.Nx
        / request.grid.x_aperture_um
    )
    sigma_y = (
        request.solver.volume_noise_correlation_um
        * grid.Ny
        / request.grid.y_aperture_um
    )
    return gaussian_filter(raw, sigma=(sigma_x, sigma_y)) * math.sqrt(sigma_x * sigma_y)


def volume_noise_seed_for_slice(
    options: PRStreamingStaticOptions,
    *,
    z_index: int,
    longitudinal_slices: int,
) -> int:
    """Resolve the deterministic noise seed for one longitudinal slice."""

    index = int(z_index)
    count = int(longitudinal_slices)
    if count < 1 or not 0 <= index < count:
        raise ValueError("z_index must select one longitudinal slice")
    explicit = options.volume_noise_seeds
    if explicit is not None:
        if len(explicit) != count:
            raise ValueError(
                "volume_noise_seeds length must equal the number of z slices"
            )
        return int(explicit[index])
    if options.volume_noise_seed is None:
        raise ValueError("volume-noise seed policy is unresolved")
    return int(
        np.random.SeedSequence([int(options.volume_noise_seed), index]).generate_state(
            1
        )[0]
    )


def _volume_noise_provenance(
    options: PRStreamingStaticOptions,
    *,
    request: PRStreamingStaticRequest,
    grid,
    xp: Any,
) -> dict[str, Any]:
    canonical = options.partition_independent_scattering
    if canonical is not None:
        return canonical_scattering_provenance(
            canonical,
            z_length_um=float(request.grid.z_length_um),
            Nx=grid.Nx,
            Ny=grid.Ny,
            x_aperture_um=float(request.grid.x_aperture_um),
            y_aperture_um=float(request.grid.y_aperture_um),
            real_dtype=grid.real_dtype,
            xp=xp,
        )
    explicit = options.volume_noise_seeds
    if explicit is not None:
        values = np.asarray(explicit, dtype="<u4")
        return {
            "mode": "legacy_per_material_slice",
            "epsilon": float(options.volume_noise_epsilon),
            "correlation_um": float(options.volume_noise_correlation_um),
            "seed_policy": "explicit_per_slice_sequence",
            "seed_count": int(values.size),
            "seed_sha256_le_u32": hashlib.sha256(values.tobytes()).hexdigest(),
            "first_seed": int(values[0]),
            "last_seed": int(values[-1]),
        }
    return {
        "mode": "legacy_per_material_slice",
        "epsilon": float(options.volume_noise_epsilon),
        "correlation_um": float(options.volume_noise_correlation_um),
        "seed_policy": "SeedSequence(base_seed, z_index)",
        "base_seed": options.volume_noise_seed,
    }


def _validate_request(request: PRStreamingStaticRequest):
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    wavelengths = tuple(float(c.wavelength_um) for c in request.beams.channels)
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("minimal PR workflow requires one shared wavelength")
    for index in request.solver.capture_y_indices:
        if index >= request.grid.Ny:
            raise ValueError("capture_y_indices contains an out-of-range index")
    for index in request.solver.capture_x_indices:
        if index >= request.grid.Nx:
            raise ValueError("capture_x_indices contains an out-of-range index")
    if request.solver.volume_noise_seeds is not None and len(
        request.solver.volume_noise_seeds
    ) != round_nz(request.grid.z_length_um, request.grid.dz_um):
        raise ValueError("volume_noise_seeds length must equal the number of z slices")
    canonical = request.solver.partition_independent_scattering
    if canonical is not None:
        canonical_slab_range(
            canonical,
            z_start_um=0.0,
            dz_um=float(request.grid.dz_um),
            z_length_um=float(request.grid.z_length_um),
        )
    return wavelengths[0]


def _capture(target: dict[int, list[np.ndarray]], indices, state, *, axis: str) -> None:
    for index in indices:
        value = state[:, index] if axis == "y" else state[index, :]
        target[int(index)].append(np.asarray(asnumpy(value)).copy())


def _finalize_captures(values: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    return {
        index: np.stack(rows, axis=0) if rows else np.empty((0,))
        for index, rows in values.items()
    }


def _run_streaming_pass(
    *,
    request: PRStreamingStaticRequest,
    A0,
    grid,
    kernel,
    legacy_kernel,
    peak_reference: float,
    wavelength_um: float,
    tolerances,
    window,
    xp: Any,
) -> tuple[PRStreamingPassResult, Any]:
    mode = request.solver.mode
    coupled = request.solver.coupled
    groups = request.beams.coherence_groups
    dx_normalized = request.material.characteristic_wavenumber_per_um * grid.dx_um
    A = A0.copy()
    previous_state = xp.zeros((grid.Nx, grid.Ny), dtype=grid.real_dtype)
    records: list[PRCoupledStaticIterationRecord] = []
    summaries: list[PRCoupledStaticSliceSummary] = []
    state_moments: list[tuple[float, float, float, float]] = []
    source_moments: list[tuple[float, float, float, float]] = []
    residual_moments: list[tuple[float, float, float, float]] = []
    full_residual_moments: list[tuple[float, float, float, float]] = []
    E_xz = {int(i): [] for i in request.solver.capture_y_indices}
    E_yz = {int(i): [] for i in request.solver.capture_x_indices}
    source_xz = {int(i): [] for i in request.solver.capture_y_indices}
    source_yz = {int(i): [] for i in request.solver.capture_x_indices}
    fft_window = xp.fft.fftshift(window)

    def residual_at(state, intensity):
        return hopping_rhs(
            state,
            intensity,
            applied_field=request.material.applied_field,
            background_intensity=request.material.background_intensity,
            dx_normalized=dx_normalized,
            xp=xp,
        )

    def production_advance(A_in, state):
        source_before = None
        if request.solver.tukey_alpha > 0.0:
            source_before = pr_driving_intensity(
                A_in,
                peak_intensity_reference=peak_reference,
                background_intensity=request.material.background_intensity,
                coherence_groups=groups,
                xp=xp,
            )
        A_out, source = advance_pr_slice_with_midpoint_source(
            A_in,
            state,
            kernel=kernel,
            optical_substeps=coupled.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=groups,
            xp=xp,
        )
        if request.solver.tukey_alpha > 0.0:
            A_out *= window[None, :, :]
            source_after = pr_driving_intensity(
                A_out,
                peak_intensity_reference=peak_reference,
                background_intensity=request.material.background_intensity,
                coherence_groups=groups,
                xp=xp,
            )
            source = 0.5 * (source_before + source_after)
        return A_out, source

    for z_index in range(grid.Nz):
        A_slice_in = A.copy()
        state = previous_state.copy()

        if mode == PR_STREAMING_PRODUCTION:
            A_trial, intensity = production_advance(A_slice_in, state)
            residual = residual_at(state, intensity)
            residual_rms, residual_max = _residual_metrics(residual, xp=xp)
            converged = _criteria_met(residual_rms, residual_max, tolerances)
            termination = (
                "residual_tolerance" if converged else "maximum_coupled_passes"
            )
            passes = 0
            delta_rms = delta_max = 0.0
            for coupled_pass in range(1, int(coupled.max_coupled_passes) + 1):
                if converged:
                    break
                passes = coupled_pass
                material_result = solve_pr_static_intensity_batched(
                    intensity,
                    applied_field=request.material.applied_field,
                    background_intensity=request.material.background_intensity,
                    dx_normalized=dx_normalized,
                    initial_E=state,
                    options=tolerances.material_solver,
                    xp=xp,
                )
                if not material_result.converged:
                    termination = f"material_{material_result.status}"
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index,
                            coupled_pass,
                            residual_rms,
                            residual_max,
                            residual_rms,
                            residual_max,
                            0.0,
                            0.0,
                            0.0,
                            material_result.status,
                            False,
                        )
                    )
                    break
                direction = material_result.E - state
                merit = 0.5 * residual_rms * residual_rms
                step_scale = 1.0
                accepted = False
                for _ in range(int(coupled.max_backtracks) + 1):
                    candidate = state + step_scale * direction
                    candidate_A, candidate_intensity = production_advance(
                        A_slice_in, candidate
                    )
                    candidate_residual = residual_at(candidate, candidate_intensity)
                    candidate_rms, candidate_max = _residual_metrics(
                        candidate_residual, xp=xp
                    )
                    candidate_merit = 0.5 * candidate_rms * candidate_rms
                    finite = bool(xp.all(xp.isfinite(candidate)).item()) and bool(
                        xp.all(xp.isfinite(candidate_A)).item()
                    )
                    if (
                        finite
                        and math.isfinite(candidate_merit)
                        and candidate_merit
                        <= (1.0 - coupled.armijo_fraction * step_scale) * merit
                    ):
                        accepted = True
                        break
                    step_scale *= 0.5
                    if step_scale < coupled.minimum_step_scale:
                        break
                if not accepted:
                    termination = "coupled_line_search_failed"
                    records.append(
                        PRCoupledStaticIterationRecord(
                            z_index,
                            coupled_pass,
                            residual_rms,
                            residual_max,
                            residual_rms,
                            residual_max,
                            0.0,
                            0.0,
                            0.0,
                            material_result.status,
                            False,
                        )
                    )
                    break
                delta_rms, delta_max = _update_metrics(candidate, state, xp=xp)
                records.append(
                    PRCoupledStaticIterationRecord(
                        z_index,
                        coupled_pass,
                        residual_rms,
                        residual_max,
                        candidate_rms,
                        candidate_max,
                        delta_rms,
                        delta_max,
                        step_scale,
                        material_result.status,
                        True,
                    )
                )
                state, A_trial, intensity, residual = (
                    candidate,
                    candidate_A,
                    candidate_intensity,
                    candidate_residual,
                )
                residual_rms, residual_max = candidate_rms, candidate_max
                converged = _criteria_met(residual_rms, residual_max, tolerances)
                if converged:
                    termination = "residual_tolerance"
                    break
            A = A_trial
        else:
            A = xp.fft.ifft2(
                xp.fft.fft2(A_slice_in, axes=(-2, -1))
                * legacy_kernel[None, :, :]
                * fft_window[None, :, :],
                axes=(-2, -1),
            )
            intensity = pr_driving_intensity(
                A,
                peak_intensity_reference=peak_reference,
                background_intensity=request.material.background_intensity,
                coherence_groups=groups,
                xp=xp,
            )
            if mode == PR_STREAMING_LEGACY_LINEARIZED_REFERENCE:
                state = solve_legacy_linearized_static_intensity_spectral(
                    intensity,
                    applied_field=request.material.applied_field,
                    background_intensity=request.material.background_intensity,
                    dx_normalized=dx_normalized,
                    xp=xp,
                )
                material_status = "linearized_spectral"
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
                state = material_result.E
                material_status = material_result.status
            full_residual = residual_at(state, intensity)
            if mode == PR_STREAMING_LEGACY_LINEARIZED_REFERENCE:
                residual = legacy_linearized_static_residual_spectral(
                    state,
                    intensity,
                    applied_field=request.material.applied_field,
                    background_intensity=request.material.background_intensity,
                    dx_normalized=dx_normalized,
                    xp=xp,
                )
            else:
                residual = full_residual
            residual_rms, residual_max = _residual_metrics(residual, xp=xp)
            converged = _criteria_met(residual_rms, residual_max, tolerances)
            termination = material_status
            passes = 1
            delta_rms, delta_max = _update_metrics(state, previous_state, xp=xp)
            full_response = half_step_response_from_E(
                state,
                dz_substep_um=2.0 * grid.dz_um,
                wavelength_um=wavelength_um,
                interaction_length_um=request.grid.z_length_um,
                gain_length_product=request.material.gain_length_product,
                xp=xp,
            )
            A *= full_response[None, :, :] * window[None, :, :]

        previous_state = state.copy()
        noise_phase = _volume_noise_phase(
            z_index,
            request=request,
            grid=grid,
            xp=xp,
        )
        if noise_phase is not None:
            A *= xp.exp(1j * noise_phase)[None, :, :]
        state_moments.append(_moments(state, xp=xp))
        source_moments.append(_moments(intensity, xp=xp))
        residual_moments.append(_moments(residual, xp=xp))
        full_residual = residual_at(state, intensity)
        full_residual_moments.append(_moments(full_residual, xp=xp))
        _capture(E_xz, request.solver.capture_y_indices, state, axis="y")
        _capture(E_yz, request.solver.capture_x_indices, state, axis="x")
        _capture(source_xz, request.solver.capture_y_indices, intensity, axis="y")
        _capture(source_yz, request.solver.capture_x_indices, intensity, axis="x")
        summaries.append(
            PRCoupledStaticSliceSummary(
                z_index=z_index,
                z_um=float(z_index * grid.dz_um),
                coupled_passes=passes,
                final_residual_rms=residual_rms,
                final_residual_max=residual_max,
                final_delta_E_rms=delta_rms,
                final_delta_E_max=delta_max,
                converged=converged,
                termination_reason=termination,
            )
        )

    return (
        PRStreamingPassResult(
            slice_summaries=tuple(summaries),
            iteration_records=tuple(records),
            state_moments=np.asarray(state_moments),
            source_moments=np.asarray(source_moments),
            residual_moments=np.asarray(residual_moments),
            full_nonlinear_residual_moments=np.asarray(full_residual_moments),
            E_xz_at_y=_finalize_captures(E_xz),
            E_yz_at_x=_finalize_captures(E_yz),
            source_xz_at_y=_finalize_captures(source_xz),
            source_yz_at_x=_finalize_captures(source_yz),
            converged=all(summary.converged for summary in summaries),
        ),
        A,
    )


def run_pr_static_streaming(
    request: PRStreamingStaticRequest,
) -> PRStreamingStaticResult:
    """Run a bounded-memory production or explicitly named reference march."""

    wavelength_um = _validate_request(request)
    backend = get_backend(request.backend)
    xp = backend.xp
    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=backend.complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(
                f"initial_A shape {A0.shape} does not match {launch.A0.shape}"
            )
    tolerances = _resolve_static_tolerances(
        request.solver.coupled, real_dtype=backend.real_dtype
    )
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.coupled.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    )
    legacy_kernel = _legacy_angular_spectrum_kernel(
        grid,
        wavelength_um=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    )
    window = trusted_tukey_window(
        grid.Nx,
        grid.Ny,
        alpha=request.solver.tukey_alpha,
        xp=xp,
        real_dtype=backend.real_dtype,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    arguments = dict(
        request=request,
        A0=A0,
        grid=grid,
        kernel=kernel,
        legacy_kernel=legacy_kernel,
        peak_reference=peak_reference,
        wavelength_um=wavelength_um,
        tolerances=tolerances,
        window=window,
        xp=xp,
    )
    first, first_A = _run_streaming_pass(**arguments)
    replay_execution = (
        _run_streaming_pass(**arguments)
        if request.solver.deterministic_replay
        else None
    )
    replay, replay_A = (None, None) if replay_execution is None else replay_execution
    rtol = tolerances.replay_rtol
    atol = tolerances.replay_atol
    if replay is None:
        replay_diagnostics = {
            "performed": False,
            "specification": "disabled explicitly",
            "consistent": False,
        }
        final_A = first_A
    else:
        field_difference = xp.abs(replay_A - first_A)
        field_max = _backend_scalar(xp.max(field_difference))
        field_consistent = bool(
            xp.all(xp.isclose(replay_A, first_A, rtol=rtol, atol=atol)).item()
        )
        moment_differences = {
            "state": float(np.max(np.abs(replay.state_moments - first.state_moments))),
            "source": float(
                np.max(np.abs(replay.source_moments - first.source_moments))
            ),
            "residual": float(
                np.max(np.abs(replay.residual_moments - first.residual_moments))
            ),
            "full_nonlinear_residual": float(
                np.max(
                    np.abs(
                        replay.full_nonlinear_residual_moments
                        - first.full_nonlinear_residual_moments
                    )
                )
            ),
        }
        moments_consistent = all(value == 0.0 for value in moment_differences.values())
        replay_diagnostics = {
            "performed": True,
            "specification": (
                "independent complete recomputation from original A0 and zero E seed; "
                "no first-pass material state is reused"
            ),
            "field_max_abs_difference": field_max,
            "field_consistent": field_consistent,
            "moment_max_abs_differences": moment_differences,
            "moments_bitwise_consistent": moments_consistent,
            "consistent": field_consistent and moments_consistent,
        }
        final_A = replay_A
    completed = grid.Nz
    converged = first.converged and (replay is None or replay.converged)
    replay_ok = replay is None or replay_diagnostics["consistent"]
    status = "converged" if converged and replay_ok else "not_converged"
    return PRStreamingStaticResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(final_A)).copy(),
        first_pass=first,
        replay_pass=replay,
        power_initial=normalized_power(A0, grid),
        power_final=normalized_power(final_A, grid),
        completed_slices=completed,
        backend_summary=backend.summary(),
        tolerance_provenance=tolerances.provenance(),
        memory_policy={
            "retained_material_volume": False,
            "retained_material_slices_across_z": 1,
            "solver_scratch": "O(Nx*Ny) backend arrays; no Nz scaling",
            "deterministic_replay_recomputes_material": bool(
                request.solver.deterministic_replay
            ),
            "volume_noise": {
                **_volume_noise_provenance(
                    request.solver,
                    request=request,
                    grid=grid,
                    xp=xp,
                ),
            },
            "captured_y_indices": tuple(request.solver.capture_y_indices),
            "captured_x_indices": tuple(request.solver.capture_x_indices),
            "host_boundary": "final fields, summaries, moments, and requested cross-sections only",
        },
        replay_diagnostics=replay_diagnostics,
        status=status,
    )


__all__ = [
    "PRStreamingPassResult",
    "PRStreamingStaticOptions",
    "PRStreamingStaticRequest",
    "PRStreamingStaticResult",
    "PR_STREAMING_FULL_NONLINEAR_REFERENCE",
    "PR_STREAMING_LEGACY_LINEARIZED_REFERENCE",
    "PR_STREAMING_MODES",
    "PR_STREAMING_PRODUCTION",
    "run_pr_static_streaming",
    "legacy_linearized_static_residual_spectral",
    "solve_legacy_linearized_static_intensity_spectral",
    "trusted_tukey_window",
    "volume_noise_seed_for_slice",
]
