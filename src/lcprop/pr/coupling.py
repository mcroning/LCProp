"""PR-specific two-beam coupling benchmarks and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.optics.splitstep import (
    advance_prepared_response,
    hop_linear_inplace,
    linear_kernel,
)
from lcprop.pr.geometry import (
    PRApertureReport,
    analyze_crossing_aperture,
    crossing_beam_channels,
)
from lcprop.pr.optical_response import half_step_response_from_E
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
)
from lcprop.pr.workflow import run_pr_timedependent


@dataclass(frozen=True)
class PlaneWaveCouplingSpec:
    """Small periodic Section 3.1 two-plane-wave benchmark."""

    Nx: int = 128
    Ny: int = 4
    x_aperture_um: float = 64.0
    y_aperture_um: float = 4.0
    interaction_length_um: float = 100.0
    dz_um: float = 5.0
    positive_mode_index: int = 4
    wavelength_um: float = 0.633
    refractive_index: float = 2.4
    beam_ratio: float = 0.01
    gain_length_product: float = 0.05
    dark_intensity: float = 0.01
    characteristic_wavenumber_per_um: float = 1.0
    Nt: int = 600
    dt_normalized: float = 0.01
    optical_substeps: int = 1
    coherence_group: str = "pr-plane-wave"


@dataclass(frozen=True)
class PlaneWaveCouplingResult:
    """Modal and continuum-theory outputs for a periodic plane-wave run."""

    request: PRRunRequest
    run_result: PRRunResult
    kx_rad_per_um: tuple[float, float]
    polar_angles_rad: tuple[float, float]
    grating_k_normalized: float
    grating_samples_per_period: float
    input_modal_powers: tuple[float, float]
    output_modal_powers: tuple[float, float]
    input_total_power: float
    output_total_power: float
    input_ratio: float
    output_ratio: float
    gamma_p_L_measured: float
    gamma_p_L_expected: float
    runtime_s: float


def analytic_plane_wave_gain_length(
    *,
    gain_length_product: float,
    signed_grating_k_normalized: float,
    internal_half_angle_rad: float,
) -> float:
    """Return paper Equation (7), ``gamma_p*L`` from the input ``gamma*L``."""

    kg = float(signed_grating_k_normalized)
    cosine = math.cos(float(internal_half_angle_rad))
    if cosine <= 0.0:
        raise ValueError("internal_half_angle_rad must have positive cosine")
    return float(gain_length_product) * 2.0 * kg / (
        cosine * (1.0 + kg * kg)
    )


def _coherent_field(A: np.ndarray) -> np.ndarray:
    if A.ndim == 2:
        return A
    if A.ndim == 3:
        return np.sum(A, axis=0)
    raise ValueError("field must have shape (Nx, Ny) or (Nch, Nx, Ny)")


def project_fourier_modes(
    field,
    grid,
    wavevectors_rad_per_um: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Project a coherent complex field onto orthogonal periodic plane waves."""

    coherent = np.asarray(_coherent_field(np.asarray(field)))
    X = np.asarray(grid.x_um)[:, None]
    Y = np.asarray(grid.y_um)[None, :]
    amplitudes = []
    for kx, ky in wavevectors_rad_per_um:
        basis_conjugate = np.exp(-1j * (float(kx) * X + float(ky) * Y))
        amplitudes.append(np.mean(coherent * basis_conjugate))
    amplitude_array = np.asarray(amplitudes)
    area = float(grid.spec.x_aperture_um) * float(grid.spec.y_aperture_um)
    return amplitude_array, np.abs(amplitude_array) ** 2 * area


def make_plane_wave_coupling_request(
    spec: PlaneWaveCouplingSpec,
) -> tuple[PRRunRequest, tuple[float, float], tuple[float, float]]:
    """Create a grid-commensurate coherent plane-wave PR request."""

    if int(spec.positive_mode_index) <= 0 or 2 * int(spec.positive_mode_index) >= int(
        spec.Nx
    ):
        raise ValueError("positive_mode_index must lie below the x Nyquist mode")
    if not math.isfinite(float(spec.beam_ratio)) or spec.beam_ratio <= 0.0:
        raise ValueError("beam_ratio must be finite and positive")
    grid_spec = GridSpec(
        Nx=int(spec.Nx),
        Ny=int(spec.Ny),
        x_aperture_um=float(spec.x_aperture_um),
        y_aperture_um=float(spec.y_aperture_um),
        z_length_um=float(spec.interaction_length_um),
        dz_um=float(spec.dz_um),
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    mode = int(spec.positive_mode_index)
    kx = 2.0 * math.pi * mode / float(spec.x_aperture_um)
    k_medium = 2.0 * math.pi * float(spec.refractive_index) / float(
        spec.wavelength_um
    )
    if abs(kx) >= k_medium:
        raise ValueError("selected Fourier mode is not a propagating input angle")
    angle = math.asin(kx / k_medium)
    ratio = float(spec.beam_ratio)
    amplitudes = (math.sqrt(1.0 / (1.0 + ratio)), math.sqrt(ratio / (1.0 + ratio)))
    X = np.asarray(grid.x_um)[:, None]
    uniform_y = np.ones((1, grid.Ny))
    initial_A = np.stack(
        (
            amplitudes[0] * np.exp(1j * kx * X) * uniform_y,
            amplitudes[1] * np.exp(-1j * kx * X) * uniform_y,
        )
    ).astype(np.complex128)
    channels = (
        BeamChannel(
            name="pump",
            wavelength_um=spec.wavelength_um,
            power_mW=1.0,
            waist_x_um=spec.x_aperture_um,
            waist_y_um=spec.y_aperture_um,
            tilt_x_rad_per_um=kx,
            coherence_group=spec.coherence_group,
        ),
        BeamChannel(
            name="signal",
            wavelength_um=spec.wavelength_um,
            power_mW=ratio,
            waist_x_um=spec.x_aperture_um,
            waist_y_um=spec.y_aperture_um,
            tilt_x_rad_per_um=-kx,
            coherence_group=spec.coherence_group,
        ),
    )
    grating_samples = math.pi / (kx * float(grid.dx_um))
    if grating_samples < 6.0:
        raise ValueError("periodic two-beam grating has fewer than six samples per period")
    request = PRRunRequest(
        grid=grid_spec,
        beams=BeamStack(channels=channels, coherence="coherent"),
        material=PRMaterialSpec(
            dark_intensity=spec.dark_intensity,
            applied_field=0.0,
            gain_length_product=spec.gain_length_product,
            refractive_index=spec.refractive_index,
            characteristic_wavenumber_per_um_override=(
                spec.characteristic_wavenumber_per_um
            ),
        ),
        solver=PRSolverOptions(
            Nt=spec.Nt,
            dt_normalized=spec.dt_normalized,
            optical_substeps=spec.optical_substeps,
        ),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
        initial_A=initial_A,
    )
    return request, (kx, -kx), (angle, angle)


def run_plane_wave_coupling(
    spec: PlaneWaveCouplingSpec = PlaneWaveCouplingSpec(),
) -> PlaneWaveCouplingResult:
    """Run and measure the periodic Section 3.1 benchmark."""

    request, kx_values, angles = make_plane_wave_coupling_request(spec)
    grid = make_grid(request.grid, real_dtype=np.float64)
    start = time.perf_counter()
    result = run_pr_timedependent(request)
    runtime = time.perf_counter() - start
    wavevectors = ((kx_values[0], 0.0), (kx_values[1], 0.0))
    _, input_powers = project_fourier_modes(result.A_initial, grid, wavevectors)
    _, output_powers = project_fourier_modes(result.A_final, grid, wavevectors)
    input_ratio = float(input_powers[1] / input_powers[0])
    output_ratio = float(output_powers[1] / output_powers[0])
    grating_k = (kx_values[1] - kx_values[0]) / (
        request.material.characteristic_wavenumber_per_um
    )
    expected = analytic_plane_wave_gain_length(
        gain_length_product=spec.gain_length_product,
        signed_grating_k_normalized=grating_k,
        internal_half_angle_rad=angles[0],
    )
    coherent_initial = _coherent_field(result.A_initial)
    coherent_output = _coherent_field(result.A_final)
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    return PlaneWaveCouplingResult(
        request=request,
        run_result=result,
        kx_rad_per_um=kx_values,
        polar_angles_rad=angles,
        grating_k_normalized=grating_k,
        grating_samples_per_period=(
            math.pi / (abs(kx_values[0]) * float(grid.dx_um))
        ),
        input_modal_powers=(float(input_powers[0]), float(input_powers[1])),
        output_modal_powers=(float(output_powers[0]), float(output_powers[1])),
        input_total_power=float(np.sum(np.abs(coherent_initial) ** 2) * dxdy),
        output_total_power=float(np.sum(np.abs(coherent_output) ** 2) * dxdy),
        input_ratio=input_ratio,
        output_ratio=output_ratio,
        gamma_p_L_measured=0.5 * math.log(output_ratio / input_ratio),
        gamma_p_L_expected=expected,
        runtime_s=runtime,
    )


@dataclass(frozen=True)
class PRPropagationTrace:
    """Frozen-state z trace using matched-field modal measurements."""

    z_um: np.ndarray
    matched_mode_powers: np.ndarray
    coherent_total_power: np.ndarray
    channel_centroid_x_um: np.ndarray
    channel_centroid_y_um: np.ndarray


def _centroid(
    intensity: np.ndarray,
    coordinate: np.ndarray,
    *,
    coordinate_axis: int,
) -> float:
    marginal = np.sum(intensity, axis=1 - int(coordinate_axis))
    return float(np.sum(marginal * coordinate) / np.sum(marginal))


def trace_frozen_pr_state(request: PRRunRequest, E) -> PRPropagationTrace:
    """Replay a final PR state and record matched modes, power, and centroids."""

    if request.backend.backend != "numpy":
        raise ValueError("diagnostic trace currently requires the NumPy backend")
    request.grid.validate()
    grid = make_grid(request.grid, real_dtype=np.float64)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128)
    A = (
        launch.A0.copy()
        if request.initial_A is None
        else np.asarray(request.initial_A, dtype=np.complex128).copy()
    )
    references = A.copy()
    state = np.asarray(E, dtype=float)
    if state.shape != (grid.Nz, grid.Nx, grid.Ny):
        raise ValueError("E must have shape (Nz, Nx, Ny)")
    wavelength = float(request.beams.channels[0].wavelength_um)
    Nsub = int(request.solver.optical_substeps)
    dz_substep = float(grid.dz_um) / Nsub
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=dz_substep,
        wavelength=wavelength,
        n_ref=request.material.refractive_index,
        xp=np,
    )
    z_values = np.arange(grid.Nz + 1, dtype=float) * float(grid.dz_um)
    matched = np.empty((grid.Nz + 1, A.shape[0]), dtype=float)
    total_power = np.empty(grid.Nz + 1, dtype=float)
    centroid_x = np.empty((grid.Nz + 1, A.shape[0]), dtype=float)
    centroid_y = np.empty((grid.Nz + 1, A.shape[0]), dtype=float)
    dxdy = float(grid.dx_um) * float(grid.dy_um)

    def record(index: int) -> None:
        total = np.sum(A, axis=0)
        gram = np.empty((A.shape[0], A.shape[0]), dtype=np.complex128)
        overlaps = np.empty(A.shape[0], dtype=np.complex128)
        for row in range(A.shape[0]):
            overlaps[row] = np.vdot(references[row], total) * dxdy
            for column in range(A.shape[0]):
                gram[row, column] = (
                    np.vdot(references[row], references[column]) * dxdy
                )
        coefficients = np.linalg.solve(gram, overlaps)
        matched[index] = np.abs(coefficients) ** 2 * np.real(np.diag(gram))
        total_power[index] = float(np.sum(np.abs(total) ** 2) * dxdy)
        for channel in range(A.shape[0]):
            intensity = np.abs(A[channel]) ** 2
            centroid_x[index, channel] = _centroid(
                intensity,
                np.asarray(grid.x_um),
                coordinate_axis=0,
            )
            centroid_y[index, channel] = _centroid(
                intensity,
                np.asarray(grid.y_um),
                coordinate_axis=1,
            )

    record(0)
    for slice_index in range(grid.Nz):
        response = half_step_response_from_E(
            state[slice_index],
            dz_substep_um=dz_substep,
            wavelength_um=wavelength,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            xp=np,
        )
        advance_prepared_response(
            A,
            kernel=kernel,
            half_step_response=response,
            Nsub=Nsub,
            xp=np,
        )
        for _ in range(Nsub):
            hop_linear_inplace(references, kernel, xp=np)
        record(slice_index + 1)
    return PRPropagationTrace(
        z_um=z_values,
        matched_mode_powers=matched,
        coherent_total_power=total_power,
        channel_centroid_x_um=centroid_x,
        channel_centroid_y_um=centroid_y,
    )


@dataclass(frozen=True)
class FiniteGaussianCouplingSpec:
    """Modest CPU finite-Gaussian crossing benchmark."""

    Nx: int = 96
    Ny: int = 32
    x_aperture_um: float = 60.0
    y_aperture_um: float = 40.0
    interaction_length_um: float = 200.0
    dz_um: float = 5.0
    wavelength_um: float = 0.633
    refractive_index: float = 1.6
    polar_angle_rad: float = 0.03
    waist_x_um: float = 8.0
    waist_y_um: float = 8.0
    beam_ratio: float = 0.25
    gain_length_product: float = 0.1
    dark_intensity: float = 0.01
    characteristic_wavenumber_per_um: float = 1.0
    Nt: int = 100
    dt_normalized: float = 0.01
    coherence_group: str = "pr-finite-two-beam"


@dataclass(frozen=True)
class FiniteGaussianCouplingResult:
    """Finite crossing run, aperture report, and z diagnostics."""

    request: PRRunRequest
    run_result: PRRunResult
    trace: PRPropagationTrace
    aperture: PRApertureReport
    predicted_crossing_z_um: float
    measured_crossing_z_um: float
    crossing_error_um: float
    grating_amplitude_by_z: np.ndarray
    output_beam_ratio: float
    runtime_s: float


def _crossing_from_centroids(trace: PRPropagationTrace) -> float:
    separation = trace.channel_centroid_x_um[:, 0] - trace.channel_centroid_x_um[:, 1]
    exact = np.flatnonzero(separation == 0.0)
    if exact.size:
        return float(trace.z_um[exact[0]])
    changes = np.flatnonzero(separation[:-1] * separation[1:] < 0.0)
    if not changes.size:
        return float(trace.z_um[np.argmin(np.abs(separation))])
    index = int(changes[0])
    fraction = -separation[index] / (separation[index + 1] - separation[index])
    return float(trace.z_um[index] + fraction * (trace.z_um[index + 1] - trace.z_um[index]))


def run_finite_gaussian_coupling(
    spec: FiniteGaussianCouplingSpec = FiniteGaussianCouplingSpec(),
) -> FiniteGaussianCouplingResult:
    """Run the finite coherent Gaussian crossing smoke benchmark."""

    crossing_z = float(spec.interaction_length_um) / 2.0
    channels = crossing_beam_channels(
        wavelength_um=spec.wavelength_um,
        refractive_index=spec.refractive_index,
        interaction_length_um=spec.interaction_length_um,
        polar_angles_rad=(spec.polar_angle_rad, spec.polar_angle_rad),
        azimuths_rad=(0.0, math.pi),
        waist_x_um=spec.waist_x_um,
        waist_y_um=spec.waist_y_um,
        crossing_z_um=crossing_z,
        beam_ratio=spec.beam_ratio,
        coherence_group=spec.coherence_group,
    )
    grid_spec = GridSpec(
        Nx=spec.Nx,
        Ny=spec.Ny,
        x_aperture_um=spec.x_aperture_um,
        y_aperture_um=spec.y_aperture_um,
        z_length_um=spec.interaction_length_um,
        dz_um=spec.dz_um,
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    aperture = analyze_crossing_aperture(
        grid,
        channels,
        refractive_index=spec.refractive_index,
        crossing_z_um=crossing_z,
        strict=True,
    )
    request = PRRunRequest(
        grid=grid_spec,
        beams=BeamStack(channels=channels, coherence="coherent"),
        material=PRMaterialSpec(
            dark_intensity=spec.dark_intensity,
            applied_field=0.0,
            gain_length_product=spec.gain_length_product,
            refractive_index=spec.refractive_index,
            characteristic_wavenumber_per_um_override=(
                spec.characteristic_wavenumber_per_um
            ),
        ),
        solver=PRSolverOptions(Nt=spec.Nt, dt_normalized=spec.dt_normalized),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )
    start = time.perf_counter()
    result = run_pr_timedependent(request)
    trace = trace_frozen_pr_state(request, result.E_final)
    runtime = time.perf_counter() - start
    measured_crossing = _crossing_from_centroids(trace)
    grating_amplitude = 0.5 * (
        np.max(result.E_final, axis=(1, 2)) - np.min(result.E_final, axis=(1, 2))
    )
    return FiniteGaussianCouplingResult(
        request=request,
        run_result=result,
        trace=trace,
        aperture=aperture,
        predicted_crossing_z_um=crossing_z,
        measured_crossing_z_um=measured_crossing,
        crossing_error_um=measured_crossing - crossing_z,
        grating_amplitude_by_z=grating_amplitude,
        output_beam_ratio=float(
            trace.matched_mode_powers[-1, 1] / trace.matched_mode_powers[-1, 0]
        ),
        runtime_s=runtime,
    )


def with_plane_wave_resolution(
    spec: PlaneWaveCouplingSpec,
    *,
    dz_um: float,
) -> PlaneWaveCouplingSpec:
    """Small convenience for explicit dz convergence studies."""

    return replace(spec, dz_um=float(dz_um))


__all__ = [
    "PlaneWaveCouplingSpec",
    "PlaneWaveCouplingResult",
    "analytic_plane_wave_gain_length",
    "project_fourier_modes",
    "make_plane_wave_coupling_request",
    "run_plane_wave_coupling",
    "PRPropagationTrace",
    "trace_frozen_pr_state",
    "FiniteGaussianCouplingSpec",
    "FiniteGaussianCouplingResult",
    "run_finite_gaussian_coupling",
    "with_plane_wave_resolution",
]
