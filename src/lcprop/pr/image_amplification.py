"""PR-owned coherent image-amplification benchmarks and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np
from scipy.ndimage import zoom

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, channel_power_integrals
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.coupling import analytic_plane_wave_gain_length
from lcprop.pr.geometry import crossing_beam_channels
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PRStreamingStaticRequest,
    PRStreamingStaticResult,
    run_pr_static_streaming,
)


@dataclass(frozen=True)
class PRImageAmplificationSpec:
    """Finite-image two-beam benchmark parameters.

    The default is a scaled CPU case. :func:`paper_figure4_spec` returns the
    research-scale geometry reported in the source paper.
    """

    Nx: int = 64
    Ny: int = 32
    x_aperture_um: float = 64.0
    y_aperture_um: float = 32.0
    interaction_length_um: float = 100.0
    dz_um: float = 10.0
    wavelength_um: float = 0.633
    refractive_index: float = 2.4
    positive_mode_index: int = 2
    beam_waist_um: float = 12.0
    image_size_factor: float = 1.0
    input_peak_ratio: float = 1e-3
    saturated_small_signal_gain: float | None = 10.0
    signal_gain_sign: int = 1
    dark_intensity: float = 0.05
    applied_field: float = 0.0
    characteristic_wavenumber_per_um: float | None = 0.5
    relative_permittivity: float = 2500.0
    mobile_charge_density_m3: float = 6.4e22
    temperature_K: float = 293.0
    gain_length_product_override: float | None = None
    tukey_alpha: float = 0.0
    volume_noise_epsilon: float = 0.0
    volume_noise_correlation_um: float = 0.4
    volume_noise_seed: int | None = None
    volume_noise_seeds: tuple[int, ...] | None = None
    Nt: int = 500
    dt_normalized: float = 0.05
    invert_image: bool = True
    coherence_group: str = "pr-image-amplification"


@dataclass(frozen=True)
class PRImageAmplificationResult:
    """Optical, gain, and image-fidelity evidence from one benchmark."""

    request: PRRunRequest
    run_result: PRRunResult
    image_transmission: np.ndarray
    signal_carrier_mask: np.ndarray
    input_signal_field: np.ndarray
    output_signal_field: np.ndarray
    backpropagated_signal_field: np.ndarray
    zero_response_backpropagated_signal_field: np.ndarray
    transverse_phase_gradients_rad_per_um: tuple[float, float]
    normalized_grating_wavenumber: float
    gain_length_product: float
    analytic_gamma_p_L: float
    analytic_absolute_signal_gain: float
    measured_absolute_signal_gain: float
    image_intensity_correlation: float
    zero_response_image_intensity_correlation: float
    normalized_image_rmse: float
    normalized_power_relative_drift: float
    pr_workflow_runtime_s: float
    reconstruction_optical_runtime_s: float
    reconstruction_runtime_s: float
    runtime_s: float


@dataclass(frozen=True)
class PRStreamingImageAmplificationResult:
    """Figure-style observables from a bounded-memory static PR march."""

    request: PRStreamingStaticRequest
    run_result: PRStreamingStaticResult
    image_transmission: np.ndarray
    input_signal_field: np.ndarray
    output_signal_field: np.ndarray
    backpropagated_signal_field: np.ndarray
    output_total_intensity: np.ndarray
    normalized_grating_wavenumber: float
    gain_length_product: float
    analytic_absolute_signal_gain: float
    measured_absolute_signal_gain: float
    image_intensity_correlation: float
    normalized_image_rmse: float
    normalized_power_relative_drift: float


def paper_absolute_signal_gain(
    *,
    input_ratio: float,
    gamma_p_L: float,
) -> float:
    """Return the paper's finite-ratio absolute signal gain.

    This is Equation (11) in the published paper,
    ``G0=(1+r)*exp(2*gamma_p*L)/(1+r*exp(2*gamma_p*L))``.
    """

    ratio = float(input_ratio)
    coupling = float(gamma_p_L)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("input_ratio must be finite and nonnegative")
    if not math.isfinite(coupling):
        raise ValueError("gamma_p_L must be finite")
    exponential = math.exp(2.0 * coupling)
    return (1.0 + ratio) * exponential / (1.0 + ratio * exponential)


def paper_figure4_spec() -> PRImageAmplificationSpec:
    """Return the research-scale parameters reported for paper Figure 4.

    The external incidence angle is 7.56 degrees. On the 4 mm periodic
    aperture it lies essentially on Fourier mode 1024; selecting that exact
    integer mode preserves periodicity and gives eight samples per grating
    period. The source paper's Tukey apodization is intentionally not included
    because LCProp's current PR workflow has periodic transverse boundaries.
    ``Nt`` and ``dt_normalized`` are provisional LCProp material-time controls,
    not parameters reported for the paper's static Figure 4 calculation.
    """

    external_angle = math.radians(7.56)
    mode = round(4000.0 * math.sin(external_angle) / 0.514)
    return PRImageAmplificationSpec(
        Nx=16384,
        Ny=2048,
        x_aperture_um=4000.0,
        y_aperture_um=4000.0,
        interaction_length_um=4350.0,
        dz_um=2.0,
        wavelength_um=0.514,
        refractive_index=2.4,
        positive_mode_index=mode,
        beam_waist_um=3400.0,
        image_size_factor=1.0,
        input_peak_ratio=1e-5,
        saturated_small_signal_gain=4000.0,
        dark_intensity=0.01,
        characteristic_wavenumber_per_um=None,
        Nt=250,
        dt_normalized=0.01,
        invert_image=False,
    )


def paper_figure6_spec() -> PRImageAmplificationSpec:
    """Return the research-scale parameters reported for paper Figure 6.

    These are the saved trusted-implementation parameters supplied with the
    benchmark, with the quoted image selector intentionally replaced by the
    Air Force resolution chart at the caller boundary. The saved calculation
    generated and saved one random volume-noise seed per longitudinal slice.
    LCProp preserves that exact sequence for deterministic replay.
    """

    from lcprop.pr.figure6_noise_seeds import FIGURE6_VOLUME_NOISE_SEEDS

    external_angle = 0.08543723722873033
    mode = round(3000.0 * math.sin(external_angle) / 0.5)
    return PRImageAmplificationSpec(
        Nx=16384,
        Ny=1024,
        x_aperture_um=3000.0,
        y_aperture_um=1000.0,
        interaction_length_um=3940.0,
        dz_um=2.0,
        wavelength_um=0.5,
        refractive_index=2.4,
        positive_mode_index=mode,
        beam_waist_um=600.0,
        image_size_factor=1.0,
        input_peak_ratio=1.0,
        saturated_small_signal_gain=None,
        dark_intensity=0.01,
        applied_field=0.0,
        characteristic_wavenumber_per_um=None,
        relative_permittivity=2500.0,
        mobile_charge_density_m3=2e22,
        temperature_K=293.0,
        gain_length_product_override=10.0,
        tukey_alpha=0.2,
        volume_noise_epsilon=0.02,
        volume_noise_correlation_um=0.4,
        volume_noise_seed=None,
        volume_noise_seeds=FIGURE6_VOLUME_NOISE_SEEDS,
        invert_image=False,
    )


def _validate_image(image_intensity) -> np.ndarray:
    supplied = np.asarray(image_intensity)
    if supplied.ndim != 2:
        raise ValueError("image_intensity must be a two-dimensional array")
    if supplied.dtype.kind not in "fiu":
        raise TypeError("image_intensity must have a real numeric dtype")
    image = supplied.astype(np.float64, copy=True)
    if not np.all(np.isfinite(image)):
        raise ValueError("image_intensity must contain only finite values")
    if np.any(image < 0.0):
        raise ValueError("image_intensity must be nonnegative")
    maximum = float(np.max(image))
    if maximum <= 0.0:
        raise ValueError("image_intensity must contain a positive value")
    return image / maximum


def prepare_image_transmission(
    image_intensity,
    grid,
    *,
    center_x_um: float,
    center_y_um: float,
    physical_size_um: float,
    invert: bool = False,
) -> np.ndarray:
    """Place a legacy-compatible real intensity transparency on the grid.

    Input arrays use conventional image indexing ``(row_y, column_x)``. They
    are padded to a square with transparent pixels, rotated into LCProp's
    ``(x, y)`` array convention, nearest-neighbor resized, and centered on the
    selected beam. Outside the image footprint the transparency is one, as in
    PRProp3D. The launch-field amplitude multiplier is the square root of this
    returned intensity transmission.
    """

    image = _validate_image(image_intensity)
    if bool(invert):
        image = 1.0 - image
    size_um = float(physical_size_um)
    if not math.isfinite(size_um) or size_um <= 0.0:
        raise ValueError("physical_size_um must be finite and positive")

    side = max(image.shape)
    if side % 2:
        side += 1
    square = np.ones((side, side), dtype=np.float64)
    y_offset = (side - image.shape[0]) // 2
    x_offset = (side - image.shape[1]) // 2
    square[
        y_offset : y_offset + image.shape[0],
        x_offset : x_offset + image.shape[1],
    ] = image
    image_xy = np.rot90(square)

    target_nx = max(2, int(round(size_um / float(grid.dx_um))))
    target_ny = max(2, int(round(size_um / float(grid.dy_um))))
    resized = zoom(
        image_xy,
        (target_nx / image_xy.shape[0], target_ny / image_xy.shape[1]),
        order=0,
        mode="nearest",
        prefilter=False,
    )
    resized = resized[:target_nx, :target_ny]

    center_x = int(np.argmin(np.abs(np.asarray(grid.x_um) - center_x_um)))
    center_y = int(np.argmin(np.abs(np.asarray(grid.y_um) - center_y_um)))
    destination = np.ones((grid.Nx, grid.Ny), dtype=np.float64)
    x_start = center_x - resized.shape[0] // 2
    y_start = center_y - resized.shape[1] // 2
    x_stop = x_start + resized.shape[0]
    y_stop = y_start + resized.shape[1]
    destination_x0 = max(0, x_start)
    destination_y0 = max(0, y_start)
    destination_x1 = min(grid.Nx, x_stop)
    destination_y1 = min(grid.Ny, y_stop)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        raise ValueError("image transparency lies outside the transverse grid")
    source_x0 = destination_x0 - x_start
    source_y0 = destination_y0 - y_start
    source_x1 = source_x0 + destination_x1 - destination_x0
    source_y1 = source_y0 + destination_y1 - destination_y0
    destination[
        destination_x0:destination_x1,
        destination_y0:destination_y1,
    ] = resized[source_x0:source_x1, source_y0:source_y1]
    return destination


def _resolved_characteristic_wavenumber(spec: PRImageAmplificationSpec) -> float:
    override = spec.characteristic_wavenumber_per_um
    if override is not None:
        return float(override)
    return PRMaterialSpec().characteristic_wavenumber_per_um


def _gain_geometry(
    spec: PRImageAmplificationSpec,
) -> tuple[float, float, float, float]:
    mode = int(spec.positive_mode_index)
    if mode <= 0 or 4 * mode >= int(spec.Nx):
        raise ValueError(
            "positive_mode_index must place the two-beam grating below Nyquist"
        )
    kx = 2.0 * math.pi * mode / float(spec.x_aperture_um)
    k_medium = 2.0 * math.pi * float(spec.refractive_index) / float(spec.wavelength_um)
    if kx >= k_medium:
        raise ValueError("selected carrier is not a propagating mode")
    internal_angle = math.asin(kx / k_medium)
    k0 = _resolved_characteristic_wavenumber(spec)
    normalized_grating = -2.0 * kx / k0
    coupling_factor = (
        2.0
        * normalized_grating
        / (math.cos(internal_angle) * (1.0 + normalized_grating**2))
    )
    if int(spec.signal_gain_sign) not in (-1, 1):
        raise ValueError("signal_gain_sign must be +1 or -1")
    override = spec.gain_length_product_override
    if override is None:
        gamma_p_L = (
            int(spec.signal_gain_sign)
            * 0.5
            * math.log(float(spec.saturated_small_signal_gain))
        )
        gain_length_product = gamma_p_L / coupling_factor
    else:
        gain_length_product = float(override)
        if not math.isfinite(gain_length_product):
            raise ValueError("gain_length_product_override must be finite")
    return kx, internal_angle, normalized_grating, gain_length_product


def make_image_amplification_request(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
) -> tuple[PRRunRequest, np.ndarray, float, float]:
    """Create an image-bearing signal and crossing coherent pump request."""

    ratio = float(spec.input_peak_ratio)
    saturated_gain = spec.saturated_small_signal_gain
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("input_peak_ratio must be finite and positive")
    if spec.gain_length_product_override is None:
        if (
            saturated_gain is None
            or not math.isfinite(float(saturated_gain))
            or float(saturated_gain) <= 1.0
        ):
            raise ValueError(
                "saturated_small_signal_gain must be greater than one when "
                "gain_length_product_override is not supplied"
            )
    if not math.isfinite(float(spec.beam_waist_um)) or spec.beam_waist_um <= 0.0:
        raise ValueError("beam_waist_um must be finite and positive")

    kx, internal_angle, normalized_grating, gain_length = _gain_geometry(spec)
    grid_spec = GridSpec(
        Nx=int(spec.Nx),
        Ny=int(spec.Ny),
        x_aperture_um=float(spec.x_aperture_um),
        y_aperture_um=float(spec.y_aperture_um),
        z_length_um=float(spec.interaction_length_um),
        dz_um=float(spec.dz_um),
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    channels = crossing_beam_channels(
        wavelength_um=float(spec.wavelength_um),
        refractive_index=float(spec.refractive_index),
        interaction_length_um=float(spec.interaction_length_um),
        polar_angles_rad=(internal_angle, internal_angle),
        azimuths_rad=(0.0, math.pi),
        waist_x_um=float(spec.beam_waist_um),
        waist_y_um=float(spec.beam_waist_um),
        beam_ratio=ratio,
        coherence_group=spec.coherence_group,
        names=("pump", "image signal"),
    )
    preliminary_beams = BeamStack(channels=channels, coherence="coherent")
    launch = build_launch(
        preliminary_beams,
        grid,
        complex_dtype=np.complex128,
    )
    A0 = np.asarray(launch.A0).copy()
    transmission = prepare_image_transmission(
        image_intensity,
        grid,
        center_x_um=channels[1].x0_um,
        center_y_um=channels[1].y0_um,
        physical_size_um=(float(spec.image_size_factor) * float(spec.beam_waist_um)),
        invert=spec.invert_image,
    )
    A0[1] *= np.sqrt(transmission)

    pump_peak = float(np.max(np.abs(A0[0]) ** 2))
    signal_peak = float(np.max(np.abs(A0[1]) ** 2))
    if signal_peak <= 0.0:
        raise ValueError("image transparency blocks the complete signal beam")
    A0[1] *= math.sqrt(ratio * pump_peak / signal_peak)
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    A0 /= math.sqrt(float(np.sum(np.abs(A0) ** 2)) * dxdy)

    powers = channel_power_integrals(A0, grid)
    resolved_channels = (
        replace(channels[0], power_mW=float(powers[0])),
        replace(channels[1], power_mW=float(powers[1])),
    )
    beams = BeamStack(channels=resolved_channels, coherence="coherent")
    material = PRMaterialSpec(
        dark_intensity=float(spec.dark_intensity),
        applied_field=float(spec.applied_field),
        gain_length_product=gain_length,
        refractive_index=float(spec.refractive_index),
        relative_permittivity=float(spec.relative_permittivity),
        mobile_charge_density_m3=float(spec.mobile_charge_density_m3),
        temperature_K=float(spec.temperature_K),
        characteristic_wavenumber_per_um_override=(
            None
            if spec.characteristic_wavenumber_per_um is None
            else float(spec.characteristic_wavenumber_per_um)
        ),
    )
    request = PRRunRequest(
        grid=grid_spec,
        beams=beams,
        material=material,
        solver=PRSolverOptions(
            Nt=int(spec.Nt),
            dt_normalized=float(spec.dt_normalized),
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        ),
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
        initial_A=A0,
    )
    return request, transmission, normalized_grating, gain_length


def signal_carrier_mask(
    grid,
    *,
    pump_kx_rad_per_um: float,
    signal_kx_rad_per_um: float,
) -> np.ndarray:
    """Return the PRProp3D nearest-carrier Fourier partition for the signal."""

    kx = 2.0 * math.pi * np.fft.fftfreq(grid.Nx, d=float(grid.dx_um))[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(grid.Ny, d=float(grid.dy_um))[None, :]
    signal_distance = (kx - float(signal_kx_rad_per_um)) ** 2 + ky**2
    pump_distance = (kx - float(pump_kx_rad_per_um)) ** 2 + ky**2
    return signal_distance <= pump_distance


def isolate_signal_carrier(field, mask: np.ndarray) -> np.ndarray:
    """Isolate one angular carrier from a coherent two-beam field."""

    supplied = np.asarray(field)
    if supplied.ndim != 2 or supplied.shape != mask.shape:
        raise ValueError("field and mask must have the same two-dimensional shape")
    return np.fft.ifft2(np.fft.fft2(supplied) * mask)


def _linear_propagate(
    field: np.ndarray, grid, *, distance_um: float, request
) -> np.ndarray:
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=float(distance_um),
        wavelength=float(request.beams.channels[0].wavelength_um),
        n_ref=float(request.material.refractive_index),
        xp=np,
    )
    return np.fft.ifft2(np.fft.fft2(field) * kernel)


def _intensity_metrics(
    reference_field: np.ndarray,
    measured_field: np.ndarray,
    roi: np.ndarray,
) -> tuple[float, float]:
    reference = np.abs(reference_field[roi]) ** 2
    measured = np.abs(measured_field[roi]) ** 2
    reference_centered = reference - np.mean(reference)
    measured_centered = measured - np.mean(measured)
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(measured_centered)
    )
    if denominator == 0.0:
        raise ValueError("image-fidelity ROI has zero intensity variance")
    correlation = float(
        np.vdot(reference_centered, measured_centered).real / denominator
    )
    measured_sum = float(np.sum(measured))
    if measured_sum <= 0.0:
        raise ValueError("measured image has zero power in the fidelity ROI")
    scaled_measured = measured * (float(np.sum(reference)) / measured_sum)
    normalized_rmse = float(
        np.linalg.norm(scaled_measured - reference) / np.linalg.norm(reference)
    )
    return correlation, normalized_rmse


def run_image_amplification(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
    *,
    backend: BackendSpec | None = None,
) -> PRImageAmplificationResult:
    """Run and measure a finite coherent image-amplification benchmark.

    ``backend`` overrides the default NumPy execution backend without changing
    the image-amplification physics or reconstruction. Workflow results are
    returned on the host, so the existing NumPy reconstruction remains shared
    by CPU and GPU runs.
    """

    request, transmission, normalized_grating, gain_length = (
        make_image_amplification_request(image_intensity, spec)
    )
    if backend is not None:
        request = replace(request, backend=backend)
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = float(request.beams.channels[0].tilt_x_rad_per_um)
    signal_kx = float(request.beams.channels[1].tilt_x_rad_per_um)
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    coherent_input = np.sum(np.asarray(request.initial_A), axis=0)
    input_signal = isolate_signal_carrier(coherent_input, mask)

    benchmark_started_at = perf_counter()
    workflow_started_at = perf_counter()
    run_result = run_pr_timedependent(request)
    pr_workflow_runtime = perf_counter() - workflow_started_at
    reconstruction_started_at = perf_counter()
    coherent_output = np.sum(np.asarray(run_result.A_final), axis=0)
    output_signal = isolate_signal_carrier(coherent_output, mask)
    reconstruction_optical_runtime = 0.0
    optical_started_at = perf_counter()
    backpropagated = _linear_propagate(
        output_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at

    optical_started_at = perf_counter()
    zero_output = _linear_propagate(
        coherent_input,
        grid,
        distance_um=float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at
    zero_signal = isolate_signal_carrier(zero_output, mask)
    optical_started_at = perf_counter()
    zero_backpropagated = _linear_propagate(
        zero_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at
    signal_channel_intensity = np.abs(np.asarray(request.initial_A)[1]) ** 2
    roi = signal_channel_intensity > 1e-4 * float(np.max(signal_channel_intensity))
    correlation, normalized_rmse = _intensity_metrics(
        input_signal,
        backpropagated,
        roi,
    )
    zero_correlation, _ = _intensity_metrics(
        input_signal,
        zero_backpropagated,
        roi,
    )

    dxdy = float(grid.dx_um) * float(grid.dy_um)
    input_signal_power = float(np.sum(np.abs(input_signal) ** 2) * dxdy)
    output_signal_power = float(np.sum(np.abs(output_signal) ** 2) * dxdy)
    measured_gain = output_signal_power / input_signal_power
    internal_angle = math.asin(
        pump_kx
        * float(spec.wavelength_um)
        / (2.0 * math.pi * float(spec.refractive_index))
    )
    analytic_gamma = analytic_plane_wave_gain_length(
        gain_length_product=gain_length,
        signed_grating_k_normalized=normalized_grating,
        internal_half_angle_rad=internal_angle,
    )
    analytic_gain = paper_absolute_signal_gain(
        input_ratio=float(spec.input_peak_ratio),
        gamma_p_L=analytic_gamma,
    )
    power_drift = float(
        (run_result.power_final - run_result.power_initial) / run_result.power_initial
    )
    reconstruction_runtime = perf_counter() - reconstruction_started_at

    return PRImageAmplificationResult(
        request=request,
        run_result=run_result,
        image_transmission=transmission,
        signal_carrier_mask=mask,
        input_signal_field=input_signal,
        output_signal_field=output_signal,
        backpropagated_signal_field=backpropagated,
        zero_response_backpropagated_signal_field=zero_backpropagated,
        transverse_phase_gradients_rad_per_um=(pump_kx, signal_kx),
        normalized_grating_wavenumber=normalized_grating,
        gain_length_product=gain_length,
        analytic_gamma_p_L=analytic_gamma,
        analytic_absolute_signal_gain=analytic_gain,
        measured_absolute_signal_gain=measured_gain,
        image_intensity_correlation=correlation,
        zero_response_image_intensity_correlation=zero_correlation,
        normalized_image_rmse=normalized_rmse,
        normalized_power_relative_drift=power_drift,
        pr_workflow_runtime_s=pr_workflow_runtime,
        reconstruction_optical_runtime_s=reconstruction_optical_runtime,
        reconstruction_runtime_s=reconstruction_runtime,
        runtime_s=perf_counter() - benchmark_started_at,
    )


def run_streaming_image_amplification(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
    *,
    solver: PRStreamingStaticOptions | None = None,
    backend: BackendSpec | None = None,
) -> PRStreamingImageAmplificationResult:
    """Run the static streaming workflow through the established measurement chain.

    The raw coherent output intensity is retained for a Figure 6-style output
    panel.  Gain and fidelity use the same Fourier carrier partition and
    matched-field back-propagation as the transient image benchmark; they do
    not spatially divide coherently overlapping beams.
    """

    transient_request, transmission, normalized_grating, gain_length = (
        make_image_amplification_request(image_intensity, spec)
    )
    selected_backend = transient_request.backend if backend is None else backend
    if solver is None:
        solver = PRStreamingStaticOptions(
            tukey_alpha=float(spec.tukey_alpha),
            volume_noise_epsilon=float(spec.volume_noise_epsilon),
            volume_noise_correlation_um=float(spec.volume_noise_correlation_um),
            volume_noise_seed=spec.volume_noise_seed,
            volume_noise_seeds=spec.volume_noise_seeds,
        )
    request = PRStreamingStaticRequest(
        grid=transient_request.grid,
        beams=transient_request.beams,
        material=transient_request.material,
        solver=solver,
        backend=selected_backend,
        initial_A=transient_request.initial_A,
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = float(request.beams.channels[0].tilt_x_rad_per_um)
    signal_kx = float(request.beams.channels[1].tilt_x_rad_per_um)
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    coherent_input = np.sum(np.asarray(request.initial_A), axis=0)
    input_signal = isolate_signal_carrier(coherent_input, mask)
    run_result = run_pr_static_streaming(request)
    coherent_output = np.sum(np.asarray(run_result.A_final), axis=0)
    output_signal = isolate_signal_carrier(coherent_output, mask)
    backpropagated = _linear_propagate(
        output_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    signal_channel_intensity = np.abs(np.asarray(request.initial_A)[1]) ** 2
    roi = signal_channel_intensity > 1e-4 * float(np.max(signal_channel_intensity))
    correlation, normalized_rmse = _intensity_metrics(
        input_signal,
        backpropagated,
        roi,
    )
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    input_signal_power = float(np.sum(np.abs(input_signal) ** 2) * dxdy)
    output_signal_power = float(np.sum(np.abs(output_signal) ** 2) * dxdy)
    internal_angle = math.asin(
        pump_kx
        * float(spec.wavelength_um)
        / (2.0 * math.pi * float(spec.refractive_index))
    )
    analytic_gamma = analytic_plane_wave_gain_length(
        gain_length_product=gain_length,
        signed_grating_k_normalized=normalized_grating,
        internal_half_angle_rad=internal_angle,
    )
    analytic_gain = paper_absolute_signal_gain(
        input_ratio=float(spec.input_peak_ratio),
        gamma_p_L=analytic_gamma,
    )
    return PRStreamingImageAmplificationResult(
        request=request,
        run_result=run_result,
        image_transmission=transmission,
        input_signal_field=input_signal,
        output_signal_field=output_signal,
        backpropagated_signal_field=backpropagated,
        output_total_intensity=np.abs(coherent_output) ** 2,
        normalized_grating_wavenumber=normalized_grating,
        gain_length_product=gain_length,
        analytic_absolute_signal_gain=analytic_gain,
        measured_absolute_signal_gain=output_signal_power / input_signal_power,
        image_intensity_correlation=correlation,
        normalized_image_rmse=normalized_rmse,
        normalized_power_relative_drift=(
            run_result.power_final - run_result.power_initial
        )
        / run_result.power_initial,
    )


__all__ = [
    "PRImageAmplificationResult",
    "PRImageAmplificationSpec",
    "PRStreamingImageAmplificationResult",
    "isolate_signal_carrier",
    "make_image_amplification_request",
    "paper_absolute_signal_gain",
    "paper_figure4_spec",
    "paper_figure6_spec",
    "prepare_image_transmission",
    "run_image_amplification",
    "run_streaming_image_amplification",
    "signal_carrier_mask",
]
