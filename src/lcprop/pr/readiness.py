"""Bounded readiness checks for future PR image-amplification work."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.pr.coupling import PRPropagationTrace, trace_frozen_pr_state
from lcprop.pr.evolution import (
    conservative_timestep_limit,
    hopping_rhs,
    semi_implicit_conservative_timestep_limit,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.static import PRStaticResult, solve_pr_static_intensity
from lcprop.pr.workflow import run_pr_timedependent


@dataclass(frozen=True)
class PRImageAmplificationReadinessSpec:
    """Small coherent pump/signal case for the numerical acceptance gate."""

    Nx: int = 64
    Ny: int = 16
    x_aperture_um: float = 64.0
    y_aperture_um: float = 32.0
    interaction_length_um: float = 100.0
    dz_um: float = 10.0
    wavelength_um: float = 0.633
    refractive_index: float = 2.4
    positive_mode_index: int = 2
    pump_power_mW: float = 10.0
    signal_power_mW: float = 0.2
    pump_waist_um: float = 200.0
    signal_waist_um: float = 10.0
    dark_intensity: float = 0.05
    gain_length_product: float = 1.5
    characteristic_wavenumber_per_um: float = 0.5
    Nt: int = 250
    dt_normalized: float = 0.05
    coherence_group: str = "pr-readiness-laser"


@dataclass(frozen=True)
class PRImageAmplificationReadinessResult:
    """Numerical evidence from the bounded pump/signal readiness case."""

    request: PRRunRequest
    run_result: PRRunResult
    static_result: PRStaticResult
    zero_response_trace: PRPropagationTrace
    final_response_trace: PRPropagationTrace
    transverse_phase_gradients_rad_per_um: tuple[float, float]
    grating_samples_per_period: float
    explicit_euler_dt_limit: float
    semi_implicit_dt_limit: float
    final_residual_rms: float
    final_residual_max: float
    static_relative_l2_error: float
    static_max_error: float
    zero_response_output_matched_powers: tuple[float, float]
    final_response_output_matched_powers: tuple[float, float]
    signal_matched_power_relative_change: float
    normalized_power_relative_drift: float
    runtime_s: float


def make_image_amplification_readiness_request(
    spec: PRImageAmplificationReadinessSpec = PRImageAmplificationReadinessSpec(),
) -> tuple[PRRunRequest, float, float]:
    """Construct the bounded, grid-commensurate pump/signal request.

    The broad pump and weak localized signal use opposite integer Fourier
    tilts. Their coherent grating is therefore periodic on the FFT aperture.
    This is a numerical-readiness case, not an image-amplification benchmark.
    """

    mode = int(spec.positive_mode_index)
    if mode <= 0 or 4 * mode >= int(spec.Nx):
        raise ValueError(
            "positive_mode_index must place the two-beam grating below Nyquist"
        )
    for name in (
        "pump_power_mW",
        "signal_power_mW",
        "pump_waist_um",
        "signal_waist_um",
        "characteristic_wavenumber_per_um",
    ):
        value = float(getattr(spec, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    kx = 2.0 * math.pi * mode / float(spec.x_aperture_um)
    k_medium = 2.0 * math.pi * float(spec.refractive_index) / float(
        spec.wavelength_um
    )
    if kx >= k_medium:
        raise ValueError("selected pump/signal Fourier modes do not propagate")
    grating_samples = float(spec.Nx) / (2.0 * mode)
    if grating_samples < 8.0:
        raise ValueError(
            "pump/signal grating must have at least eight samples per period"
        )

    grid = GridSpec(
        Nx=int(spec.Nx),
        Ny=int(spec.Ny),
        x_aperture_um=float(spec.x_aperture_um),
        y_aperture_um=float(spec.y_aperture_um),
        z_length_um=float(spec.interaction_length_um),
        dz_um=float(spec.dz_um),
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="broad pump",
                wavelength_um=float(spec.wavelength_um),
                power_mW=float(spec.pump_power_mW),
                waist_x_um=float(spec.pump_waist_um),
                waist_y_um=float(spec.pump_waist_um),
                tilt_x_rad_per_um=kx,
                coherence_group=spec.coherence_group,
            ),
            BeamChannel(
                name="weak spatial signal",
                wavelength_um=float(spec.wavelength_um),
                power_mW=float(spec.signal_power_mW),
                waist_x_um=float(spec.signal_waist_um),
                waist_y_um=float(spec.signal_waist_um),
                tilt_x_rad_per_um=-kx,
                coherence_group=spec.coherence_group,
            ),
        ),
        coherence="coherent",
    )
    request = PRRunRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=float(spec.dark_intensity),
            applied_field=0.0,
            gain_length_product=float(spec.gain_length_product),
            refractive_index=float(spec.refractive_index),
            characteristic_wavenumber_per_um_override=float(
                spec.characteristic_wavenumber_per_um
            ),
        ),
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
    )
    return request, kx, grating_samples


def _residual_metrics(residual: np.ndarray) -> tuple[float, float]:
    values = np.asarray(residual, dtype=np.float64)
    return (
        float(np.sqrt(np.mean(values * values))),
        float(np.max(np.abs(values))),
    )


def run_image_amplification_readiness(
    spec: PRImageAmplificationReadinessSpec = PRImageAmplificationReadinessSpec(),
) -> PRImageAmplificationReadinessResult:
    """Run the bounded numerical gate preceding an image benchmark."""

    request, kx, grating_samples = make_image_amplification_readiness_request(
        spec
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    explicit_limit = conservative_timestep_limit(grid, request.material)
    semi_implicit_limit = semi_implicit_conservative_timestep_limit(
        grid,
        request.material,
    )

    started_at = perf_counter()
    run_result = run_pr_timedependent(request)
    dx_normalized = (
        request.material.characteristic_wavenumber_per_um * grid.dx_um
    )
    residual = hopping_rhs(
        run_result.E_final,
        run_result.source_intensity_stack,
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity,
        dx_normalized=dx_normalized,
        xp=np,
    )
    residual_rms, residual_max = _residual_metrics(residual)

    # Start independently from zero rather than using the transient as Newton's
    # initial guess. The optical source remains prescribed at its final value.
    static_result = solve_pr_static_intensity(
        run_result.source_intensity_stack,
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity,
        dx_normalized=dx_normalized,
    )
    difference = np.asarray(run_result.E_final) - static_result.E
    static_norm = float(np.linalg.norm(static_result.E))
    static_relative_l2_error = float(np.linalg.norm(difference) / static_norm)
    static_max_error = float(np.max(np.abs(difference)))

    zero_response_trace = trace_frozen_pr_state(
        request,
        np.zeros_like(run_result.E_final),
    )
    final_response_trace = trace_frozen_pr_state(request, run_result.E_final)
    zero_output = np.asarray(zero_response_trace.matched_mode_powers[-1])
    final_output = np.asarray(final_response_trace.matched_mode_powers[-1])
    signal_relative_change = float(
        (final_output[1] - zero_output[1]) / zero_output[1]
    )
    power_relative_drift = float(
        (run_result.power_final - run_result.power_initial)
        / run_result.power_initial
    )
    runtime = perf_counter() - started_at

    return PRImageAmplificationReadinessResult(
        request=request,
        run_result=run_result,
        static_result=static_result,
        zero_response_trace=zero_response_trace,
        final_response_trace=final_response_trace,
        transverse_phase_gradients_rad_per_um=(kx, -kx),
        grating_samples_per_period=grating_samples,
        explicit_euler_dt_limit=explicit_limit,
        semi_implicit_dt_limit=semi_implicit_limit,
        final_residual_rms=residual_rms,
        final_residual_max=residual_max,
        static_relative_l2_error=static_relative_l2_error,
        static_max_error=static_max_error,
        zero_response_output_matched_powers=(
            float(zero_output[0]),
            float(zero_output[1]),
        ),
        final_response_output_matched_powers=(
            float(final_output[0]),
            float(final_output[1]),
        ),
        signal_matched_power_relative_change=signal_relative_change,
        normalized_power_relative_drift=power_relative_drift,
        runtime_s=runtime,
    )


__all__ = [
    "PRImageAmplificationReadinessResult",
    "PRImageAmplificationReadinessSpec",
    "make_image_amplification_readiness_request",
    "run_image_amplification_readiness",
]
