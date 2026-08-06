from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.image_amplification import (
    PRImageAmplificationSpec,
    make_image_amplification_request,
    paper_figure6_spec,
    run_streaming_image_amplification,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PRStreamingStaticRequest,
    PR_STREAMING_FULL_NONLINEAR_REFERENCE,
    PR_STREAMING_LEGACY_LINEARIZED_REFERENCE,
    PR_STREAMING_PRODUCTION,
    legacy_linearized_static_residual_spectral,
    run_pr_static_streaming,
    solve_legacy_linearized_static_intensity_spectral,
    trusted_tukey_window,
    volume_noise_seed_for_slice,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)


def _request(*, mode=PR_STREAMING_PRODUCTION, backend="numpy", precision="float64"):
    grid = GridSpec(
        Nx=12,
        Ny=6,
        x_aperture_um=24.0,
        y_aperture_um=18.0,
        z_length_um=15.0,
        dz_um=5.0,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=7.0,
                waist_y_um=6.0,
                tilt_x_rad_per_um=2.0 * math.pi / 24.0,
                coherence_group="laser",
            ),
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=7.0,
                waist_y_um=6.0,
                tilt_x_rad_per_um=-2.0 * math.pi / 24.0,
                power_mW=0.4,
                coherence_group="laser",
            ),
        ),
        coherence="coherent",
    )
    coupled = PRStaticWorkflowOptions(
        material_solver=PRStaticSolverOptions(
            max_iterations=30,
            residual_rms_tolerance=1e-10 if precision == "float64" else 2e-6,
            residual_max_tolerance=1e-9 if precision == "float64" else 1e-5,
        ),
        max_coupled_passes=20,
        residual_rms_tolerance=1e-8 if precision == "float64" else 2e-6,
        residual_max_tolerance=1e-7 if precision == "float64" else 1e-5,
    )
    return PRStreamingStaticRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=0.1,
            gain_length_product=0.2,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRStreamingStaticOptions(
            coupled=coupled,
            mode=mode,
            capture_y_indices=(grid.Ny // 2,),
            capture_x_indices=(grid.Nx // 2,),
        ),
        backend=BackendSpec(backend=backend, precision=precision, verbose=False),
    )


def test_paper_figure6_spec_preserves_equal_ratio_published_contract():
    spec = paper_figure6_spec()
    angle = math.asin(
        spec.positive_mode_index * spec.wavelength_um / spec.x_aperture_um
    )

    assert (spec.Nx, spec.Ny) == (16384, 1024)
    assert (spec.x_aperture_um, spec.y_aperture_um) == (3000.0, 1000.0)
    assert spec.interaction_length_um == 3940.0
    assert spec.dz_um == 2.0
    assert spec.wavelength_um == 0.5
    assert spec.beam_waist_um == 600.0
    assert spec.input_peak_ratio == 1.0
    assert spec.saturated_small_signal_gain is None
    assert spec.gain_length_product_override == 10.0
    assert spec.dark_intensity == 0.01
    assert spec.mobile_charge_density_m3 == 2e22
    assert spec.tukey_alpha == 0.2
    assert spec.volume_noise_epsilon == 0.02
    assert spec.volume_noise_correlation_um == 0.4
    assert spec.volume_noise_seed is None
    assert len(spec.volume_noise_seeds) == 1970
    assert spec.volume_noise_seeds[:3] == (
        1304151306,
        2998548564,
        21195955,
    )
    assert spec.volume_noise_seeds[-1] == 2453989226
    seed_digest = hashlib.sha256(
        np.asarray(spec.volume_noise_seeds, dtype="<u4").tobytes()
    ).hexdigest()
    assert seed_digest == (
        "ade77c0e678bf3c2836131c4771e9df22774eba3cc3eb30e17150107adf2f32f"
    )
    assert spec.positive_mode_index == 512
    assert spec.Nx / (2 * spec.positive_mode_index) == pytest.approx(16.0)
    assert angle == pytest.approx(0.08543723722873033, abs=2e-6)


def test_figure6_direct_gain_and_material_parameters_reach_request():
    spec = replace(
        paper_figure6_spec(),
        Nx=64,
        Ny=32,
        x_aperture_um=96.0,
        y_aperture_um=32.0,
        interaction_length_um=40.0,
        dz_um=10.0,
        positive_mode_index=2,
        beam_waist_um=19.2,
    )
    request, _transmission, _grating, gain_length = make_image_amplification_request(
        np.ones((8, 8)), spec
    )

    assert gain_length == 10.0
    assert request.material.gain_length_product == 10.0
    assert request.material.applied_field == 0.0
    assert request.material.dark_intensity == 0.01
    assert request.material.relative_permittivity == 2500.0
    assert request.material.mobile_charge_density_m3 == 2e22
    assert request.material.temperature_K == 293.0


def test_legacy_spectral_solver_matches_direct_trusted_formula():
    Nx, Ny = 32, 3
    dx = 0.25
    x = np.arange(Nx) * dx
    intensity = 1.2 + 0.15 * np.cos(2.0 * math.pi * 3 * x / (Nx * dx))[:, None]
    intensity = np.broadcast_to(intensity, (Nx, Ny)).copy()
    applied = 0.4
    background = 0.2

    actual = solve_legacy_linearized_static_intensity_spectral(
        intensity,
        applied_field=applied,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    k = 2.0 * math.pi * np.fft.fftfreq(Nx, d=dx)[:, None]
    I_x = np.fft.ifft(1j * k * np.fft.fft(intensity, axis=0), axis=0).real
    expected = np.fft.ifft(
        np.fft.fft((applied * background + I_x) / intensity, axis=0)
        / (1.0 + 1j * applied * k + k**2),
        axis=0,
    ).real

    assert np.allclose(actual, expected, rtol=0.0, atol=2e-15)
    residual = legacy_linearized_static_residual_spectral(
        actual,
        intensity,
        applied_field=applied,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    assert np.max(np.abs(residual)) < 2e-14


def test_production_streaming_matches_retained_volume_and_replays_deterministically():
    request = _request()
    streaming = run_pr_static_streaming(request)
    volume = run_pr_static(
        PRStaticRunRequest(
            grid=request.grid,
            beams=request.beams,
            material=request.material,
            solver=request.solver.coupled,
            backend=request.backend,
        )
    )

    assert streaming.status == "converged"
    assert streaming.first_pass.converged
    assert streaming.replay_pass is not None
    assert streaming.replay_pass.converged
    assert streaming.replay_diagnostics["consistent"]
    assert streaming.replay_diagnostics["field_max_abs_difference"] == 0.0
    assert streaming.replay_diagnostics["moments_bitwise_consistent"]
    assert np.allclose(streaming.A_final, volume.A_final, rtol=0.0, atol=2e-14)
    assert streaming.memory_policy["retained_material_volume"] is False
    assert streaming.memory_policy["retained_material_slices_across_z"] == 1
    assert streaming.first_pass.E_xz_at_y[3].shape == (3, 12)
    assert streaming.first_pass.E_yz_at_x[6].shape == (3, 6)
    assert streaming.power_final == pytest.approx(streaming.power_initial, rel=2e-14)


def test_reference_modes_are_explicit_and_not_aliased_to_production():
    legacy = run_pr_static_streaming(
        _request(mode=PR_STREAMING_LEGACY_LINEARIZED_REFERENCE)
    )
    nonlinear_lie = run_pr_static_streaming(
        _request(mode=PR_STREAMING_FULL_NONLINEAR_REFERENCE)
    )
    production = run_pr_static_streaming(_request(mode=PR_STREAMING_PRODUCTION))

    assert legacy.status == "converged"
    assert nonlinear_lie.status == "converged"
    assert production.status == "converged"
    assert legacy.first_pass.slice_summaries[0].termination_reason == (
        "linearized_spectral"
    )
    assert nonlinear_lie.first_pass.slice_summaries[0].termination_reason == (
        "converged"
    )
    assert production.first_pass.slice_summaries[0].termination_reason == (
        "residual_tolerance"
    )
    assert not np.array_equal(legacy.A_final, nonlinear_lie.A_final)
    assert not np.array_equal(nonlinear_lie.A_final, production.A_final)
    assert np.max(np.abs(legacy.first_pass.residual_moments[:, :2])) < 2e-12
    assert (
        np.max(np.abs(legacy.first_pass.full_nonlinear_residual_moments[:, :2])) > 1e-6
    )


def test_tukey_is_explicit_periodic_and_disabled_by_zero_alpha():
    unity = trusted_tukey_window(16, 8, alpha=0.0, xp=np, real_dtype=np.float64)
    window = trusted_tukey_window(16, 8, alpha=0.05, xp=np, real_dtype=np.float64)

    assert np.array_equal(unity, np.ones((16, 8)))
    assert window.shape == (16, 8)
    assert window[0, 0] == 0.0
    assert np.max(window) == 1.0
    assert np.min(window) == 0.0


def test_seeded_volume_noise_replays_deterministically():
    request = _request(mode=PR_STREAMING_LEGACY_LINEARIZED_REFERENCE)
    result = run_pr_static_streaming(
        replace(
            request,
            solver=replace(
                request.solver,
                volume_noise_epsilon=0.02,
                volume_noise_correlation_um=0.4,
                volume_noise_seed=12345,
            ),
        )
    )

    assert result.replay_diagnostics["consistent"]
    assert result.replay_diagnostics["field_max_abs_difference"] == 0.0
    assert result.memory_policy["volume_noise"]["base_seed"] == 12345


def test_explicit_per_slice_noise_seeds_are_selected_and_provenanced():
    request = _request(mode=PR_STREAMING_LEGACY_LINEARIZED_REFERENCE)
    seeds = (11, 22, 33)
    options = replace(
        request.solver,
        volume_noise_epsilon=0.02,
        volume_noise_seed=None,
        volume_noise_seeds=seeds,
    )

    assert volume_noise_seed_for_slice(options, z_index=1, longitudinal_slices=3) == 22
    result = run_pr_static_streaming(replace(request, solver=options))

    assert result.replay_diagnostics["consistent"]
    provenance = result.memory_policy["volume_noise"]
    assert provenance["seed_policy"] == "explicit_per_slice_sequence"
    assert provenance["seed_count"] == 3
    assert provenance["first_seed"] == 11
    assert provenance["last_seed"] == 33
    assert len(provenance["seed_sha256_le_u32"]) == 64


def test_explicit_noise_seed_count_must_equal_longitudinal_slices():
    request = _request()
    with pytest.raises(ValueError, match="length must equal"):
        run_pr_static_streaming(
            replace(
                request,
                solver=replace(
                    request.solver,
                    volume_noise_epsilon=0.02,
                    volume_noise_seeds=(1, 2),
                ),
            )
        )


def test_unseeded_volume_noise_is_rejected_for_deterministic_replay():
    request = _request()
    with pytest.raises(ValueError, match="volume_noise_seed"):
        run_pr_static_streaming(
            replace(
                request,
                solver=replace(request.solver, volume_noise_epsilon=0.02),
            )
        )


def test_streaming_float32_agrees_with_float64_and_preserves_power():
    reference = run_pr_static_streaming(_request())
    actual = run_pr_static_streaming(_request(precision="float32"))

    relative = np.linalg.norm(actual.A_final - reference.A_final) / np.linalg.norm(
        reference.A_final
    )
    assert actual.status == "converged"
    assert relative < 8e-7
    assert (
        abs((actual.power_final - actual.power_initial) / actual.power_initial) < 2e-6
    )


def test_streaming_cupy_agrees_with_numpy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    cpu = run_pr_static_streaming(_request())
    gpu = run_pr_static_streaming(_request(backend="cupy"))

    assert gpu.backend_summary["backend"] == "cupy"
    assert gpu.replay_diagnostics["consistent"]
    assert np.allclose(gpu.A_final, cpu.A_final, rtol=2e-11, atol=2e-12)
    assert np.allclose(
        gpu.first_pass.state_moments,
        cpu.first_pass.state_moments,
        rtol=2e-11,
        atol=2e-12,
    )


def test_scaled_streaming_image_case_returns_figure6_observables():
    y, x = np.mgrid[:12, :12]
    image = (((x - 5.5) ** 2 + (y - 5.5) ** 2) > 9.0).astype(float)
    result = run_streaming_image_amplification(
        image,
        PRImageAmplificationSpec(
            Nx=24,
            Ny=12,
            x_aperture_um=48.0,
            y_aperture_um=24.0,
            interaction_length_um=20.0,
            dz_um=10.0,
            positive_mode_index=2,
            beam_waist_um=12.0,
            input_peak_ratio=1.0,
            saturated_small_signal_gain=2.0,
            dark_intensity=0.1,
            characteristic_wavenumber_per_um=0.2,
            invert_image=False,
        ),
        solver=PRStreamingStaticOptions(
            coupled=PRStaticWorkflowOptions(
                material_solver=PRStaticSolverOptions(max_iterations=30),
                max_coupled_passes=20,
            )
        ),
    )

    assert result.run_result.status == "converged"
    assert result.output_total_intensity.shape == (24, 12)
    assert result.output_signal_field.shape == (24, 12)
    assert result.backpropagated_signal_field.shape == (24, 12)
    assert result.analytic_absolute_signal_gain == pytest.approx(4.0 / 3.0)
    assert math.isfinite(result.measured_absolute_signal_gain)
    assert math.isfinite(result.image_intensity_correlation)
    assert math.isfinite(result.normalized_image_rmse)
    assert abs(result.normalized_power_relative_drift) < 2e-13


@pytest.mark.parametrize("alpha", [-0.1, 1.1, math.inf])
def test_invalid_tukey_alpha_is_rejected(alpha):
    request = _request()
    with pytest.raises(ValueError, match="tukey_alpha"):
        run_pr_static_streaming(
            replace(
                request,
                solver=replace(request.solver, tukey_alpha=alpha),
            )
        )
