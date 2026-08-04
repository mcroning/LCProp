from dataclasses import replace
import math

import numpy as np
import pytest

from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.optics.splitstep import hop_linear_inplace, linear_kernel
from lcprop.pr.coupling import (
    FiniteGaussianCouplingSpec,
    PlaneWaveCouplingSpec,
    analytic_plane_wave_gain_length,
    make_plane_wave_coupling_request,
    project_fourier_modes,
    run_finite_gaussian_coupling,
    run_plane_wave_coupling,
)
from lcprop.pr.geometry import (
    analyze_crossing_aperture,
    crossing_beam_channels,
    paraxial_kernel_slope,
)


def _axis_centroid(field, coordinate, *, axis):
    intensity = np.abs(field) ** 2
    marginal = np.sum(intensity, axis=1 - axis)
    return float(np.sum(marginal * coordinate) / np.sum(marginal))


@pytest.mark.parametrize(
    ("azimuths", "axis", "tilt_name"),
    (((0.0, math.pi), 0, "tilt_x_rad_per_um"),
     ((math.pi / 2.0, 3.0 * math.pi / 2.0), 1, "tilt_y_rad_per_um")),
    ids=("positive-negative-x", "positive-negative-y"),
)
def test_linear_kernel_rays_cross_at_requested_midpoint(azimuths, axis, tilt_name):
    wavelength = 0.633
    index = 1.6
    length = 80.0
    crossing_z = length / 2.0
    theta = 0.06
    channels = crossing_beam_channels(
        wavelength_um=wavelength,
        refractive_index=index,
        interaction_length_um=length,
        polar_angles_rad=(theta, theta),
        azimuths_rad=azimuths,
        waist_x_um=4.0,
        waist_y_um=4.0,
        crossing_z_um=crossing_z,
        beam_ratio=1.0,
    )
    positive_tilt = getattr(channels[0], tilt_name)
    negative_tilt = getattr(channels[1], tilt_name)
    assert positive_tilt > 0.0
    assert negative_tilt < 0.0
    launch_coordinate = "x0_um" if axis == 0 else "y0_um"
    assert getattr(channels[0], launch_coordinate) < 0.0
    assert getattr(channels[1], launch_coordinate) > 0.0
    assert paraxial_kernel_slope(
        transverse_phase_gradient_rad_per_um=positive_tilt,
        wavelength_um=wavelength,
        refractive_index=index,
    ) == pytest.approx(math.sin(theta))

    grid = make_grid(
        GridSpec(
            Nx=128,
            Ny=128,
            x_aperture_um=80.0,
            y_aperture_um=80.0,
            z_length_um=length,
            dz_um=2.0,
        ),
        real_dtype=np.float64,
    )
    A = build_launch(
        BeamStack(channels=channels, coherence="coherent"),
        grid,
        complex_dtype=np.complex128,
    ).A0.copy()
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=wavelength,
        n_ref=index,
        xp=np,
    )
    for _ in range(int(round(crossing_z / grid.dz_um))):
        hop_linear_inplace(A, kernel, xp=np)
    coordinate = np.asarray(grid.x_um if axis == 0 else grid.y_um)
    centroids = tuple(
        _axis_centroid(A[channel], coordinate, axis=axis) for channel in range(2)
    )
    assert centroids[0] == pytest.approx(0.0, abs=0.05)
    assert centroids[1] == pytest.approx(0.0, abs=0.05)
    assert abs(centroids[0] - centroids[1]) < 0.05


def test_crossing_and_sampling_validation_failures_are_explicit():
    with pytest.raises(ValueError, match="within the interaction region"):
        crossing_beam_channels(
            wavelength_um=0.633,
            refractive_index=1.6,
            interaction_length_um=100.0,
            polar_angles_rad=(0.03, 0.03),
            azimuths_rad=(0.0, math.pi),
            waist_x_um=8.0,
            waist_y_um=8.0,
            crossing_z_um=101.0,
        )

    channels = crossing_beam_channels(
        wavelength_um=0.633,
        refractive_index=1.6,
        interaction_length_um=100.0,
        polar_angles_rad=(0.15, 0.15),
        azimuths_rad=(0.0, math.pi),
        waist_x_um=8.0,
        waist_y_um=8.0,
    )
    grid = make_grid(
        GridSpec(
            Nx=16,
            Ny=16,
            x_aperture_um=40.0,
            y_aperture_um=40.0,
            z_length_um=100.0,
            dz_um=20.0,
        )
    )
    report = analyze_crossing_aperture(
        grid,
        channels,
        refractive_index=1.6,
        crossing_z_um=50.0,
        strict=False,
    )
    assert "beam waist is inadequately sampled" in report.warnings
    assert "two-beam grating period is inadequately sampled" in report.warnings
    assert "interaction region has too few longitudinal steps" in report.warnings
    with pytest.raises(ValueError, match="inadequately sampled"):
        analyze_crossing_aperture(
            grid,
            channels,
            refractive_index=1.6,
            crossing_z_um=50.0,
            strict=True,
        )


@pytest.fixture(scope="module")
def plane_result():
    return run_plane_wave_coupling()


def test_plane_wave_modal_projection_recovers_known_input_ratio():
    spec = PlaneWaveCouplingSpec(Nt=0, gain_length_product=0.0)
    request, kx, _ = make_plane_wave_coupling_request(spec)
    grid = make_grid(request.grid, real_dtype=np.float64)
    amplitudes, powers = project_fourier_modes(
        request.initial_A,
        grid,
        ((kx[0], 0.0), (kx[1], 0.0)),
    )
    assert powers[1] / powers[0] == pytest.approx(spec.beam_ratio, abs=1e-14)
    assert abs(amplitudes[0]) ** 2 + abs(amplitudes[1]) ** 2 == pytest.approx(1.0)


def test_paper_plane_wave_analytic_gain_relation():
    # Section 3.1 quotes kg/k0=0.76 and gamma*L=3, giving about 2.89.
    predicted = analytic_plane_wave_gain_length(
        gain_length_product=3.0,
        signed_grating_k_normalized=0.76,
        internal_half_angle_rad=0.0,
    )
    assert predicted == pytest.approx(2.890465, rel=1e-6)


def test_zero_pr_coupling_leaves_modal_powers_unchanged():
    result = run_plane_wave_coupling(
        PlaneWaveCouplingSpec(Nt=20, gain_length_product=0.0)
    )
    assert result.output_modal_powers == pytest.approx(
        result.input_modal_powers,
        rel=1e-13,
        abs=1e-13,
    )
    assert result.output_total_power == pytest.approx(
        result.input_total_power,
        rel=1e-13,
    )
    assert result.gamma_p_L_measured == pytest.approx(0.0, abs=1e-13)


def test_plane_wave_gain_matches_paper_relation(plane_result):
    assert plane_result.gamma_p_L_measured < 0.0
    assert plane_result.gamma_p_L_expected < 0.0
    assert plane_result.gamma_p_L_measured == pytest.approx(
        plane_result.gamma_p_L_expected,
        rel=0.03,
    )


def test_gain_sign_reversal_and_total_power_conservation(plane_result):
    reversed_result = run_plane_wave_coupling(
        replace(PlaneWaveCouplingSpec(), gain_length_product=-0.05)
    )
    assert reversed_result.gamma_p_L_measured > 0.0
    assert reversed_result.gamma_p_L_measured == pytest.approx(
        -plane_result.gamma_p_L_measured,
        rel=0.03,
    )
    for result in (plane_result, reversed_result):
        assert result.output_total_power == pytest.approx(
            result.input_total_power,
            rel=5e-13,
        )


def test_plane_wave_gain_converges_with_dz():
    base = PlaneWaveCouplingSpec(Nt=300)
    gains = [
        run_plane_wave_coupling(replace(base, dz_um=dz)).gamma_p_L_measured
        for dz in (10.0, 5.0, 2.5)
    ]
    assert abs(gains[2] - gains[1]) < abs(gains[1] - gains[0])
    assert abs(gains[2] - gains[1]) < 2e-6


def test_material_time_approaches_steady_plane_wave_gain():
    base = PlaneWaveCouplingSpec()
    results = [
        run_plane_wave_coupling(replace(base, Nt=Nt))
        for Nt in (50, 300, 600, 900)
    ]
    magnitudes = [abs(result.gamma_p_L_measured) for result in results]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[-1] - magnitudes[-2] < magnitudes[-2] - magnitudes[-3]
    assert results[-1].gamma_p_L_measured == pytest.approx(
        results[-1].gamma_p_L_expected,
        rel=0.03,
    )


def test_plane_wave_request_is_reproducible(plane_result):
    repeated = run_plane_wave_coupling()
    assert repeated.gamma_p_L_measured == pytest.approx(
        plane_result.gamma_p_L_measured,
        rel=0.0,
        abs=1e-14,
    )
    assert repeated.output_modal_powers == pytest.approx(
        plane_result.output_modal_powers,
        rel=0.0,
        abs=1e-12,
    )


def test_finite_gaussian_crossing_smoke_benchmark():
    result = run_finite_gaussian_coupling(
        FiniteGaussianCouplingSpec(Nt=60)
    )
    assert result.aperture.warnings == ()
    assert result.aperture.grating_samples_per_period > 10.0
    assert min(value for pair in result.aperture.samples_per_waist for value in pair) >= 6.0
    assert abs(result.crossing_error_um) < 0.1
    assert np.max(result.grating_amplitude_by_z) > 0.0
    assert result.output_beam_ratio > 0.0
    assert result.trace.coherent_total_power[-1] == pytest.approx(
        result.trace.coherent_total_power[0],
        rel=2e-12,
    )
    expected_planes = int(
        round(result.request.grid.z_length_um / result.request.grid.dz_um)
    ) + 1
    assert result.trace.matched_mode_powers.shape == (expected_planes, 2)
