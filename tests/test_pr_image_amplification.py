from dataclasses import replace
import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.pr.image_amplification import (
    PRImageAmplificationSpec,
    isolate_signal_carrier,
    make_image_amplification_request,
    paper_absolute_signal_gain,
    paper_figure4_spec,
    paper_figure6_spec,
    prepare_image_transmission,
    run_image_amplification,
    signal_carrier_mask,
)
from lcprop.pr.specs import PR_SEMI_IMPLICIT_INTEGRATOR


def _ring_target() -> np.ndarray:
    y, x = np.mgrid[:28, :28]
    radius = np.sqrt(((x - 13.5) / 8.0) ** 2 + ((y - 13.5) / 11.0) ** 2)
    return ((radius > 0.72) & (radius < 1.0)).astype(float)


def test_paper_absolute_signal_gain_has_correct_small_and_finite_ratio_limits():
    gamma_p_L = 0.5 * math.log(4000.0)

    assert paper_absolute_signal_gain(
        input_ratio=0.0,
        gamma_p_L=gamma_p_L,
    ) == pytest.approx(4000.0)
    assert paper_absolute_signal_gain(
        input_ratio=1e-3,
        gamma_p_L=gamma_p_L,
    ) == pytest.approx(800.8)
    assert paper_absolute_signal_gain(
        input_ratio=1.0,
        gamma_p_L=gamma_p_L,
    ) == pytest.approx(8000.0 / 4001.0)


def test_paper_figure4_spec_preserves_reported_geometry_and_sampling():
    spec = paper_figure4_spec()
    grating_samples = spec.Nx / (2.0 * spec.positive_mode_index)
    external_angle = math.asin(
        spec.positive_mode_index * spec.wavelength_um / spec.x_aperture_um
    )

    assert (spec.Nx, spec.Ny) == (16384, 2048)
    assert spec.interaction_length_um == 4350.0
    assert spec.dz_um == 2.0
    assert spec.wavelength_um == 0.514
    assert spec.beam_waist_um == 3400.0
    assert spec.input_peak_ratio == 1e-5
    assert spec.saturated_small_signal_gain == 4000.0
    assert math.degrees(external_angle) == pytest.approx(7.56, abs=0.01)
    assert grating_samples == pytest.approx(8.0)


def test_real_image_preprocessing_is_bounded_and_uses_transparent_exterior():
    grid = make_grid(
        GridSpec(
            Nx=20,
            Ny=16,
            x_aperture_um=20.0,
            y_aperture_um=16.0,
            z_length_um=10.0,
            dz_um=2.0,
        ),
        real_dtype=np.float64,
    )
    image = np.zeros((4, 6), dtype=float)
    image[1:3, 2:4] = 1.0
    transmission = prepare_image_transmission(
        image,
        grid,
        center_x_um=0.0,
        center_y_um=0.0,
        physical_size_um=8.0,
        invert=False,
    )

    assert transmission.shape == (grid.Nx, grid.Ny)
    assert np.min(transmission) == 0.0
    assert np.max(transmission) == 1.0
    assert transmission[0, 0] == 1.0
    assert np.any(transmission < 1.0)


def test_figure6_inverts_before_padding_and_keeps_exterior_transparent():
    grid = make_grid(
        GridSpec(
            Nx=20,
            Ny=16,
            x_aperture_um=20.0,
            y_aperture_um=16.0,
            z_length_um=10.0,
            dz_um=2.0,
        ),
        real_dtype=np.float64,
    )
    image = np.ones((4, 6), dtype=float)
    image[1:3, 2:4] = 0.0
    transmission = prepare_image_transmission(
        image,
        grid,
        center_x_um=0.0,
        center_y_um=0.0,
        physical_size_um=8.0,
        invert=paper_figure6_spec().invert_image,
    )

    assert transmission[0, 0] == 1.0
    assert np.min(transmission) == 0.0
    assert np.max(transmission) == 1.0
    assert np.count_nonzero(transmission == 1.0) > image.size
    assert np.any(transmission[6:14, 4:12] == 0.0)
    assert np.any(transmission[6:14, 4:12] == 1.0)


def test_request_has_periodic_crossing_carriers_and_exact_peak_ratio():
    spec = PRImageAmplificationSpec(Nt=0)
    request, _transmission, normalized_grating, gain_length = (
        make_image_amplification_request(_ring_target(), spec)
    )
    channels = request.beams.channels
    peak_ratio = float(
        np.max(np.abs(request.initial_A[1]) ** 2)
        / np.max(np.abs(request.initial_A[0]) ** 2)
    )

    assert request.solver.integrator == PR_SEMI_IMPLICIT_INTEGRATOR
    assert channels[0].tilt_x_rad_per_um > 0.0
    assert channels[1].tilt_x_rad_per_um < 0.0
    assert channels[0].x0_um < 0.0
    assert channels[1].x0_um > 0.0
    assert peak_ratio == pytest.approx(spec.input_peak_ratio, rel=2e-13)
    assert normalized_grating < 0.0
    assert gain_length < 0.0
    assert np.sum(np.abs(request.initial_A) ** 2) * (
        request.grid.x_aperture_um / request.grid.Nx
    ) * (
        request.grid.y_aperture_um / request.grid.Ny
    ) == pytest.approx(1.0)


def test_explicit_backend_override_reaches_image_workflow():
    result = run_image_amplification(
        _ring_target(),
        replace(PRImageAmplificationSpec(), Nt=0),
        backend=BackendSpec(
            backend="numpy",
            precision="float32",
            verbose=False,
        ),
    )

    assert result.request.backend.backend == "numpy"
    assert result.request.backend.precision == "float32"
    assert result.run_result.diagnostics["backend"] == {
        "backend": "numpy",
        "real_dtype": "float32",
        "complex_dtype": "complex64",
        "is_gpu": False,
    }


def test_nearest_carrier_partition_recovers_known_periodic_signal_mode():
    spec = PRImageAmplificationSpec(Nt=0)
    request, _transmission, _normalized_grating, _gain_length = (
        make_image_amplification_request(np.ones((8, 8)), spec)
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = request.beams.channels[0].tilt_x_rad_per_um
    signal_kx = request.beams.channels[1].tilt_x_rad_per_um
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    signal = np.exp(1j * signal_kx * np.asarray(grid.x_um)[:, None]) * np.ones(
        (1, grid.Ny)
    )

    recovered = isolate_signal_carrier(signal, mask)

    assert np.allclose(recovered, signal, rtol=0.0, atol=2e-14)


@pytest.fixture(scope="module")
def scaled_image_result():
    return run_image_amplification(_ring_target())


def test_scaled_image_benchmark_amplifies_and_preserves_the_image(scaled_image_result):
    result = scaled_image_result

    assert result.run_result.status == "completed"
    assert result.analytic_absolute_signal_gain == pytest.approx(
        10.0 * 1.001 / 1.01
    )
    assert result.measured_absolute_signal_gain > 6.0
    assert result.measured_absolute_signal_gain < result.analytic_absolute_signal_gain
    assert result.image_intensity_correlation > 0.96
    assert result.zero_response_image_intensity_correlation == pytest.approx(
        1.0,
        abs=2e-13,
    )
    assert result.normalized_image_rmse < 0.45
    assert abs(result.normalized_power_relative_drift) < 2e-12


def test_reversing_coupling_direction_reverses_signal_transfer():
    amplified = run_image_amplification(
        _ring_target(),
        replace(
            PRImageAmplificationSpec(),
            Nt=200,
            saturated_small_signal_gain=10.0,
            signal_gain_sign=1,
        ),
    )
    deamplified = run_image_amplification(
        _ring_target(),
        replace(
            PRImageAmplificationSpec(),
            Nt=200,
            saturated_small_signal_gain=10.0,
            signal_gain_sign=-1,
        ),
    )

    assert amplified.measured_absolute_signal_gain > 1.0
    assert deamplified.measured_absolute_signal_gain < 1.0
    assert amplified.analytic_gamma_p_L > 0.0
    assert deamplified.analytic_gamma_p_L < 0.0
