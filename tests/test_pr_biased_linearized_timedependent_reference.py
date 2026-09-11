from __future__ import annotations

import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.linearized_timedependent_reference import (
    biased_linearized_timedependent_fourier_symbol,
    linearized_timedependent_rhs,
    solve_pr_biased_linearized_timedependent_reference,
)
from lcprop.pr.transverse.transport import (
    potential_rhs,
    spectral_derivatives,
    spectral_wavevectors,
)


def _backend(precision: str = "float64", backend: str = "numpy") -> BackendSpec:
    return BackendSpec(backend=backend, precision=precision, verbose=False)


def _grid(shape=(31, 33)):
    nx, ny = shape
    dx = 2.0 * math.pi / nx
    dy = 2.0 * math.pi / ny
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    return dx, dy, x, y


def _spec(shape=(31, 33), applied_field=0.7):
    dx, dy, _, _ = _grid(shape)
    return PRBiasedLinearizedReferenceSpec(
        reference_intensity=1.3,
        applied_field=applied_field,
        dx_normalized=dx,
        dy_normalized=dy,
        m_y=1.4,
        h_y=2.1,
    )


def _coefficient(field, mode_x, mode_y):
    return np.fft.fft2(field)[mode_x, mode_y] / field.size


@pytest.mark.parametrize(
    ("mode_x", "mode_y", "applied_field"),
    [(2, 0, 0.8), (0, 3, -0.8), (2, 3, 0.8)],
)
def test_modal_coefficients_give_exact_transient_amplitude_phase_and_decay(
    mode_x,
    mode_y,
    applied_field,
):
    shape = (31, 33)
    spec = _spec(shape, applied_field)
    _, _, x, y = _grid(shape)
    forcing_amplitude = 0.018
    forcing_phase = 0.37
    initial_amplitude = 0.011
    initial_phase = -0.29
    theta = mode_x * x + mode_y * y
    intensity = (
        spec.reference_intensity
        + forcing_amplitude * np.cos(theta + forcing_phase)
    ).astype(np.float64)
    initial = (
        initial_amplitude * np.cos(theta + initial_phase)
    ).astype(np.float64)
    time = 0.43

    result = solve_pr_biased_linearized_timedependent_reference(
        intensity,
        time_normalized=time,
        spec=spec,
        initial_delta_psi=initial,
    )
    a_M = mode_x**2 + spec.m_y * mode_y**2
    a_H = mode_x**2 + spec.h_y * mode_y**2
    decay = spec.reference_intensity * (
        a_M * (1.0 + a_H) + 1j * applied_field * mode_x * a_H
    ) / a_H
    source = -(a_M + 1j * applied_field * mode_x) / a_H
    forcing_hat = 0.5 * forcing_amplitude * np.exp(1j * forcing_phase)
    initial_hat = 0.5 * initial_amplitude * np.exp(1j * initial_phase)
    equilibrium_hat = source * forcing_hat / decay
    expected = equilibrium_hat + (initial_hat - equilibrium_hat) * np.exp(
        -decay * time
    )

    actual = _coefficient(result.delta_psi, mode_x, mode_y)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-17)
    assert result.delta_psi.dtype == np.float64
    assert result.decay_rate.dtype == np.complex128
    assert result.source_kernel.dtype == np.complex128
    assert result.equilibrium_kernel.dtype == np.complex128
    assert result.real_dtype == "float64"
    assert result.complex_dtype == "complex128"
    symbol = biased_linearized_timedependent_fourier_symbol(shape, spec=spec)
    np.testing.assert_allclose(
        symbol.decay_rate[mode_x, mode_y], decay, rtol=2e-15, atol=2e-15
    )
    np.testing.assert_allclose(
        symbol.source_kernel[mode_x, mode_y], source, rtol=2e-15, atol=2e-15
    )
    assert decay.real > 0.0
    np.testing.assert_allclose(decay.imag, spec.reference_intensity * applied_field * mode_x)


def test_bias_reversal_conjugates_modal_dynamics_and_transient():
    shape = (31, 33)
    common = _spec(shape, 0.75)
    positive = biased_linearized_timedependent_fourier_symbol(shape, spec=common)
    negative_spec = PRBiasedLinearizedReferenceSpec(
        common.reference_intensity,
        -common.applied_field,
        common.dx_normalized,
        common.dy_normalized,
        common.m_y,
        common.h_y,
    )
    negative = biased_linearized_timedependent_fourier_symbol(
        shape, spec=negative_spec
    )
    np.testing.assert_array_equal(negative.decay_rate, positive.decay_rate.conjugate())
    np.testing.assert_array_equal(
        negative.source_kernel, positive.source_kernel.conjugate()
    )
    resolved = positive.resolved_mask
    np.testing.assert_allclose(
        positive.source_kernel[resolved] / positive.decay_rate[resolved],
        positive.equilibrium_kernel[resolved],
        rtol=6e-16,
        atol=0.0,
    )
    assert np.all(positive.decay_rate.real[resolved] > 0.0)

    _, _, x, y = _grid(shape)
    intensity = (common.reference_intensity + 0.02 * np.cos(2 * x + 3 * y)).astype(
        np.float64
    )
    plus = solve_pr_biased_linearized_timedependent_reference(
        intensity, time_normalized=0.31, spec=common
    )
    minus = solve_pr_biased_linearized_timedependent_reference(
        intensity, time_normalized=0.31, spec=negative_spec
    )
    np.testing.assert_allclose(
        _coefficient(minus.delta_psi, 2, 3),
        _coefficient(plus.delta_psi, 2, 3).conjugate(),
        rtol=2e-14,
        atol=2e-17,
    )


def test_zero_arbitrary_and_equilibrium_initial_conditions():
    shape = (31, 33)
    spec = _spec(shape, 0.6)
    _, _, x, y = _grid(shape)
    uniform = np.full(shape, spec.reference_intensity, dtype=np.float64)
    zero = solve_pr_biased_linearized_timedependent_reference(
        uniform, time_normalized=0.0, spec=spec
    )
    assert np.array_equal(zero.delta_psi, np.zeros(shape))
    assert np.array_equal(zero.delta_psi_initial, np.zeros(shape))

    initial = (0.03 * np.cos(2 * x - 3 * y + 0.2)).astype(np.float64)
    decayed = solve_pr_biased_linearized_timedependent_reference(
        uniform,
        time_normalized=0.27,
        spec=spec,
        initial_delta_psi=initial,
    )
    symbol = biased_linearized_timedependent_fourier_symbol(shape, spec=spec)
    expected_coefficient = _coefficient(initial, 2, -3) * np.exp(
        -symbol.decay_rate[2, -3] * 0.27
    )
    np.testing.assert_allclose(
        _coefficient(decayed.delta_psi, 2, -3),
        expected_coefficient,
        rtol=2e-14,
        atol=2e-17,
    )

    intensity = (
        spec.reference_intensity
        + 0.02 * np.cos(2 * x)
        - 0.015 * np.sin(3 * y)
        + 0.01 * np.cos(x + 2 * y)
    ).astype(np.float64)
    static = solve_pr_biased_linearized_reference(intensity, spec=spec)
    equilibrium = solve_pr_biased_linearized_timedependent_reference(
        intensity,
        time_normalized=1.7,
        spec=spec,
        initial_delta_psi=static.delta_psi,
    )
    np.testing.assert_allclose(
        equilibrium.delta_psi, static.delta_psi, rtol=0.0, atol=2e-17
    )


def test_batched_planes_are_independent():
    shape = (29, 31)
    spec = _spec(shape, -0.55)
    _, _, x, y = _grid(shape)
    intensity = np.stack(
        (
            spec.reference_intensity + 0.02 * np.cos(2 * x) + 0.0 * y,
            spec.reference_intensity - 0.015 * np.sin(3 * y) + 0.0 * x,
        )
    ).astype(np.float64)
    initial = np.stack(
        (0.01 * np.sin(x + y), -0.008 * np.cos(2 * x - y))
    ).astype(np.float64)
    batched = solve_pr_biased_linearized_timedependent_reference(
        intensity,
        time_normalized=0.22,
        spec=spec,
        initial_delta_psi=initial,
    )
    for batch in range(2):
        separate = solve_pr_biased_linearized_timedependent_reference(
            intensity[batch],
            time_normalized=0.22,
            spec=spec,
            initial_delta_psi=initial[batch],
        )
        np.testing.assert_allclose(
            batched.delta_psi[batch], separate.delta_psi, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("shape", [(7, 9), (8, 9), (7, 10), (8, 10)])
def test_even_odd_grids_project_all_and_only_derivative_null_modes(shape):
    spec = _spec(shape, 0.7)
    _, _, x, y = _grid(shape)
    resolved = np.cos(x + y)
    null = np.ones(shape)
    if shape[0] % 2 == 0:
        null = null + (-1.0) ** np.arange(shape[0])[:, None]
    if shape[1] % 2 == 0:
        null = null + (-1.0) ** np.arange(shape[1])[None, :]
    if shape[0] % 2 == 0 and shape[1] % 2 == 0:
        null = null + (
            (-1.0) ** np.arange(shape[0])[:, None]
            * (-1.0) ** np.arange(shape[1])[None, :]
        )
    intensity_resolved = (spec.reference_intensity + 0.01 * resolved).astype(
        np.float64
    )
    intensity_with_null = (intensity_resolved + 0.001 * null).astype(np.float64)
    initial_resolved = (0.02 * np.sin(x - y)).astype(np.float64)
    initial_with_null = (initial_resolved + 0.002 * null).astype(np.float64)
    expected = solve_pr_biased_linearized_timedependent_reference(
        intensity_resolved,
        time_normalized=0.19,
        spec=spec,
        initial_delta_psi=initial_resolved,
    )
    actual = solve_pr_biased_linearized_timedependent_reference(
        intensity_with_null,
        time_normalized=0.19,
        spec=spec,
        initial_delta_psi=initial_with_null,
    )
    symbol = biased_linearized_timedependent_fourier_symbol(shape, spec=spec)
    expected_null_count = math.prod(2 if size % 2 == 0 else 1 for size in shape)
    assert np.count_nonzero(~symbol.resolved_mask) == expected_null_count
    assert np.all(symbol.decay_rate[~symbol.resolved_mask] == 0.0)
    assert np.all(symbol.source_kernel[~symbol.resolved_mask] == 0.0)
    np.testing.assert_allclose(actual.delta_psi, expected.delta_psi, rtol=0, atol=2e-16)
    assert abs(float(np.mean(actual.delta_psi))) < 2e-18


def test_long_time_multimode_limit_matches_static_reference_near_machine_precision():
    shape = (32, 35)
    spec = _spec(shape, 0.85)
    _, _, x, y = _grid(shape)
    intensity = (
        spec.reference_intensity
        + 0.02 * np.cos(2 * x)
        - 0.015 * np.sin(3 * y)
        + 0.012 * np.cos(2 * x - 3 * y)
    ).astype(np.float64)
    initial = (0.009 * np.sin(x + 2 * y)).astype(np.float64)
    static = solve_pr_biased_linearized_reference(intensity, spec=spec)
    transient = solve_pr_biased_linearized_timedependent_reference(
        intensity,
        time_normalized=40.0,
        spec=spec,
        initial_delta_psi=initial,
    )
    np.testing.assert_allclose(transient.delta_psi, static.delta_psi, rtol=0, atol=2e-17)
    np.testing.assert_allclose(
        transient.delta_psi_equilibrium, static.delta_psi, rtol=0, atol=2e-17
    )


def test_short_time_derivative_matches_independent_linearized_rhs():
    shape = (31, 33)
    spec = _spec(shape, -0.65)
    _, _, x, y = _grid(shape)
    intensity = (
        spec.reference_intensity
        + 0.017 * np.cos(2 * x + y)
        - 0.011 * np.sin(3 * y)
    ).astype(np.float64)
    initial = (0.013 * np.cos(x - 2 * y + 0.4)).astype(np.float64)
    rhs = linearized_timedependent_rhs(initial, intensity, spec=spec)
    dt = 1.0e-7
    evolved = solve_pr_biased_linearized_timedependent_reference(
        intensity,
        time_normalized=dt,
        spec=spec,
        initial_delta_psi=initial,
    )
    finite_difference = (evolved.delta_psi - evolved.delta_psi_initial) / dt
    relative_error = np.linalg.norm(finite_difference - rhs) / np.linalg.norm(rhs)
    assert relative_error < 2.0e-6


def test_nonlinear_rhs_taylor_remainder_is_second_order():
    shape = (31, 33)
    spec = _spec(shape, 0.7)
    _, _, x, y = _grid(shape)
    intensity_direction = 0.07 * np.cos(2 * x + 3 * y) - 0.03 * np.sin(y)
    potential_direction = 0.04 * np.cos(x - 2 * y + 0.2)
    tangent = linearized_timedependent_rhs(
        potential_direction.astype(np.float64),
        (spec.reference_intensity + intensity_direction).astype(np.float64),
        spec=spec,
    )
    errors = []
    for epsilon in (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        nonlinear = potential_rhs(
            (epsilon * potential_direction).astype(np.float64),
            (spec.reference_intensity + epsilon * intensity_direction).astype(
                np.float64
            ),
            dx_normalized=spec.dx_normalized,
            dy_normalized=spec.dy_normalized,
            m_y=spec.m_y,
            h_y=spec.h_y,
            applied_field_x=spec.applied_field,
            xp=np,
        )
        errors.append(float(np.linalg.norm(nonlinear - epsilon * tangent)))
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    assert all(1.99 < order < 2.01 for order in orders)


def test_reconstructed_fields_are_finite_zero_mean_curl_free_and_gauss_consistent():
    shape = (31, 33)
    spec = _spec(shape, 0.5)
    _, _, x, y = _grid(shape)
    intensity = (
        spec.reference_intensity + 0.02 * np.cos(2 * x - 3 * y)
    ).astype(np.float64)
    result = solve_pr_biased_linearized_timedependent_reference(
        intensity, time_normalized=0.4, spec=spec
    )
    kx, ky = spectral_wavevectors(
        shape,
        dx_normalized=spec.dx_normalized,
        dy_normalized=spec.dy_normalized,
    )
    E_y_x, _ = spectral_derivatives(result.delta_E_y, kx=kx, ky=ky)
    _, E_x_y = spectral_derivatives(result.delta_E_x, kx=kx, ky=ky)
    E_x_x, _ = spectral_derivatives(result.delta_E_x, kx=kx, ky=ky)
    _, E_y_y = spectral_derivatives(result.delta_E_y, kx=kx, ky=ky)
    curl = E_y_x - E_x_y
    gauss = result.delta_P - E_x_x - spec.h_y * E_y_y

    for value in (
        result.delta_psi,
        result.delta_E_x,
        result.delta_E_y,
        result.delta_P,
    ):
        assert np.all(np.isfinite(value))
        assert abs(float(np.mean(value))) < 3e-18
    assert np.max(np.abs(curl)) < 2e-16
    assert np.max(np.abs(gauss)) < 3e-15


def test_numpy_float32_preserves_dtype_and_matches_float64():
    shape = (32, 35)
    spec = _spec(shape, 0.85)
    _, _, x, y = _grid(shape)
    intensity64 = (
        spec.reference_intensity
        + 0.02 * np.cos(2 * x)
        - 0.015 * np.sin(3 * y)
        + 0.012 * np.cos(2 * x - 3 * y)
    ).astype(np.float64)
    initial64 = (0.009 * np.sin(x + 2 * y)).astype(np.float64)
    result64 = solve_pr_biased_linearized_timedependent_reference(
        intensity64,
        time_normalized=0.37,
        spec=spec,
        initial_delta_psi=initial64,
        backend=_backend("float64"),
    )
    result32 = solve_pr_biased_linearized_timedependent_reference(
        intensity64.astype(np.float32),
        time_normalized=0.37,
        spec=spec,
        initial_delta_psi=initial64.astype(np.float32),
        backend=_backend("float32"),
    )
    for name in (
        "delta_psi_initial",
        "delta_psi",
        "delta_psi_equilibrium",
        "delta_E_x",
        "delta_E_y",
        "delta_P",
    ):
        value32 = getattr(result32, name)
        assert value32.dtype == np.float32
        np.testing.assert_allclose(value32, getattr(result64, name), rtol=4e-5, atol=2e-6)
    assert result32.decay_rate.dtype == np.complex64
    assert result32.source_kernel.dtype == np.complex64
    assert result32.equilibrium_kernel.dtype == np.complex64
    assert result32.real_dtype == "float32"
    assert result32.complex_dtype == "complex64"


def test_cupy_matches_numpy_when_a_local_device_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no local CUDA device")
        _ = cp.arange(1)
    except Exception as error:
        pytest.skip(f"local CuPy/CUDA unavailable: {error}")
    shape = (15, 17)
    spec = _spec(shape, 0.6)
    _, _, x, y = _grid(shape)
    intensity = (spec.reference_intensity + 0.02 * np.cos(2 * x + y)).astype(
        np.float64
    )
    numpy_result = solve_pr_biased_linearized_timedependent_reference(
        intensity, time_normalized=0.2, spec=spec
    )
    cupy_result = solve_pr_biased_linearized_timedependent_reference(
        cp.asarray(intensity),
        time_normalized=0.2,
        spec=spec,
        backend=_backend("float64", "cupy"),
    )
    assert cupy_result.resolved_backend == "cupy"
    assert cupy_result.is_gpu
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_psi),
        numpy_result.delta_psi,
        rtol=3e-14,
        atol=3e-16,
    )
