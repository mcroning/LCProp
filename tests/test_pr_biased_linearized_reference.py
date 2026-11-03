from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.pr.transverse.linearized_reference import (
    FIXED_MEAN_FIELD_ENSEMBLE,
    PR_PERIODIC_BIASED_CURRENT_PROFILE_V1,
    PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1,
    PRBiasedLinearizedReferenceSpec,
    biased_linearized_fourier_symbol,
    solve_pr_biased_linearized_reference,
    total_fields_from_perturbation,
)
from lcprop.pr.transverse.transport import potential_rhs


def _cupy_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no local CUDA device")
        _ = cp.arange(1)
    except Exception as error:
        pytest.skip(f"local CuPy/CUDA unavailable: {error}")
    return cp


def _backend_spec(backend: str, precision: str = "float64") -> BackendSpec:
    return BackendSpec(backend=backend, precision=precision, verbose=False)


def _mode_case(
    *,
    mode_x: int,
    mode_y: int,
    applied_field: float,
    epsilon: float = 2.0e-4,
    reference_intensity: float = 1.3,
    m_y: float = 1.4,
    h_y: float = 2.1,
):
    nx, ny = 33, 35
    length_x = length_y = 2.0 * math.pi
    dx = length_x / nx
    dy = length_y / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    theta = mode_x * X + mode_y * Y
    intensity = (
        reference_intensity + epsilon * np.cos(theta)
    ).astype(np.float64)
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=reference_intensity,
        applied_field=applied_field,
        dx_normalized=dx,
        dy_normalized=dy,
        m_y=m_y,
        h_y=h_y,
    )
    result = solve_pr_biased_linearized_reference(intensity, spec=spec)

    kx = float(mode_x)
    ky = float(mode_y)
    a_M = kx * kx + m_y * ky * ky
    a_H = kx * kx + h_y * ky * ky
    response = -(a_M + 1j * applied_field * kx) / (
        reference_intensity
        * (a_M * (1.0 + a_H) + 1j * applied_field * kx * a_H)
    )
    expected_psi = epsilon * (
        response.real * np.cos(theta) - response.imag * np.sin(theta)
    )
    common_field = epsilon * (
        response.real * np.sin(theta) + response.imag * np.cos(theta)
    )
    return result, expected_psi, kx * common_field, ky * common_field, a_H


@pytest.mark.parametrize(
    ("mode_x", "mode_y", "applied_field"),
    [
        (2, 0, 0.0),
        (0, 3, 0.0),
        (2, 0, 0.8),
        (2, 0, -0.8),
        (2, 3, 0.8),
    ],
)
def test_analytic_single_modes_match_amplitude_phase_and_fields(
    mode_x,
    mode_y,
    applied_field,
):
    result, expected_psi, expected_E_x, expected_E_y, a_H = _mode_case(
        mode_x=mode_x,
        mode_y=mode_y,
        applied_field=applied_field,
    )

    np.testing.assert_allclose(result.delta_psi, expected_psi, rtol=0, atol=2e-16)
    np.testing.assert_allclose(result.delta_E_x, expected_E_x, rtol=0, atol=4e-16)
    np.testing.assert_allclose(result.delta_E_y, expected_E_y, rtol=0, atol=4e-16)
    np.testing.assert_allclose(result.delta_P, a_H * expected_psi, rtol=0, atol=3e-15)
    assert abs(float(np.mean(result.delta_psi))) < 2e-20
    assert abs(float(np.mean(result.delta_P))) < 2e-19


def test_bias_reversal_conjugates_kernel_and_reverses_phase_skew():
    common = dict(
        reference_intensity=1.2,
        dx_normalized=0.3,
        dy_normalized=0.4,
        m_y=1.6,
        h_y=2.2,
    )
    positive = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=0.75, **common),
    )
    negative = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=-0.75, **common),
    )

    np.testing.assert_array_equal(
        negative.response_kernel,
        positive.response_kernel.conjugate(),
    )
    np.testing.assert_array_equal(
        negative.denominator,
        positive.denominator.conjugate(),
    )

    plus, expected_plus, *_ = _mode_case(
        mode_x=2,
        mode_y=1,
        applied_field=0.75,
    )
    minus, expected_minus, *_ = _mode_case(
        mode_x=2,
        mode_y=1,
        applied_field=-0.75,
    )
    np.testing.assert_allclose(plus.delta_psi, expected_plus, rtol=0, atol=2e-16)
    np.testing.assert_allclose(minus.delta_psi, expected_minus, rtol=0, atol=2e-16)
    assert np.linalg.norm(plus.delta_psi - minus.delta_psi) > 1.0e-8


def test_unbiased_kernel_is_exact_screened_poisson_response():
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=0.9,
        applied_field=0.0,
        dx_normalized=0.2,
        dy_normalized=0.35,
        m_y=1.7,
        h_y=2.4,
    )
    symbol = biased_linearized_fourier_symbol((31, 33), spec=spec)
    expected = np.zeros_like(symbol.response_kernel)
    expected[symbol.resolved_mask] = -1.0 / (
        spec.reference_intensity * (1.0 + symbol.a_H[symbol.resolved_mask])
    )

    np.testing.assert_allclose(
        symbol.response_kernel,
        expected,
        rtol=4.0e-16,
        atol=0.0,
    )


def test_y_independent_solution_matches_reduced_fixed_field_with_a7_condition():
    reference_intensity = 0.8
    background_intensity = reference_intensity
    applied_field = -0.6
    equilibrium_field_a7 = applied_field * background_intensity / reference_intensity
    assert equilibrium_field_a7 == applied_field

    result, _, _, _, _ = _mode_case(
        mode_x=3,
        mode_y=0,
        applied_field=applied_field,
        epsilon=1.0e-4,
        reference_intensity=reference_intensity,
        m_y=1.0,
        h_y=1.0,
    )
    nx, ny = result.delta_E_x.shape
    x = np.arange(nx) * (2.0 * math.pi / nx)
    theta = 3.0 * x[:, None]
    reduced_response = (3.0j - applied_field) / (
        reference_intensity * (1.0 + 9.0 + 3.0j * applied_field)
    )
    expected_E_x = 1.0e-4 * (
        reduced_response.real * np.cos(theta)
        - reduced_response.imag * np.sin(theta)
    )
    expected_E_x = np.repeat(expected_E_x, ny, axis=1)

    np.testing.assert_allclose(result.delta_E_x, expected_E_x, rtol=0, atol=5e-16)
    np.testing.assert_allclose(result.delta_E_y, 0.0, rtol=0, atol=1e-18)


@pytest.mark.parametrize("applied_field", [-3.0, 0.0, 2.5])
@pytest.mark.parametrize("shape", [(31, 33), (32, 34)])
def test_operator_denominator_is_nonsingular_on_every_resolved_mode(
    applied_field,
    shape,
):
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=0.4,
        applied_field=applied_field,
        dx_normalized=0.15,
        dy_normalized=0.23,
        m_y=0.7,
        h_y=3.1,
    )
    symbol = biased_linearized_fourier_symbol(shape, spec=spec)

    assert np.all(symbol.denominator.real[symbol.resolved_mask] > 0.0)
    assert np.all(np.abs(symbol.denominator[symbol.resolved_mask]) > 0.0)
    assert np.all(symbol.response_kernel[~symbol.resolved_mask] == 0.0)
    expected_null_modes = math.prod(2 if size % 2 == 0 else 1 for size in shape)
    assert np.count_nonzero(~symbol.resolved_mask) == expected_null_modes


def test_uniform_zero_mode_changes_mean_current_but_not_material_perturbation():
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=1.0,
        applied_field=0.7,
        dx_normalized=0.2,
        dy_normalized=0.3,
    )
    intensity = np.full((9, 11), 1.25, dtype=np.float64)
    result = solve_pr_biased_linearized_reference(intensity, spec=spec)

    assert np.array_equal(result.delta_psi, np.zeros_like(intensity))
    assert np.array_equal(result.delta_E_x, np.zeros_like(intensity))
    assert np.array_equal(result.delta_E_y, np.zeros_like(intensity))
    assert np.array_equal(result.delta_P, np.zeros_like(intensity))
    np.testing.assert_array_equal(result.delta_mean_current, np.array([-0.175, 0.0]))
    assert result.mean_intensity_perturbation == np.asarray(0.25)

    total_E_x, total_E_y = total_fields_from_perturbation(result)
    np.testing.assert_array_equal(total_E_x, np.full_like(intensity, 0.7))
    np.testing.assert_array_equal(total_E_y, np.zeros_like(intensity))
    assert result.model_id == PR_PERIODIC_BIASED_LINEARIZED_REFERENCE_V1
    assert result.profile_id == PR_PERIODIC_BIASED_CURRENT_PROFILE_V1
    assert result.electrical_ensemble == FIXED_MEAN_FIELD_ENSEMBLE
    assert result.forward_fft_count == 1
    assert result.inverse_fft_count == 4
    assert result.requested_backend == "numpy"
    assert result.resolved_backend == "numpy"
    assert result.real_dtype == "float64"
    assert result.complex_dtype == "complex128"
    assert result.is_gpu is False
    assert result.device_identity is None


def test_batched_planes_have_independent_mean_current_outputs():
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=1.0,
        applied_field=-0.5,
        dx_normalized=0.2,
        dy_normalized=0.3,
    )
    intensity = np.stack(
        (
            np.full((7, 9), 1.1),
            np.full((7, 9), 0.8),
        )
    ).astype(np.float64)
    result = solve_pr_biased_linearized_reference(intensity, spec=spec)

    np.testing.assert_allclose(
        result.delta_mean_current,
        np.array([[0.05, 0.0], [-0.1, 0.0]]),
        rtol=0,
        atol=8e-17,
    )


def test_nonlinear_biased_td_residual_has_second_order_taylor_remainder():
    nx = ny = 33
    length = 2.0 * math.pi
    spacing = length / nx
    x = np.arange(nx) * spacing
    X, Y = np.meshgrid(x, x, indexing="ij")
    shape = np.cos(2.0 * X + 3.0 * Y)
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=1.3,
        applied_field=0.7,
        dx_normalized=spacing,
        dy_normalized=spacing,
        m_y=1.4,
        h_y=2.1,
    )
    errors = []
    for epsilon in (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        intensity = (spec.reference_intensity + epsilon * shape).astype(np.float64)
        linearized = solve_pr_biased_linearized_reference(intensity, spec=spec)
        residual = potential_rhs(
            linearized.delta_psi,
            intensity,
            dx_normalized=spec.dx_normalized,
            dy_normalized=spec.dy_normalized,
            m_y=spec.m_y,
            h_y=spec.h_y,
            applied_field_x=spec.applied_field,
            xp=np,
        )
        errors.append(float(np.sqrt(np.mean(residual * residual))))

    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    assert all(1.99 < order < 2.01 for order in orders)


@pytest.mark.parametrize(
    ("intensity", "match"),
    [
        (np.ones((5, 5), dtype=np.float32), "requires float64"),
        (np.ones((2, 5), dtype=np.float64), "shape"),
        (
            np.ones((2, 2, 5), dtype=np.float64),
            "Nbatch.*not longitudinal",
        ),
        (np.zeros((5, 5), dtype=np.float64), "strictly positive"),
        (np.full((5, 5), np.nan, dtype=np.float64), "finite"),
    ],
)
def test_invalid_intensity_is_rejected(intensity, match):
    spec = PRBiasedLinearizedReferenceSpec(1.0, 0.0, 0.2, 0.3)
    with pytest.raises((TypeError, ValueError), match=match):
        solve_pr_biased_linearized_reference(intensity, spec=spec)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"reference_intensity": 0.0}, "reference_intensity"),
        ({"applied_field": np.inf}, "applied_field"),
        ({"dx_normalized": -1.0}, "dx_normalized"),
        ({"m_y": 0.0}, "m_y"),
        ({"h_y": np.nan}, "h_y"),
    ],
)
def test_invalid_spec_is_rejected(changes, match):
    values = dict(
        reference_intensity=1.0,
        applied_field=0.0,
        dx_normalized=0.2,
        dy_normalized=0.3,
        m_y=1.0,
        h_y=1.0,
    )
    values.update(changes)
    spec = PRBiasedLinearizedReferenceSpec(**values)
    with pytest.raises(ValueError, match=match):
        biased_linearized_fourier_symbol((5, 7), spec=spec)


def test_explicit_numpy_float32_policy_preserves_backend_native_dtype():
    spec = PRBiasedLinearizedReferenceSpec(1.0, 0.4, 0.2, 0.3)
    intensity = np.full((7, 9), 1.1, dtype=np.float32)
    result = solve_pr_biased_linearized_reference(
        intensity,
        spec=spec,
        backend=_backend_spec("numpy", "float32"),
    )

    for value in (
        result.delta_psi,
        result.delta_E_x,
        result.delta_E_y,
        result.delta_P,
        result.delta_mean_current,
        result.mean_intensity_perturbation,
    ):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
    assert result.response_kernel.dtype == np.complex64
    assert result.denominator.dtype == np.complex64
    assert result.real_dtype == "float32"
    assert result.complex_dtype == "complex64"


def test_numpy_float32_resolved_multimode_matches_float64_reference():
    nx, ny = 32, 35
    dx = 2.0 * math.pi / nx
    dy = 2.0 * math.pi / ny
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    perturbation = (
        0.025
        + 0.02 * np.cos(2.0 * x)
        - 0.015 * np.sin(3.0 * y)
        + 0.012 * np.cos(2.0 * x - 3.0 * y)
    )
    intensity64 = (1.2 + perturbation).astype(np.float64)
    intensity32 = intensity64.astype(np.float32)
    spec = PRBiasedLinearizedReferenceSpec(1.2, 0.85, dx, dy, 1.3, 2.2)
    result64 = solve_pr_biased_linearized_reference(
        intensity64, spec=spec, backend=_backend_spec("numpy")
    )
    result32 = solve_pr_biased_linearized_reference(
        intensity32, spec=spec, backend=_backend_spec("numpy", "float32")
    )

    for name in (
        "delta_psi",
        "delta_E_x",
        "delta_E_y",
        "delta_P",
        "delta_mean_current",
        "mean_intensity_perturbation",
    ):
        value32 = getattr(result32, name)
        assert value32.dtype == np.float32
        np.testing.assert_allclose(
            value32,
            getattr(result64, name),
            rtol=3e-5,
            atol=3e-6,
        )
    assert result32.response_kernel.dtype == np.complex64
    assert result32.denominator.dtype == np.complex64
    np.testing.assert_allclose(
        result32.response_kernel,
        result64.response_kernel,
        rtol=5e-6,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        result32.denominator,
        result64.denominator,
        rtol=5e-6,
        atol=5e-5,
    )


def test_numpy_float32_bias_reversal_conjugates_resolved_symbol():
    common = dict(
        reference_intensity=1.2,
        dx_normalized=0.3,
        dy_normalized=0.4,
        m_y=1.6,
        h_y=2.2,
    )
    positive = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=0.75, **common),
        backend=_backend_spec("numpy", "float32"),
    )
    negative = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=-0.75, **common),
        backend=_backend_spec("numpy", "float32"),
    )

    np.testing.assert_array_equal(
        negative.response_kernel,
        positive.response_kernel.conjugate(),
    )
    np.testing.assert_array_equal(
        negative.denominator,
        positive.denominator.conjugate(),
    )


def test_numpy_float32_unbiased_kernel_matches_screened_poisson_limit():
    spec = PRBiasedLinearizedReferenceSpec(0.9, 0.0, 0.2, 0.35, 1.7, 2.4)
    symbol = biased_linearized_fourier_symbol(
        (31, 33), spec=spec, backend=_backend_spec("numpy", "float32")
    )
    expected = np.zeros_like(symbol.response_kernel)
    expected[symbol.resolved_mask] = -1.0 / (
        np.float32(spec.reference_intensity)
        * (np.float32(1.0) + symbol.a_H[symbol.resolved_mask])
    )

    assert symbol.response_kernel.dtype == np.complex64
    np.testing.assert_allclose(
        symbol.response_kernel,
        expected,
        rtol=2e-6,
        atol=2e-7,
    )


def test_numpy_float32_y_independent_matches_qualified_reduced_limit():
    reference_intensity = 0.8
    background_intensity = reference_intensity
    applied_field = -0.6
    assert reference_intensity == background_intensity
    nx, ny = 33, 35
    dx = 2.0 * math.pi / nx
    dy = 2.0 * math.pi / ny
    x = np.arange(nx, dtype=np.float32)[:, None] * np.float32(dx)
    theta = np.float32(3.0) * x
    intensity = np.repeat(
        np.float32(reference_intensity)
        + np.float32(1.0e-4) * np.cos(theta).astype(np.float32),
        ny,
        axis=1,
    )
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity,
        applied_field,
        dx,
        dy,
        1.0,
        1.0,
    )
    result = solve_pr_biased_linearized_reference(
        intensity, spec=spec, backend=_backend_spec("numpy", "float32")
    )
    reduced_response = (3.0j - applied_field) / (
        reference_intensity * (1.0 + 9.0 + 3.0j * applied_field)
    )
    expected_E_x = np.float32(1.0e-4) * (
        np.float32(reduced_response.real) * np.cos(theta)
        - np.float32(reduced_response.imag) * np.sin(theta)
    )
    expected_E_x = np.repeat(expected_E_x, ny, axis=1).astype(np.float32)

    assert result.delta_E_x.dtype == np.float32
    np.testing.assert_allclose(
        result.delta_E_x,
        expected_E_x,
        rtol=3e-5,
        atol=2e-8,
    )
    np.testing.assert_allclose(result.delta_E_y, 0.0, rtol=0.0, atol=2e-6)


def test_numpy_float32_even_grid_resolves_only_non_null_modes():
    nx, ny = 8, 10
    ix = np.arange(nx, dtype=np.float32)[:, None]
    iy = np.arange(ny, dtype=np.float32)[None, :]
    resolved_mode = np.cos(
        np.float32(2.0 * math.pi) * (ix / nx + iy / ny)
    ).astype(np.float32)
    unresolved_modes = (
        (-1.0) ** ix + (-1.0) ** iy + (-1.0) ** (ix + iy)
    ).astype(np.float32)
    spec = PRBiasedLinearizedReferenceSpec(1.1, 0.7, 0.2, 0.3, 1.4, 2.1)
    backend = _backend_spec("numpy", "float32")
    symbol = biased_linearized_fourier_symbol((nx, ny), spec=spec, backend=backend)
    resolved_only = (
        np.float32(spec.reference_intensity)
        + np.float32(1.0e-3) * resolved_mode
    ).astype(np.float32)
    with_null_modes = (
        resolved_only + np.float32(2.0e-4) * unresolved_modes
    ).astype(np.float32)
    expected = solve_pr_biased_linearized_reference(
        resolved_only, spec=spec, backend=backend
    )
    actual = solve_pr_biased_linearized_reference(
        with_null_modes, spec=spec, backend=backend
    )

    assert np.count_nonzero(~symbol.resolved_mask) == 4
    assert np.all(symbol.response_kernel[~symbol.resolved_mask] == 0.0)
    assert np.all(symbol.denominator[~symbol.resolved_mask] == 0.0)
    assert np.all(np.isfinite(symbol.response_kernel[symbol.resolved_mask]))
    assert np.any(np.abs(actual.delta_psi) > 0.0)
    for name in ("delta_psi", "delta_E_x", "delta_E_y", "delta_P"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=3e-5,
            atol=5e-7,
        )


def test_explicit_cupy_request_uses_standard_unavailable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    spec = PRBiasedLinearizedReferenceSpec(1.0, 0.4, 0.2, 0.3)
    with pytest.raises(RuntimeError, match="requested cupy backend.*unavailable"):
        solve_pr_biased_linearized_reference(
            np.ones((5, 7), dtype=np.float64),
            spec=spec,
            backend=_backend_spec("cupy"),
        )


@pytest.mark.parametrize("shape", [(8, 10), (8, 9), (7, 10), (7, 9)])
def test_cupy_and_numpy_classify_identical_derivative_null_modes(shape):
    cp = _cupy_device()
    spec = PRBiasedLinearizedReferenceSpec(1.1, 0.7, 0.2, 0.3, 1.4, 2.1)
    numpy_symbol = biased_linearized_fourier_symbol(
        shape, spec=spec, backend=_backend_spec("numpy")
    )
    cupy_symbol = biased_linearized_fourier_symbol(
        shape, spec=spec, backend=_backend_spec("cupy")
    )

    np.testing.assert_array_equal(
        cp.asnumpy(cupy_symbol.resolved_mask), numpy_symbol.resolved_mask
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_symbol.kx), numpy_symbol.kx, rtol=3e-15, atol=3e-15
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_symbol.ky), numpy_symbol.ky, rtol=3e-15, atol=3e-15
    )


@pytest.mark.parametrize(
    ("mode_x", "mode_y", "applied_field"),
    [(2, 0, 0.0), (0, 3, 0.0), (2, 0, 0.8), (2, 0, -0.8), (2, 3, 0.8)],
)
def test_cupy_matches_numpy_analytic_modes(mode_x, mode_y, applied_field):
    cp = _cupy_device()
    numpy_result, expected_psi, expected_E_x, expected_E_y, a_H = _mode_case(
        mode_x=mode_x,
        mode_y=mode_y,
        applied_field=applied_field,
    )
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=numpy_result.reference_intensity,
        applied_field=applied_field,
        dx_normalized=numpy_result.dx_normalized,
        dy_normalized=numpy_result.dy_normalized,
        m_y=numpy_result.m_y,
        h_y=numpy_result.h_y,
    )
    intensity = spec.reference_intensity + 2.0e-4 * np.cos(
        mode_x * np.arange(33)[:, None] * spec.dx_normalized
        + mode_y * np.arange(35)[None, :] * spec.dy_normalized
    )
    cupy_result = solve_pr_biased_linearized_reference(
        cp.asarray(intensity, dtype=cp.float64),
        spec=spec,
        backend=_backend_spec("cupy"),
    )

    assert isinstance(cupy_result.delta_psi, cp.ndarray)
    assert isinstance(cupy_result.response_kernel, cp.ndarray)
    assert cupy_result.requested_backend == "cupy"
    assert cupy_result.resolved_backend == "cupy"
    assert cupy_result.real_dtype == "float64"
    assert cupy_result.complex_dtype == "complex128"
    assert cupy_result.is_gpu is True
    assert cupy_result.device_identity.startswith("cuda:")
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_psi),
        expected_psi,
        rtol=3e-13,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_E_x),
        expected_E_x,
        rtol=3e-13,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_E_y),
        expected_E_y,
        rtol=3e-13,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_P),
        a_H * expected_psi,
        rtol=3e-13,
        atol=8e-15,
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.response_kernel),
        numpy_result.response_kernel,
        rtol=3e-13,
        atol=3e-15,
    )


def test_cupy_matches_numpy_deterministic_multimode_and_batched_fixture():
    cp = _cupy_device()
    nx, ny = 32, 35
    dx = 2.0 * math.pi / nx
    dy = 2.0 * math.pi / ny
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    perturbation = (
        0.03
        + 0.02 * np.cos(2.0 * x)
        - 0.015 * np.sin(3.0 * y)
        + 0.01 * np.cos(2.0 * x - 3.0 * y)
    )
    intensity = np.stack((1.2 + perturbation, 1.2 - 0.6 * perturbation)).astype(
        np.float64
    )
    spec = PRBiasedLinearizedReferenceSpec(1.2, 0.85, dx, dy, 1.3, 2.2)
    numpy_result = solve_pr_biased_linearized_reference(
        intensity, spec=spec, backend=_backend_spec("numpy")
    )
    cupy_result = solve_pr_biased_linearized_reference(
        intensity, spec=spec, backend=_backend_spec("cupy")
    )

    for name in (
        "delta_psi",
        "delta_E_x",
        "delta_E_y",
        "delta_P",
        "delta_mean_current",
        "mean_intensity_perturbation",
        "response_kernel",
        "denominator",
    ):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(cupy_result, name)),
            getattr(numpy_result, name),
            rtol=3e-13,
            atol=3e-13,
        )


def test_cupy_float32_preserves_dtype_and_matches_numpy_float32():
    cp = _cupy_device()
    rng = np.random.default_rng(271828)
    intensity = (1.1 + 0.01 * rng.normal(size=(17, 19))).astype(np.float32)
    spec = PRBiasedLinearizedReferenceSpec(1.1, -0.6, 0.2, 0.3, 1.4, 2.1)
    numpy_result = solve_pr_biased_linearized_reference(
        intensity, spec=spec, backend=_backend_spec("numpy", "float32")
    )
    cupy_result = solve_pr_biased_linearized_reference(
        cp.asarray(intensity),
        spec=spec,
        backend=_backend_spec("cupy", "float32"),
    )

    assert cupy_result.delta_psi.dtype == cp.float32
    assert cupy_result.response_kernel.dtype == cp.complex64
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_psi),
        numpy_result.delta_psi,
        rtol=2e-5,
        atol=2e-6,
    )


def test_cupy_bias_reversal_unbiased_and_reduced_limits():
    cp = _cupy_device()
    common = dict(
        reference_intensity=1.2,
        dx_normalized=2.0 * math.pi / 33,
        dy_normalized=2.0 * math.pi / 35,
        m_y=1.0,
        h_y=1.0,
    )
    positive_spec = PRBiasedLinearizedReferenceSpec(applied_field=0.75, **common)
    negative_spec = PRBiasedLinearizedReferenceSpec(applied_field=-0.75, **common)
    positive = biased_linearized_fourier_symbol(
        (33, 35), spec=positive_spec, backend=_backend_spec("cupy")
    )
    negative = biased_linearized_fourier_symbol(
        (33, 35), spec=negative_spec, backend=_backend_spec("cupy")
    )
    cp.testing.assert_array_equal(
        negative.response_kernel, positive.response_kernel.conj()
    )

    unbiased_spec = PRBiasedLinearizedReferenceSpec(applied_field=0.0, **common)
    unbiased = biased_linearized_fourier_symbol(
        (33, 35), spec=unbiased_spec, backend=_backend_spec("cupy")
    )
    expected = cp.zeros_like(unbiased.response_kernel)
    expected[unbiased.resolved_mask] = -1.0 / (
        unbiased_spec.reference_intensity
        * (1.0 + unbiased.a_H[unbiased.resolved_mask])
    )
    cp.testing.assert_allclose(
        unbiased.response_kernel, expected, rtol=3e-13, atol=3e-15
    )

    x = np.arange(33)[:, None] * common["dx_normalized"]
    intensity = np.repeat(
        common["reference_intensity"] + 1.0e-4 * np.cos(3.0 * x),
        35,
        axis=1,
    ).astype(np.float64)
    cupy_result = solve_pr_biased_linearized_reference(
        intensity, spec=negative_spec, backend=_backend_spec("cupy")
    )
    numpy_result = solve_pr_biased_linearized_reference(
        intensity, spec=negative_spec, backend=_backend_spec("numpy")
    )
    np.testing.assert_allclose(
        cp.asnumpy(cupy_result.delta_E_x),
        numpy_result.delta_E_x,
        rtol=3e-13,
        atol=3e-13,
    )
    cp.testing.assert_allclose(cupy_result.delta_E_y, 0.0, atol=1e-18)


def test_cupy_nonlinear_taylor_oracle_retains_second_order_remainder():
    cp = _cupy_device()
    nx = ny = 33
    spacing = 2.0 * math.pi / nx
    x = cp.arange(nx, dtype=cp.float64) * spacing
    X, Y = cp.meshgrid(x, x, indexing="ij")
    shape = cp.cos(2.0 * X + 3.0 * Y)
    spec = PRBiasedLinearizedReferenceSpec(1.3, 0.7, spacing, spacing, 1.4, 2.1)
    errors = []
    for epsilon in (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        intensity = (spec.reference_intensity + epsilon * shape).astype(cp.float64)
        linearized = solve_pr_biased_linearized_reference(
            intensity, spec=spec, backend=_backend_spec("cupy")
        )
        residual = potential_rhs(
            linearized.delta_psi,
            intensity,
            dx_normalized=spec.dx_normalized,
            dy_normalized=spec.dy_normalized,
            m_y=spec.m_y,
            h_y=spec.h_y,
            applied_field_x=spec.applied_field,
            xp=cp,
        )
        errors.append(float(cp.sqrt(cp.mean(residual * residual)).item()))

    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    assert all(1.98 < order < 2.02 for order in orders)


def test_reference_solver_is_not_registered_on_pr_package():
    import lcprop.pr as pr

    assert not hasattr(pr, "solve_pr_biased_linearized_reference")
