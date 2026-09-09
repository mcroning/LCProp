from __future__ import annotations

import math

import numpy as np
import pytest

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
