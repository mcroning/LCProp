from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from lcprop.pr.transverse.static import (
    PRTransverseStaticMaterialSolverOptions,
    derivative_null_residual,
    project_production_resolved_modes,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.static import _jacobian_action, _symbols
from lcprop.pr.transverse.transport import potential_rhs, state_from_potential


def _constructed_equilibrium(n: int, dtype=np.float64):
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    psi = (
        0.025 * np.cos(2.0 * x)
        + 0.018 * np.sin(3.0 * y)
        + 0.01 * np.cos(x - y)
    ).astype(dtype)
    psi -= np.mean(psi)
    state = state_from_potential(
        psi, dx_normalized=spacing, dy_normalized=spacing
    )
    intensity = (np.exp(-state.psi) / state.carrier_density).astype(dtype)
    return spacing, psi, intensity


def test_uniform_intensity_has_exact_uniform_static_equilibrium():
    intensity = np.full((2, 15, 13), 0.31, dtype=np.float64)
    result = solve_pr_transverse_static_intensity(
        intensity, dx_normalized=0.4, dy_normalized=0.5
    )
    assert result.converged
    assert result.status == "converged"
    assert np.array_equal(result.psi, np.zeros_like(intensity))
    assert np.array_equal(result.equilibrium_residual, np.zeros_like(intensity))
    assert np.array_equal(result.td_rhs_residual, np.zeros_like(intensity))
    assert all(summary.carrier_mean == 1.0 for summary in result.plane_summaries)


def test_constructed_nonuniform_zero_flux_state_is_recovered_and_physical():
    spacing, expected, intensity = _constructed_equilibrium(24)
    options = PRTransverseStaticMaterialSolverOptions(
        td_rhs_rms_tolerance=1.0e-10,
        td_rhs_max_tolerance=1.0e-9,
    )
    result = solve_pr_transverse_static_intensity(
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
        options=options,
    )
    assert result.converged
    np.testing.assert_allclose(result.psi, expected, rtol=0, atol=3e-10)
    state = state_from_potential(
        result.psi, dx_normalized=spacing, dy_normalized=spacing
    )
    assert np.min(state.carrier_density) > 0.0
    assert abs(np.mean(state.psi)) < 1e-16
    assert abs(np.mean(state.carrier_density) - 1.0) < 1e-15
    assert np.all(np.isfinite(state.E_x))
    assert np.all(np.isfinite(state.E_y))


def test_td_rhs_at_constructed_equilibrium_converges_under_resolution_refinement():
    maxima = []
    for n in (12, 24, 32):
        spacing, psi, intensity = _constructed_equilibrium(n)
        residual = static_equilibrium_residual(
            psi,
            intensity,
            dx_normalized=spacing,
            dy_normalized=spacing,
        )
        assert np.max(np.abs(residual)) < 3e-15
        td_rhs = potential_rhs(
            psi,
            intensity,
            dx_normalized=spacing,
            dy_normalized=spacing,
        )
        maxima.append(float(np.max(np.abs(td_rhs))))
    assert maxima[1] < maxima[0] * 1e-4
    assert maxima[2] < maxima[1] * 1e-2
    assert maxima[2] < 1e-13


def test_even_grid_derivative_null_residual_is_accounted_separately():
    spacing, psi, intensity = _constructed_equilibrium(16)
    residual = static_equilibrium_residual(
        psi,
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    null = derivative_null_residual(
        residual,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    resolved = project_production_resolved_modes(
        residual,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    np.testing.assert_allclose(residual, null + resolved, rtol=0, atol=2e-16)
    projected_null = project_production_resolved_modes(
        null,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    assert np.max(np.abs(projected_null)) < 1e-30


def test_static_solver_is_translation_equivariant_and_deterministic():
    spacing, _, intensity = _constructed_equilibrium(24)
    direct = solve_pr_transverse_static_intensity(
        intensity, dx_normalized=spacing, dy_normalized=spacing
    )
    repeat = solve_pr_transverse_static_intensity(
        intensity, dx_normalized=spacing, dy_normalized=spacing
    )
    shift = (5, -4)
    shifted = solve_pr_transverse_static_intensity(
        np.roll(intensity, shift, axis=(0, 1)),
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    np.testing.assert_array_equal(repeat.psi, direct.psi)
    np.testing.assert_allclose(
        shifted.psi, np.roll(direct.psi, shift, axis=(0, 1)), rtol=0, atol=2e-14
    )


def test_static_solver_preserves_xy_exchange_symmetry():
    n = 21
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    intensity = 0.4 + np.exp(0.15 * np.cos(x) + 0.08 * np.cos(2.0 * y))
    direct = solve_pr_transverse_static_intensity(
        intensity, dx_normalized=spacing, dy_normalized=spacing
    )
    transposed = solve_pr_transverse_static_intensity(
        intensity.T, dx_normalized=spacing, dy_normalized=spacing
    )
    assert direct.converged and transposed.converged
    np.testing.assert_allclose(transposed.psi.T, direct.psi, rtol=0, atol=2e-13)


def test_analytic_projected_jacobian_is_symmetric_for_pcg():
    rng = np.random.default_rng(85)
    shape = (17, 15)
    denominator, null_mask = _symbols(
        shape, dx_normalized=0.4, dy_normalized=0.5, h_y=1.0
    )
    weight = np.exp(rng.normal(scale=0.1, size=shape))
    weight /= np.mean(weight)
    left = project_production_resolved_modes(
        rng.normal(size=shape), dx_normalized=0.4, dy_normalized=0.5
    )
    right = project_production_resolved_modes(
        rng.normal(size=shape), dx_normalized=0.4, dy_normalized=0.5
    )
    j_left = _jacobian_action(
        left, weight=weight, denominator=denominator, null_mask=null_mask
    )
    j_right = _jacobian_action(
        right, weight=weight, denominator=denominator, null_mask=null_mask
    )
    np.testing.assert_allclose(
        np.vdot(left, j_right).real,
        np.vdot(j_left, right).real,
        rtol=2e-14,
        atol=2e-13,
    )


def test_static_equilibrium_rejects_nonpositive_or_nonfinite_input():
    intensity = np.ones((9, 7), dtype=np.float64)
    for value in (0.0, -1.0):
        bad = intensity.copy()
        bad[2, 3] = value
        with pytest.raises(ValueError, match="strictly positive"):
            solve_pr_transverse_static_intensity(
                bad, dx_normalized=0.5, dy_normalized=0.6
            )
    bad = intensity.copy()
    bad[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_pr_transverse_static_intensity(
            bad, dx_normalized=0.5, dy_normalized=0.6
        )
    with pytest.raises(ValueError, match="requires h_y=1"):
        solve_pr_transverse_static_intensity(
            intensity, dx_normalized=0.5, dy_normalized=0.6, h_y=2.0
        )
    with pytest.raises(TypeError, match="requires float64"):
        solve_pr_transverse_static_intensity(
            intensity.astype(np.float32),
            dx_normalized=0.5,
            dy_normalized=0.6,
        )
    x = 2.0 * math.pi * np.arange(9)[:, None] / 9
    nonphysical = np.repeat((2.0 * np.cos(x)), 7, axis=1)
    with pytest.raises(ValueError, match="nonpositive carrier"):
        solve_pr_transverse_static_intensity(
            intensity,
            dx_normalized=0.5,
            dy_normalized=0.6,
            initial_psi=nonphysical,
        )


def test_static_equilibrium_has_no_transport_tensor_parameter():
    parameters = inspect.signature(solve_pr_transverse_static_intensity).parameters
    assert "m_y" not in parameters


def test_y_independent_static_state_is_stationary_for_spectral_a7():
    nx, ny = 48, 5
    dx = 2.0 * math.pi / nx
    x = 2.0 * math.pi * np.arange(nx)[:, None] / nx
    psi = np.repeat((0.02 * np.cos(x) + 0.01 * np.cos(2.0 * x)), ny, axis=1)
    state = state_from_potential(psi, dx_normalized=dx, dy_normalized=1.0)
    intensity = np.exp(-state.psi) / state.carrier_density
    result = solve_pr_transverse_static_intensity(
        intensity, dx_normalized=dx, dy_normalized=1.0
    )
    assert result.converged
    solved_state = state_from_potential(
        result.psi, dx_normalized=dx, dy_normalized=1.0
    )
    psi_tau = potential_rhs(
        result.psi, intensity, dx_normalized=dx, dy_normalized=1.0
    )
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    E_tau = -np.fft.ifft(
        1j * kx * np.fft.fft(psi_tau, axis=0), axis=0
    ).real
    assert np.max(np.abs(E_tau)) < 1e-8
    assert np.max(np.abs(solved_state.E_y)) < 1e-13
