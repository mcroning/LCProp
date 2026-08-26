from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import lcprop.pr.transverse.static as static_module
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseStaticMaterialSolverOptions,
    derivative_null_residual,
    project_production_resolved_modes,
    production_steady_jvp,
    production_steady_residual,
    solve_pr_transverse_discrete_static_intensity,
    solve_pr_transverse_static_intensity,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.static import (
    _jacobian_action,
    _pcg,
    _resolved_pcg_forcing,
    _right_preconditioned_gmres,
    _symbols,
)
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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_gmres_forcing_prevents_unconverged_zero_iteration_success(dtype):
    rhs = np.full((5, 7), dtype(2.0e-7), dtype=dtype)
    initial_norm = float(np.linalg.norm(rhs.ravel()))

    solution, iterations, restarts, relative_residual, converged, status = (
        _right_preconditioned_gmres(
            rhs,
            apply_operator=lambda value: value.copy(),
            apply_preconditioner=lambda value: value,
            relative_tolerance=2.0e-4,
            absolute_tolerance=2.0 * initial_norm,
            restart=4,
            max_iterations=8,
            xp=np,
        )
    )

    assert converged
    assert status == "converged"
    assert iterations == 1
    assert restarts == 1
    assert relative_residual <= 2.0e-6
    np.testing.assert_allclose(solution, rhs, rtol=2.0e-6, atol=0.0)


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
        equilibrium_rms_tolerance=1.0e-11,
        equilibrium_max_tolerance=1.0e-10,
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
    summary = result.plane_summaries[0]
    assert summary.physical_state_valid
    assert summary.carrier_minimum > 0.0


def test_td_rhs_is_diagnostic_only_for_authoritative_zero_flux_solution():
    spacing, psi, intensity = _nyquist_rich_continuum_equilibrium(16)
    options = PRTransverseStaticMaterialSolverOptions(
        td_rhs_rms_tolerance=1.0e-30,
        td_rhs_max_tolerance=1.0e-30,
    )
    result = solve_pr_transverse_static_intensity(
        intensity,
        initial_psi=psi,
        dx_normalized=spacing,
        dy_normalized=spacing,
        options=options,
    )
    assert result.converged
    summary = result.plane_summaries[0]
    assert summary.equilibrium_rms <= options.equilibrium_rms_tolerance
    assert summary.equilibrium_max <= options.equilibrium_max_tolerance
    assert summary.td_rhs_rms > options.td_rhs_rms_tolerance
    assert summary.td_rhs_max > options.td_rhs_max_tolerance


def test_nonlinear_aware_pcg_forcing_tightens_only_near_zero_flux_gate():
    options = PRTransverseStaticMaterialSolverOptions(
        pcg_relative_tolerance=2.0e-5,
        pcg_absolute_tolerance=1.0e-7,
        equilibrium_rms_tolerance=4.0e-6,
        equilibrium_max_tolerance=2.0e-5,
    )
    assert _resolved_pcg_forcing(4.1e-6, 4.0e-5, options) == (
        2.0e-5,
        1.0e-7,
        False,
    )
    assert _resolved_pcg_forcing(3.9e-6, 4.0e-5, options) == (
        1.0e-7,
        1.0e-9,
        True,
    )
    assert _resolved_pcg_forcing(3.9e-6, 1.9e-5, options) == (
        2.0e-5,
        1.0e-7,
        False,
    )
    float64_defaults = PRTransverseStaticMaterialSolverOptions()
    assert _resolved_pcg_forcing(1.0e-10, 2.0e-8, float64_defaults) == (
        float64_defaults.pcg_relative_tolerance,
        float64_defaults.pcg_absolute_tolerance,
        True,
    )


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
        # The constructed state is algebraically exact; this measures only
        # platform-dependent FFT/exp roundoff, not a solver tolerance.
        assert np.max(np.abs(residual)) < 5e-15
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


def test_volume_solve_constructs_invariant_spectral_operators_once(monkeypatch):
    calls = 0
    original = static_module._symbols

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(static_module, "_symbols", counted)
    result = solve_pr_transverse_static_intensity(
        np.ones((3, 9, 7), dtype=np.float64),
        dx_normalized=0.4,
        dy_normalized=0.5,
    )
    assert result.converged
    assert calls == 1


@pytest.mark.parametrize("shape", ((17, 15), (18, 16)))
def test_pcg_reports_direct_true_linear_residual_norm(shape):
    rng = np.random.default_rng(16005 + shape[0])
    denominator, null_mask = _symbols(
        shape, dx_normalized=0.37, dy_normalized=0.43, h_y=1.0
    )
    weight = np.exp(rng.normal(scale=0.12, size=shape))
    weight /= np.mean(weight)
    rhs = project_production_resolved_modes(
        rng.normal(size=shape), dx_normalized=0.37, dy_normalized=0.43
    )
    options = PRTransverseStaticMaterialSolverOptions(
        pcg_relative_tolerance=1.0e-10,
        pcg_absolute_tolerance=1.0e-12,
    )
    (
        solution,
        iterations,
        converged,
        status,
        tolerance,
        reported_residual_norm,
    ) = _pcg(
        rhs,
        weight=weight,
        denominator=denominator,
        null_mask=null_mask,
        options=options,
        relative_tolerance=options.pcg_relative_tolerance,
        absolute_tolerance=options.pcg_absolute_tolerance,
    )
    direct_residual = rhs - _jacobian_action(
        solution,
        weight=weight,
        denominator=denominator,
        null_mask=null_mask,
    )
    direct_norm = float(np.linalg.norm(direct_residual.ravel()))
    assert converged
    assert status == "converged"
    assert iterations > 0
    assert reported_residual_norm <= tolerance
    assert reported_residual_norm == pytest.approx(
        direct_norm, rel=2.0e-15, abs=2.0e-15
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


@pytest.mark.parametrize("shape", ((15, 13), (16, 14)))
@pytest.mark.parametrize("nyquist_rich", (False, True))
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_production_discrete_jvp_matches_centered_directional_difference(
    shape, nyquist_rich, dtype
):
    rng = np.random.default_rng(82026 + shape[0])
    dx, dy = 0.37, 0.43
    x = 2.0 * math.pi * np.arange(shape[0])[:, None] / shape[0]
    y = 2.0 * math.pi * np.arange(shape[1])[None, :] / shape[1]
    if nyquist_rich:
        psi = (
            2.0e-4 * np.cos((shape[0] // 2 - 1) * x)
            + 1.5e-4 * np.sin((shape[1] // 2 - 1) * y)
        )
        intensity = 0.3 + np.exp(
            0.4 * np.cos((shape[0] // 2 - 2) * x)
            + 0.2 * np.sin((shape[1] // 2 - 2) * y)
        )
    else:
        psi = rng.normal(scale=2.0e-3, size=shape)
        intensity = 0.2 + rng.random(shape)
    psi = project_production_resolved_modes(
        psi.astype(dtype), dx_normalized=dx, dy_normalized=dy
    )
    intensity = intensity.astype(dtype)
    vector = project_production_resolved_modes(
        rng.normal(size=shape).astype(dtype),
        dx_normalized=dx,
        dy_normalized=dy,
    )
    vector /= np.asarray(
        np.sqrt(np.mean(vector.astype(np.float64) ** 2)), dtype=dtype
    )
    analytic = production_steady_jvp(
        psi,
        intensity,
        vector,
        dx_normalized=dx,
        dy_normalized=dy,
    )
    epsilon = np.asarray(1.0e-3 if dtype is np.float64 else 2.0e-2, dtype=dtype)
    finite_difference = (
        production_steady_residual(
            psi + epsilon * vector,
            intensity,
            dx_normalized=dx,
            dy_normalized=dy,
        )
        - production_steady_residual(
            psi - epsilon * vector,
            intensity,
            dx_normalized=dx,
            dy_normalized=dy,
        )
    ) / (2.0 * epsilon)
    relative = np.linalg.norm(finite_difference - analytic) / np.linalg.norm(
        analytic
    )
    tolerance = 2.0e-11 if dtype is np.float64 else 2.0e-6
    assert relative < tolerance


def _nyquist_rich_continuum_equilibrium(n: int):
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    k = n // 2 - 1
    raw = (
        np.cos(k * x)
        + 0.8 * np.sin((k - 1) * y)
        + 0.5 * np.cos((k - 2) * x - (k - 1) * y)
    )
    unscaled = state_from_potential(
        raw, dx_normalized=spacing, dy_normalized=spacing
    )
    amplitude = 0.35 / np.max(np.abs(unscaled.carrier_density - 1.0))
    psi = amplitude * raw
    state = state_from_potential(
        psi, dx_normalized=spacing, dy_normalized=spacing
    )
    intensity = np.exp(-state.psi) / state.carrier_density
    return spacing, psi, intensity


@pytest.mark.parametrize("n", (15, 16))
def test_discrete_corrector_removes_continuum_product_rule_residual(n):
    spacing, continuum_psi, intensity = _nyquist_rich_continuum_equilibrium(n)
    continuum_residual = static_equilibrium_residual(
        continuum_psi,
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    assert np.max(np.abs(continuum_residual)) < 2.0e-14
    initial_production = production_steady_residual(
        continuum_psi,
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    assert np.sqrt(np.mean(initial_production * initial_production)) > 1.0e-6

    result = solve_pr_transverse_discrete_static_intensity(
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    assert result.converged
    assert result.continuum_result.converged
    summary = result.plane_summaries[0]
    assert summary.initial_residual_rms > 1.0e-6
    assert summary.final_residual_rms < 1.0e-8
    assert summary.final_residual_max < 1.0e-7
    assert summary.final_residual_rms < summary.initial_residual_rms * 1.0e-4
    assert summary.carrier_minimum > 0.0
    assert abs(summary.carrier_mean - 1.0) < 2.0e-15
    assert result.iteration_records
    assert all(record.gmres_converged for record in result.iteration_records)


def test_discrete_corrector_is_translation_equivariant_and_deterministic():
    spacing, _, intensity = _nyquist_rich_continuum_equilibrium(15)
    options = PRTransverseDiscreteStaticCorrectorOptions(gmres_restart=12)
    direct = solve_pr_transverse_discrete_static_intensity(
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
        corrector_options=options,
    )
    repeat = solve_pr_transverse_discrete_static_intensity(
        intensity,
        dx_normalized=spacing,
        dy_normalized=spacing,
        corrector_options=options,
    )
    shift = (4, -3)
    shifted = solve_pr_transverse_discrete_static_intensity(
        np.roll(intensity, shift, axis=(0, 1)),
        dx_normalized=spacing,
        dy_normalized=spacing,
        corrector_options=options,
    )
    np.testing.assert_array_equal(repeat.psi, direct.psi)
    np.testing.assert_allclose(
        shifted.psi,
        np.roll(direct.psi, shift, axis=(0, 1)),
        rtol=0,
        atol=3.0e-12,
    )


def test_discrete_corrector_preserves_isotropic_xy_exchange_symmetry():
    n = 15
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    intensity = 0.4 + np.exp(
        0.2 * np.cos(5.0 * x) + 0.13 * np.sin(4.0 * y)
    )
    direct = solve_pr_transverse_discrete_static_intensity(
        intensity, dx_normalized=spacing, dy_normalized=spacing
    )
    transposed = solve_pr_transverse_discrete_static_intensity(
        intensity.T, dx_normalized=spacing, dy_normalized=spacing
    )
    assert direct.converged and transposed.converged
    np.testing.assert_allclose(
        transposed.psi.T, direct.psi, rtol=0, atol=3.0e-12
    )


def test_production_residual_and_jvp_ignore_joint_null_potential_modes():
    rng = np.random.default_rng(82027)
    shape = (16, 14)
    dx, dy = 0.4, 0.5
    psi = rng.normal(scale=1.0e-3, size=shape)
    intensity = 0.2 + rng.random(shape)
    vector = rng.normal(size=shape)
    kx = 2.0 * math.pi * np.fft.fftfreq(shape[0], d=dx)[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(shape[1], d=dy)[None, :]
    kx[shape[0] // 2, 0] = 0.0
    ky[0, shape[1] // 2] = 0.0
    null_mask = kx * kx + ky * ky == 0.0
    null_hat = np.zeros(shape, dtype=np.complex128)
    null_hat[null_mask] = np.arange(1, np.count_nonzero(null_mask) + 1)
    null_mode = np.fft.ifft2(null_hat).real
    np.testing.assert_allclose(
        production_steady_residual(
            psi + null_mode,
            intensity,
            dx_normalized=dx,
            dy_normalized=dy,
        ),
        production_steady_residual(
            psi, intensity, dx_normalized=dx, dy_normalized=dy
        ),
        rtol=0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        production_steady_jvp(
            psi,
            intensity,
            vector + null_mode,
            dx_normalized=dx,
            dy_normalized=dy,
        ),
        production_steady_jvp(
            psi, intensity, vector, dx_normalized=dx, dy_normalized=dy
        ),
        rtol=0,
        atol=2.0e-13,
    )
