import numpy as np
import pytest

from lcprop.pr.evolution import (
    hopping_rhs,
    semi_implicit_trapezoidal_step,
)
from lcprop.pr.static import (
    PRStaticSolverOptions,
    fixed_intensity_jacobian_rows,
    solve_pr_static_intensity,
    solve_pr_static_intensity_batched,
)


def _metrics(residual):
    return (
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(np.abs(residual))),
    )


def test_fixed_intensity_analytic_jacobian_matches_directional_difference():
    rng = np.random.default_rng(31)
    E = rng.normal(scale=0.08, size=(2, 18, 3))
    intensity = 0.4 + rng.random(E.shape)
    direction = rng.normal(size=E.shape)
    dx = 0.27
    epsilon = 2e-7
    lower, diagonal, upper = fixed_intensity_jacobian_rows(
        E,
        intensity,
        dx_normalized=dx,
    )
    analytic = (
        lower * np.roll(direction, 1, axis=-2)
        + diagonal * direction
        + upper * np.roll(direction, -1, axis=-2)
    )

    def residual(state):
        return hopping_rhs(
            state,
            intensity,
            applied_field=0.45,
            background_intensity=0.2,
            dx_normalized=dx,
            xp=np,
        )

    finite_difference = (
        residual(E + epsilon * direction)
        - residual(E - epsilon * direction)
    ) / (2.0 * epsilon)
    relative_error = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(
        finite_difference
    )

    assert relative_error < 2e-9


def test_uniform_static_root_is_analytic_and_has_no_spatial_structure():
    intensity_value = 1.3
    background = 0.2
    applied = 0.75
    intensity = np.full((3, 24, 4), intensity_value)
    x = np.arange(intensity.shape[-2])
    initial = (
        0.2
        + 0.05 * np.sin(2.0 * np.pi * x / x.size)[None, :, None]
    ) * np.ones((intensity.shape[0], 1, intensity.shape[-1]))

    result = solve_pr_static_intensity(
        intensity,
        applied_field=applied,
        background_intensity=background,
        dx_normalized=0.3,
        initial_E=initial,
    )
    expected = applied * background / intensity_value

    assert result.converged
    assert result.status == "converged"
    assert result.iterations >= 1
    assert result.residual_rms <= 1e-10
    assert result.residual_max <= 1e-9
    assert np.allclose(result.E, expected, rtol=0.0, atol=2e-13)
    assert np.max(np.ptp(result.E, axis=-2)) < 2e-13


def test_nonlinear_static_root_is_independent_of_reasonable_initial_guess():
    Nx, Ny = 36, 3
    x = np.arange(Nx)
    intensity = (
        1.05
        + 0.22 * np.cos(2.0 * np.pi * x / Nx)
        + 0.08 * np.sin(4.0 * np.pi * x / Nx)
    )[:, None] * np.ones((1, Ny))
    common = dict(
        applied_field=0.55,
        background_intensity=0.18,
        dx_normalized=0.24,
    )
    from_zero = solve_pr_static_intensity(intensity, **common)
    continuation_guess = from_zero.E + 0.015 * np.cos(
        6.0 * np.pi * x / Nx
    )[:, None]
    from_guess = solve_pr_static_intensity(
        intensity,
        initial_E=continuation_guess,
        **common,
    )

    assert from_zero.converged
    assert from_guess.converged
    assert np.allclose(from_guess.E, from_zero.E, rtol=0.0, atol=2e-11)
    assert all(
        later.residual_rms < earlier.residual_rms
        for earlier, later in zip(from_guess.records, from_guess.records[1:])
    )


def test_batched_static_root_matches_dense_solver_and_iteration_contract():
    Nx, Ny = 48, 4
    x = np.arange(Nx)
    intensity = (
        1.0
        + 0.24 * np.cos(2.0 * np.pi * x / Nx)
        + 0.07 * np.sin(6.0 * np.pi * x / Nx)
    )[:, None] * np.ones((1, Ny))
    initial = 0.03 * np.cos(4.0 * np.pi * x / Nx)[:, None]
    common = dict(
        applied_field=0.45,
        background_intensity=0.16,
        dx_normalized=0.21,
        initial_E=np.broadcast_to(initial, intensity.shape).copy(),
    )

    dense = solve_pr_static_intensity(intensity, **common)
    batched = solve_pr_static_intensity_batched(intensity, **common)

    assert dense.converged
    assert batched.converged
    assert batched.status == dense.status
    assert batched.iterations == dense.iterations
    assert tuple(record.step_scale for record in batched.records) == tuple(
        record.step_scale for record in dense.records
    )
    assert np.allclose(batched.E, dense.E, rtol=0.0, atol=3e-14)
    assert batched.residual_rms == pytest.approx(dense.residual_rms, abs=2e-15)
    assert batched.residual_max == pytest.approx(dense.residual_max, abs=2e-14)


def test_static_solver_does_not_claim_convergence_from_no_update():
    intensity = np.ones((16, 2))
    options = PRStaticSolverOptions(
        max_iterations=0,
        residual_rms_tolerance=1e-14,
        residual_max_tolerance=1e-14,
    )

    result = solve_pr_static_intensity(
        intensity,
        applied_field=0.5,
        background_intensity=0.2,
        dx_normalized=0.3,
        options=options,
    )

    assert not result.converged
    assert result.status == "max_iterations"
    assert result.iterations == 0
    assert result.residual_rms > options.residual_rms_tolerance
    assert result.residual_max > options.residual_max_tolerance


@pytest.mark.parametrize("profile", ("uniform", "modulated"))
def test_long_time_semi_implicit_transient_reaches_strict_static_root(profile):
    Nx, Ny = 32, 2
    x = np.arange(Nx)
    if profile == "uniform":
        intensity = np.full((Nx, Ny), 1.2)
    else:
        intensity = (
            1.1
            + 0.2 * np.cos(2.0 * np.pi * x / Nx)
            + 0.05 * np.sin(4.0 * np.pi * x / Nx)
        )[:, None] * np.ones((1, Ny))
    common = dict(
        applied_field=0.3,
        background_intensity=0.2,
        dx_normalized=0.25,
    )
    static = solve_pr_static_intensity(intensity, **common)
    state = np.zeros_like(intensity)
    dt = 0.1
    for _ in range(200):
        state = semi_implicit_trapezoidal_step(
            state,
            lambda _state: intensity,
            dt_normalized=dt,
            xp=np,
            **common,
        )
    residual = hopping_rhs(state, intensity, xp=np, **common)
    residual_rms, residual_max = _metrics(residual)
    normalized_l2 = np.linalg.norm(state - static.E) / np.linalg.norm(static.E)

    assert static.converged
    assert normalized_l2 < 5e-10
    assert np.max(np.abs(state - static.E)) < 5e-11
    assert residual_rms < 5e-11
    assert residual_max < 6e-11


def test_static_reference_solver_validates_shapes_and_uses_float64():
    intensity = np.ones((12, 3), dtype=np.float32)
    result = solve_pr_static_intensity(
        intensity,
        applied_field=0.2,
        background_intensity=0.1,
        dx_normalized=0.2,
    )
    assert result.E.dtype == np.float64

    with pytest.raises(ValueError, match="same shape"):
        solve_pr_static_intensity(
            intensity,
            applied_field=0.2,
            background_intensity=0.1,
            dx_normalized=0.2,
            initial_E=np.zeros((11, 3)),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        solve_pr_static_intensity(
            -intensity,
            applied_field=0.2,
            background_intensity=0.1,
            dx_normalized=0.2,
        )


def test_batched_static_material_solve_stays_on_cupy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    Nx, Ny = 32, 3
    x = cp.arange(Nx, dtype=cp.float64)
    intensity = (
        1.1 + 0.2 * cp.cos(2.0 * cp.pi * x / Nx)
    )[:, None] * cp.ones((1, Ny), dtype=cp.float64)
    gpu = solve_pr_static_intensity_batched(
        intensity,
        applied_field=0.3,
        background_intensity=0.2,
        dx_normalized=0.25,
        xp=cp,
    )
    cpu = solve_pr_static_intensity(
        cp.asnumpy(intensity),
        applied_field=0.3,
        background_intensity=0.2,
        dx_normalized=0.25,
    )

    assert gpu.converged
    assert isinstance(gpu.E, cp.ndarray)
    assert np.allclose(cp.asnumpy(gpu.E), cpu.E, rtol=2e-12, atol=2e-13)
