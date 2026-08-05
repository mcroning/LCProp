import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.evolution import (
    diffusion_implicit_split,
    euler_step,
    hopping_rhs,
    periodic_derivatives_x,
    semi_implicit_trapezoidal_step,
    solve_periodic_variable_diffusion,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.workflow import _optical_pass


def _periodic_residual(solution, rhs, intensity, *, alpha, dx):
    _, second = periodic_derivatives_x(
        solution,
        dx_normalized=dx,
        xp=np,
    )
    return solution - alpha * intensity * second - rhs


def _coupled_optical_case():
    grid_spec = GridSpec(
        Nx=16,
        Ny=6,
        x_aperture_um=32.0,
        y_aperture_um=18.0,
        dz_um=4.0,
        z_length_um=12.0,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="coupled-order beam",
                wavelength_um=0.633,
                power_mW=1.0,
                waist_x_um=5.0,
                waist_y_um=4.0,
                x0_um=-1.5,
                tilt_x_rad_per_um=0.12,
                coherence_group="order-test",
            ),
        )
    )
    material = PRMaterialSpec(
        dark_intensity=0.15,
        uniform_background_intensity=0.05,
        applied_field=0.4,
        gain_length_product=1.8,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.5,
    )
    request = PRRunRequest(
        grid=grid_spec,
        beams=beams,
        material=material,
        solver=PRSolverOptions(Nt=1, dt_normalized=1e-3),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    launch = build_launch(beams, grid, complex_dtype=np.complex128)
    A0 = launch.A0.copy()
    peak_reference = channel_peak_intensity_reference(A0, xp=np)
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=beams.channels[0].wavelength_um,
        n_ref=material.refractive_index,
        xp=np,
    )
    z_scale = np.linspace(0.8, 1.2, grid.Nz)[:, None, None]
    x_phase = 2.0 * np.pi * np.arange(grid.Nx) / grid.Nx
    initial_E = (
        0.06
        * z_scale
        * np.sin(x_phase)[None, :, None]
        * np.ones((1, 1, grid.Ny))
    )

    def optical_map(E):
        return _optical_pass(
            A0,
            E,
            request=request,
            grid=grid,
            kernel=kernel,
            peak_reference=peak_reference,
            wavelength_um=beams.channels[0].wavelength_um,
        )

    return request, grid, A0, initial_E, optical_map


def test_diffusion_split_reconstructs_authoritative_residual():
    rng = np.random.default_rng(13)
    E = rng.normal(scale=0.2, size=(3, 12, 4))
    intensity = 0.3 + rng.random(E.shape)

    implicit, explicit = diffusion_implicit_split(
        E,
        intensity,
        applied_field=0.4,
        background_intensity=0.25,
        dx_normalized=0.3,
        xp=np,
    )
    residual = hopping_rhs(
        E,
        intensity,
        applied_field=0.4,
        background_intensity=0.25,
        dx_normalized=0.3,
        xp=np,
    )

    assert np.allclose(implicit + explicit, residual, rtol=0.0, atol=2e-15)


@pytest.mark.parametrize("shape", ((17, 5), (3, 17, 5)))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_periodic_variable_diffusion_solve_residual_and_dtype(shape, dtype):
    rng = np.random.default_rng(7)
    rhs = rng.normal(size=shape).astype(dtype)
    intensity = (0.05 + rng.random(size=shape)).astype(dtype)
    alpha = 0.19
    dx = 0.27

    solution = solve_periodic_variable_diffusion(
        rhs,
        intensity,
        alpha=alpha,
        dx_normalized=dx,
        xp=np,
    )
    residual = _periodic_residual(
        solution,
        rhs,
        intensity,
        alpha=alpha,
        dx=dx,
    )

    tolerance = 3e-6 if dtype is np.float32 else 2e-14
    assert solution.dtype == dtype
    assert np.max(np.abs(residual)) < tolerance


def test_periodic_variable_diffusion_matches_dense_reference():
    rng = np.random.default_rng(11)
    Nx, Ny = 9, 3
    rhs = rng.normal(size=(Nx, Ny))
    intensity = 0.2 + rng.random(size=(Nx, Ny))
    alpha = 0.08
    dx = 0.4

    actual = solve_periodic_variable_diffusion(
        rhs,
        intensity,
        alpha=alpha,
        dx_normalized=dx,
        xp=np,
    )
    expected = np.empty_like(actual)
    for y_index in range(Ny):
        q = alpha * intensity[:, y_index] / dx**2
        matrix = np.zeros((Nx, Nx))
        for x_index in range(Nx):
            matrix[x_index, x_index] = 1.0 + 2.0 * q[x_index]
            matrix[x_index, (x_index - 1) % Nx] = -q[x_index]
            matrix[x_index, (x_index + 1) % Nx] = -q[x_index]
        expected[:, y_index] = np.linalg.solve(matrix, rhs[:, y_index])

    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_periodic_variable_diffusion_validates_inputs():
    rhs = np.ones((8, 3), dtype=np.float64)
    intensity = np.ones_like(rhs)

    with pytest.raises(ValueError, match="identical shapes"):
        solve_periodic_variable_diffusion(
            rhs,
            intensity[:-1],
            alpha=0.1,
            dx_normalized=0.2,
            xp=np,
        )
    with pytest.raises(ValueError, match="Nx >= 3"):
        solve_periodic_variable_diffusion(
            rhs[:2],
            intensity[:2],
            alpha=0.1,
            dx_normalized=0.2,
            xp=np,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        solve_periodic_variable_diffusion(
            rhs,
            -intensity,
            alpha=0.1,
            dx_normalized=0.2,
            xp=np,
        )
    with pytest.raises(ValueError, match="alpha"):
        solve_periodic_variable_diffusion(
            rhs,
            intensity,
            alpha=-0.1,
            dx_normalized=0.2,
            xp=np,
        )
    with pytest.raises(TypeError, match="floating-point"):
        solve_periodic_variable_diffusion(
            rhs.astype(np.int64),
            intensity,
            alpha=0.1,
            dx_normalized=0.2,
            xp=np,
        )


def test_constant_diffusion_solve_gives_crank_nicolson_mode_factor():
    Nx, Ny = 64, 2
    dx = 0.15
    mode = 7
    dt = 0.2
    intensity_value = 1.3
    x = np.arange(Nx) * dx
    wavenumber = 2.0 * np.pi * mode / (Nx * dx)
    initial = np.sin(wavenumber * x)[:, None] * np.ones((1, Ny))
    intensity = np.full_like(initial, intensity_value)
    _, second = periodic_derivatives_x(
        initial,
        dx_normalized=dx,
        xp=np,
    )

    actual = solve_periodic_variable_diffusion(
        initial + 0.5 * dt * intensity * second,
        intensity,
        alpha=0.5 * dt,
        dx_normalized=dx,
        xp=np,
    )
    k2_squared = 4.0 * np.sin(0.5 * wavenumber * dx) ** 2 / dx**2
    z = -dt * intensity_value * k2_squared
    expected = ((1.0 + 0.5 * z) / (1.0 - 0.5 * z)) * initial

    assert np.allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_semi_implicit_uniform_state_reduces_to_heun_reaction_step():
    E = np.full((12, 4), 0.3)
    intensity_value = 1.2
    intensity = np.full_like(E, intensity_value)
    background = 0.2
    applied = 0.7
    dt = 0.04
    calls = []

    def fixed_intensity(state):
        calls.append(state.copy())
        return intensity

    actual = semi_implicit_trapezoidal_step(
        E,
        fixed_intensity,
        dt_normalized=dt,
        applied_field=applied,
        background_intensity=background,
        dx_normalized=0.25,
        xp=np,
    )
    source = applied * background
    slope_n = source - intensity_value * E
    predictor = E + dt * slope_n
    expected = E + 0.5 * dt * (
        slope_n + source - intensity_value * predictor
    )

    assert len(calls) == 2
    assert np.allclose(calls[0], E)
    assert np.allclose(calls[1], predictor)
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-15)


def _temporal_convergence_errors(*, state_dependent_intensity):
    Nx, Ny = 18, 2
    dx = 0.3
    x = np.arange(Nx) * dx
    initial = (
        0.08 * np.sin(2.0 * np.pi * 2.0 * x / (Nx * dx))[:, None]
        * np.ones((1, Ny))
    )
    fixed_profile = (
        1.1
        + 0.15 * np.cos(2.0 * np.pi * x / (Nx * dx))[:, None]
        * np.ones((1, Ny))
    )
    background = 0.2
    applied = 0.35
    final_time = 0.12

    def intensity_for(state):
        if state_dependent_intensity:
            return fixed_profile + 0.08 * np.tanh(state)
        return fixed_profile

    def ode_rhs(_time, flat_state):
        state = flat_state.reshape(initial.shape)
        return hopping_rhs(
            state,
            intensity_for(state),
            applied_field=applied,
            background_intensity=background,
            dx_normalized=dx,
            xp=np,
        ).ravel()

    reference_solution = solve_ivp(
        ode_rhs,
        (0.0, final_time),
        initial.ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert reference_solution.success
    reference = reference_solution.y[:, -1].reshape(initial.shape)

    errors = []
    for step_count in (4, 8, 16):
        dt = final_time / step_count
        state = initial.copy()
        for _ in range(step_count):
            state = semi_implicit_trapezoidal_step(
                state,
                intensity_for,
                dt_normalized=dt,
                applied_field=applied,
                background_intensity=background,
                dx_normalized=dx,
                xp=np,
            )
        errors.append(np.linalg.norm(state - reference))
    return np.asarray(errors)


@pytest.mark.parametrize("state_dependent_intensity", (False, True))
def test_semi_implicit_prototype_has_second_order_temporal_convergence(
    state_dependent_intensity,
):
    errors = _temporal_convergence_errors(
        state_dependent_intensity=state_dependent_intensity,
    )

    orders = np.log2(errors[:-1] / errors[1:])
    assert errors[2] < errors[1] < errors[0]
    assert np.all(orders > 1.85)
    assert np.all(orders < 2.2)


def _coupled_temporal_convergence_errors(*, method="semi_implicit"):
    request, grid, _A0, initial_E, optical_map = _coupled_optical_case()
    material = request.material
    dx_normalized = material.characteristic_wavenumber_per_um * grid.dx_um
    final_time = 0.08

    def intensity_for(E):
        _A, intensity = optical_map(E)
        return intensity

    def ode_rhs(_time, flat_state):
        state = flat_state.reshape(initial_E.shape)
        return hopping_rhs(
            state,
            intensity_for(state),
            applied_field=material.applied_field,
            background_intensity=material.background_intensity,
            dx_normalized=dx_normalized,
            xp=np,
        ).ravel()

    reference_solution = solve_ivp(
        ode_rhs,
        (0.0, final_time),
        initial_E.ravel(),
        method="DOP853",
        rtol=5e-11,
        atol=5e-13,
    )
    assert reference_solution.success
    reference = reference_solution.y[:, -1].reshape(initial_E.shape)

    errors = []
    for step_count in (2, 4, 8):
        state = initial_E.copy()
        dt = final_time / step_count
        for _ in range(step_count):
            if method == "semi_implicit":
                state = semi_implicit_trapezoidal_step(
                    state,
                    intensity_for,
                    dt_normalized=dt,
                    applied_field=material.applied_field,
                    background_intensity=material.background_intensity,
                    dx_normalized=dx_normalized,
                    xp=np,
                )
            elif method == "euler":
                state = euler_step(
                    state,
                    intensity_for(state),
                    dt_normalized=dt,
                    applied_field=material.applied_field,
                    background_intensity=material.background_intensity,
                    dx_normalized=dx_normalized,
                    xp=np,
                )
            else:
                raise ValueError(f"unknown prototype method: {method}")
        errors.append(np.linalg.norm(state - reference))
    return np.asarray(errors)


def test_semi_implicit_prototype_is_second_order_with_actual_optical_map():
    errors = _coupled_temporal_convergence_errors()
    orders = np.log2(errors[:-1] / errors[1:])

    assert errors[2] < errors[1] < errors[0]
    assert np.all(orders > 1.8)
    assert np.all(orders < 2.2)


def test_coupled_prototype_improves_temporal_order_over_euler_reference():
    semi_implicit_errors = _coupled_temporal_convergence_errors()
    euler_errors = _coupled_temporal_convergence_errors(method="euler")
    euler_orders = np.log2(euler_errors[:-1] / euler_errors[1:])

    assert np.all(euler_orders > 0.9)
    assert np.all(euler_orders < 1.1)
    assert np.all(semi_implicit_errors < euler_errors)


def test_coupled_step_uses_two_power_conserving_optical_passes():
    request, grid, A0, initial_E, optical_map = _coupled_optical_case()
    material = request.material
    power_initial = normalized_power(A0, grid)
    observed_powers = []

    def checked_intensity(E):
        A, intensity = optical_map(E)
        observed_powers.append(normalized_power(A, grid))
        return intensity

    stepped = semi_implicit_trapezoidal_step(
        initial_E,
        checked_intensity,
        dt_normalized=0.01,
        applied_field=material.applied_field,
        background_intensity=material.background_intensity,
        dx_normalized=(
            material.characteristic_wavenumber_per_um * grid.dx_um
        ),
        xp=np,
    )

    assert len(observed_powers) == 2
    assert np.allclose(observed_powers, power_initial, rtol=2e-14, atol=0.0)
    assert np.all(np.isfinite(stepped))


def test_semi_implicit_prototype_is_cupy_compatible_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    rng = np.random.default_rng(23)
    E = cp.asarray(rng.normal(scale=0.1, size=(2, 12, 3)), dtype=cp.float32)
    intensity = cp.asarray(
        0.2 + rng.random(size=E.shape),
        dtype=cp.float32,
    )

    solution = solve_periodic_variable_diffusion(
        E,
        intensity,
        alpha=0.1,
        dx_normalized=0.25,
        xp=cp,
    )
    stepped = semi_implicit_trapezoidal_step(
        E,
        lambda _state: intensity,
        dt_normalized=0.01,
        applied_field=0.2,
        background_intensity=0.1,
        dx_normalized=0.25,
        xp=cp,
    )
    _, second = periodic_derivatives_x(
        solution,
        dx_normalized=0.25,
        xp=cp,
    )
    residual = solution - 0.1 * intensity * second - E

    assert isinstance(solution, cp.ndarray)
    assert isinstance(stepped, cp.ndarray)
    assert solution.dtype == cp.float32
    assert stepped.dtype == cp.float32
    assert float(cp.max(cp.abs(residual)).item()) < 2e-5
    assert bool(cp.all(cp.isfinite(stepped)).item())
