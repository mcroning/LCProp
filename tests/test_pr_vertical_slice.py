import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.splitstep import (
    advance_prepared_response,
    hop_linear,
    linear_kernel,
)
from lcprop.pr.evolution import (
    centered_difference_symbols,
    conservative_timestep_limit,
    euler_step,
    hopping_rhs,
    legacy_conservative_timestep_limit,
    legacy_mode_timestep_rule,
    linearized_euler_mode_limit,
    paper_conservative_timestep_limit,
    paper_mode_timestep_rule,
    periodic_derivatives_x,
    semi_implicit_amplification,
    semi_implicit_conservative_timestep_limit,
    validate_timestep,
)
from lcprop.pr.optical_response import half_step_response_from_E
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.workflow import run_pr_timedependent


def test_dark_only_uniform_state_relaxes_toward_applied_field():
    E = np.full((12, 7), 0.2)
    background = 0.4
    applied = 0.75
    dt = 0.01
    intensity = np.full_like(E, background)

    actual = euler_step(
        E,
        intensity,
        dt_normalized=dt,
        applied_field=applied,
        background_intensity=background,
        dx_normalized=0.2,
        xp=np,
    )

    expected = E + dt * background * (applied - E)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_weak_sinusoidal_grating_matches_reduced_paper_equation():
    Nx, Ny = 128, 3
    dx = 0.15
    mode = 3
    x = np.arange(Nx) * dx
    k = 2.0 * np.pi * mode / (Nx * dx)
    epsilon = 1e-7
    background = 0.2
    base_intensity = 1.0 + background
    E = epsilon * np.sin(k * x)[:, None] * np.ones((1, Ny))
    intensity = base_intensity + epsilon * np.cos(k * x)[:, None]
    intensity = np.broadcast_to(intensity, E.shape).copy()

    actual = hopping_rhs(
        E,
        intensity,
        applied_field=0.0,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    I_x, _ = periodic_derivatives_x(
        intensity,
        dx_normalized=dx,
        xp=np,
    )
    discrete_k_squared = 4.0 * np.sin(0.5 * k * dx) ** 2 / dx**2
    reduced = -E * base_intensity * (1.0 + discrete_k_squared) + I_x

    # The omitted products of the two weak modulations are O(epsilon**2).
    assert np.max(np.abs(actual - reduced)) < 3e-14

    # Several explicit steps exhibit the same weak-grating build/decay law,
    # rather than merely agreeing at the initial derivative evaluation.
    dt = 1e-3
    full_state = np.zeros_like(E)
    reduced_state = np.zeros_like(E)
    for _ in range(12):
        full_state = euler_step(
            full_state,
            intensity,
            dt_normalized=dt,
            applied_field=0.0,
            background_intensity=background,
            dx_normalized=dx,
            xp=np,
        )
        reduced_state += dt * (
            -base_intensity
            * (1.0 + discrete_k_squared)
            * reduced_state
            + I_x
        )
    assert np.max(np.abs(full_state - reduced_state)) < 2e-14
    assert np.linalg.norm(full_state) > 0.0


def test_periodic_finite_difference_derivative_signs_along_x():
    Nx, Ny = 96, 5
    dx = 0.1
    mode = 4
    x = np.arange(Nx) * dx
    k = 2.0 * np.pi * mode / (Nx * dx)
    field = np.sin(k * x)[:, None] * np.ones((1, Ny))

    first, second = periodic_derivatives_x(
        field,
        dx_normalized=dx,
        xp=np,
    )
    first_expected = (
        np.sin(k * dx) / dx * np.cos(k * x)[:, None] * np.ones((1, Ny))
    )
    second_expected = (
        -4.0
        * np.sin(0.5 * k * dx) ** 2
        / dx**2
        * field
    )

    assert np.allclose(first, first_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(second, second_expected, rtol=1e-12, atol=1e-12)
    assert first[0, 0] > 0.0


@pytest.mark.parametrize("mode", (1, 16, 48), ids=("low", "moderate", "high"))
def test_centered_difference_frequency_sweep(mode):
    Nx, Ny = 128, 3
    dx = 0.1
    x = np.arange(Nx) * dx
    k = 2.0 * np.pi * mode / (Nx * dx)
    phase = k * x
    field = np.sin(phase)[:, None] * np.ones((1, Ny))
    first_exact = k * np.cos(phase)[:, None] * np.ones((1, Ny))
    second_exact = -(k**2) * field
    first, second = periodic_derivatives_x(
        field,
        dx_normalized=dx,
        xp=np,
    )
    k1, k2_squared = centered_difference_symbols(k, dx_normalized=dx)

    assert np.linalg.norm(first - first_exact) / np.linalg.norm(first_exact) == (
        pytest.approx(abs(k1 / k - 1.0), rel=1e-12, abs=1e-14)
    )
    assert np.linalg.norm(second - second_exact) / np.linalg.norm(second_exact) == (
        pytest.approx(abs(k2_squared / k**2 - 1.0), rel=1e-12, abs=1e-14)
    )

    epsilon = 1e-7
    background = 0.2
    base_intensity = 1.0 + background
    E = epsilon * field
    intensity = base_intensity + epsilon * np.cos(phase)[:, None]
    intensity = np.broadcast_to(intensity, E.shape).copy()
    rhs_discrete = hopping_rhs(
        E,
        intensity,
        applied_field=0.0,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    E_x = epsilon * first_exact
    E_xx = epsilon * second_exact
    I_x = -epsilon * k * np.sin(phase)[:, None]
    I_x = np.broadcast_to(I_x, E.shape)
    rhs_continuum = -(E * intensity - I_x) * (1.0 + E_x) + intensity * E_xx
    E_x_modified = epsilon * k1 * np.cos(phase)[:, None]
    E_x_modified = np.broadcast_to(E_x_modified, E.shape)
    E_xx_modified = -epsilon * k2_squared * field
    I_x_modified = -epsilon * k1 * np.sin(phase)[:, None]
    I_x_modified = np.broadcast_to(I_x_modified, E.shape)
    rhs_modified = -(E * intensity - I_x_modified) * (
        1.0 + E_x_modified
    ) + intensity * E_xx_modified
    assert np.allclose(rhs_discrete, rhs_modified, rtol=2e-9, atol=2e-14)
    relative_rhs_error = np.linalg.norm(rhs_discrete - rhs_continuum) / np.linalg.norm(
        rhs_continuum
    )
    assert relative_rhs_error > 0.0

    decay_rhs = hopping_rhs(
        E,
        np.full_like(E, base_intensity),
        applied_field=0.0,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    measured_decay_rate = -np.vdot(E, decay_rhs).real / np.vdot(E, E).real
    assert measured_decay_rate == pytest.approx(
        base_intensity * (1.0 + k2_squared),
        rel=1e-12,
    )

    buildup_intensity = base_intensity + epsilon * field
    buildup_rhs = hopping_rhs(
        np.zeros_like(E),
        buildup_intensity,
        applied_field=0.0,
        background_intensity=background,
        dx_normalized=dx,
        xp=np,
    )
    cosine = np.cos(phase)[:, None] * np.ones((1, Ny))
    measured_buildup = np.vdot(cosine, buildup_rhs).real / np.vdot(
        cosine,
        cosine,
    ).real
    assert measured_buildup == pytest.approx(epsilon * k1, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("k", (0.0, 2.0, 0.8 * np.pi / 0.2))
def test_timestep_rules_at_zero_moderate_and_high_resolved_k(k):
    dx = 0.2
    intensity = 1.3
    exact_euler = linearized_euler_mode_limit(
        k,
        dx_normalized=dx,
        uniform_intensity=intensity,
    )
    _, k2_squared = centered_difference_symbols(k, dx_normalized=dx)
    eigenvalue = -intensity * (1.0 + k2_squared)

    assert exact_euler == pytest.approx(
        2.0 / (intensity * (1.0 + k2_squared))
    )
    assert abs(1.0 + exact_euler * eigenvalue) == pytest.approx(1.0)
    assert abs(1.0 + 1.001 * exact_euler * eigenvalue) > 1.0
    equilibrium_field = 0.6
    k1, _ = centered_difference_symbols(k, dx_normalized=dx)
    drift_limit = linearized_euler_mode_limit(
        k,
        dx_normalized=dx,
        uniform_intensity=intensity,
        equilibrium_field=equilibrium_field,
    )
    drift_eigenvalue = eigenvalue - 1j * intensity * equilibrium_field * k1
    assert abs(1.0 + drift_limit * drift_eigenvalue) == pytest.approx(1.0)
    assert paper_mode_timestep_rule(k) == pytest.approx(1.0 / (4.0 * (1 + k**2)))
    if k == 0.0:
        assert legacy_mode_timestep_rule(k) == np.inf
        assert exact_euler / 8.0 == pytest.approx(1.0 / (4.0 * intensity))
    else:
        assert legacy_mode_timestep_rule(k) == pytest.approx(1.0 / (4.0 * k**2))


def test_conservative_timestep_guard_reconciles_paper_legacy_and_new_rules():
    grid_spec = GridSpec(
        Nx=16,
        Ny=4,
        x_aperture_um=16.0,
        y_aperture_um=4.0,
        dz_um=1.0,
        z_length_um=1.0,
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    material = PRMaterialSpec(
        dark_intensity=0.2,
        uniform_background_intensity=0.1,
        applied_field=0.5,
        characteristic_wavenumber_per_um_override=1.0,
    )
    limit = conservative_timestep_limit(grid, material)
    paper_limit = paper_conservative_timestep_limit(grid, material)
    legacy_limit = legacy_conservative_timestep_limit(grid, material)

    assert limit == pytest.approx(1.0 / (4.0 * 1.3 * (1.0 + 4.0)))
    assert paper_limit == pytest.approx(1.0 / (4.0 * (1.0 + np.pi**2)))
    legacy_kmax = 2.0 * np.pi * 7.0 / 16.0
    assert legacy_limit == pytest.approx(1.0 / (4.0 * legacy_kmax**2))
    assert validate_timestep(limit, grid, material) == pytest.approx(limit)
    with pytest.raises(ValueError, match="exceeds conservative PR limit"):
        validate_timestep(np.nextafter(limit, np.inf), grid, material)


def test_semi_implicit_guard_removes_grid_scale_diffusion_restriction():
    grid = make_grid(
        GridSpec(
            Nx=128,
            Ny=4,
            x_aperture_um=20.0,
            y_aperture_um=4.0,
            dz_um=1.0,
            z_length_um=1.0,
        ),
        real_dtype=np.float64,
    )
    material = PRMaterialSpec(
        dark_intensity=0.2,
        applied_field=0.5,
        characteristic_wavenumber_per_um_override=1.0,
    )
    euler_limit = conservative_timestep_limit(grid, material)
    semi_implicit_limit = semi_implicit_conservative_timestep_limit(
        grid,
        material,
    )

    # The zero mode retains explicit Heun reaction with exact boundary 2/I0;
    # the conservative policy uses one eighth of that boundary.
    assert semi_implicit_limit == pytest.approx(1.0 / (4.0 * 1.2))
    assert semi_implicit_limit > 100.0 * euler_limit
    assert validate_timestep(
        semi_implicit_limit,
        grid,
        material,
        integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
    ) == pytest.approx(semi_implicit_limit)
    with pytest.raises(ValueError, match=PR_SEMI_IMPLICIT_INTEGRATOR):
        validate_timestep(
            np.nextafter(semi_implicit_limit, np.inf),
            grid,
            material,
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        )


def test_semi_implicit_scalar_amplification_reduces_to_cn_and_heun():
    dt = 0.3
    lambda_implicit = -2.0
    lambda_explicit = -0.7

    cn = semi_implicit_amplification(
        dt,
        lambda_implicit=lambda_implicit,
        lambda_explicit=0.0,
    )
    heun = semi_implicit_amplification(
        dt,
        lambda_implicit=0.0,
        lambda_explicit=lambda_explicit,
    )

    z_implicit = dt * lambda_implicit
    z_explicit = dt * lambda_explicit
    assert cn == pytest.approx(
        (1.0 + 0.5 * z_implicit) / (1.0 - 0.5 * z_implicit)
    )
    assert heun == pytest.approx(
        1.0 + z_explicit + 0.5 * z_explicit**2
    )


def test_zero_pr_response_equals_pure_diffraction():
    rng = np.random.default_rng(12)
    A0 = rng.normal(size=(2, 12, 10)) + 1j * rng.normal(size=(2, 12, 10))
    grid = make_grid(
        GridSpec(
            Nx=12,
            Ny=10,
            x_aperture_um=24.0,
            y_aperture_um=20.0,
            dz_um=0.5,
            z_length_um=0.5,
        ),
        real_dtype=np.float64,
    )
    Nsub = 3
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / Nsub,
        wavelength=0.633,
        n_ref=2.4,
        xp=np,
    )
    response = half_step_response_from_E(
        np.linspace(-1.0, 1.0, 120).reshape(12, 10),
        dz_substep_um=grid.dz_um / Nsub,
        wavelength_um=0.633,
        interaction_length_um=grid.spec.z_length_um,
        gain_length_product=0.0,
        xp=np,
    )

    actual = A0.copy()
    advance_prepared_response(
        actual,
        kernel=kernel,
        half_step_response=response,
        Nsub=Nsub,
        xp=np,
    )
    expected = A0.copy()
    for _ in range(Nsub):
        expected = hop_linear(expected, kernel, xp=np)

    assert np.allclose(response, 1.0)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_nonzero_pr_state_produces_expected_half_substep_screen():
    E = np.asarray([[0.0, 0.2], [-0.4, 0.6]])
    dz_substep_um = 2.5
    length_um = 20.0
    gain_length_product = 0.8

    actual = half_step_response_from_E(
        E,
        dz_substep_um=dz_substep_um,
        wavelength_um=0.633,
        interaction_length_um=length_um,
        gain_length_product=gain_length_product,
        xp=np,
    )
    expected = np.exp(
        -1j * gain_length_product * E * dz_substep_um / length_um
    )

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_real_pr_phase_response_conserves_optical_power():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(2, 14, 9)) + 1j * rng.normal(size=(2, 14, 9))
    E = rng.normal(size=(14, 9))
    fxy2 = np.fft.fftfreq(14, d=1.0)[:, None] ** 2
    fxy2 = fxy2 + np.fft.fftfreq(9, d=1.0)[None, :] ** 2
    kernel = linear_kernel(
        fxy2,
        dz=0.4,
        wavelength=0.633,
        n_ref=2.4,
        xp=np,
    )
    response = half_step_response_from_E(
        E,
        dz_substep_um=0.4,
        wavelength_um=0.633,
        interaction_length_um=10.0,
        gain_length_product=2.0,
        xp=np,
    )
    power_before = np.sum(np.abs(A) ** 2)

    advance_prepared_response(
        A,
        kernel=kernel,
        half_step_response=response,
        xp=np,
    )

    assert np.sum(np.abs(A) ** 2) == pytest.approx(power_before, rel=2e-14)


def test_frozen_response_lie_and_strang_converge_but_strang_is_more_accurate():
    Nx, Ny = 24, 20
    dx = dy = 0.5
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    A0 = np.exp(-((X - 0.7) ** 2 + (Y + 0.4) ** 2) / 3.0)[None].astype(
        np.complex128
    )
    potential = 0.7 * np.cos(2.0 * np.pi * X / (Nx * dx))
    potential += 0.35 * np.sin(4.0 * np.pi * Y / (Ny * dy))
    fxy2 = (
        np.fft.fftfreq(Nx, d=dx)[:, None] ** 2
        + np.fft.fftfreq(Ny, d=dy)[None, :] ** 2
    )
    length = 2.0

    def propagate(nsteps, *, strang):
        dz = length / nsteps
        kernel = linear_kernel(
            fxy2,
            dz=dz,
            wavelength=0.633,
            n_ref=2.4,
            xp=np,
        )
        A = A0.copy()
        if strang:
            half_response = np.exp(0.5j * potential * dz)
            for _ in range(nsteps):
                advance_prepared_response(
                    A,
                    kernel=kernel,
                    half_step_response=half_response,
                    xp=np,
                )
        else:
            full_response = np.exp(1j * potential * dz)
            for _ in range(nsteps):
                A[...] = hop_linear(A, kernel, xp=np) * full_response[None]
        return A

    reference = propagate(2048, strang=True)
    lie_errors = []
    strang_errors = []
    for nsteps in (4, 8, 16):
        lie_errors.append(np.linalg.norm(propagate(nsteps, strang=False) - reference))
        strang_errors.append(np.linalg.norm(propagate(nsteps, strang=True) - reference))

    assert lie_errors[2] < lie_errors[1] < lie_errors[0]
    assert strang_errors[2] < strang_errors[1] < strang_errors[0]
    assert all(s < l for s, l in zip(strang_errors, lie_errors))
    assert lie_errors[0] / lie_errors[1] > 1.8
    assert strang_errors[0] / strang_errors[1] > 3.5


def _plane_wave_request(*, backend="numpy", integrator="euler"):
    grid = GridSpec(
        Nx=8,
        Ny=6,
        x_aperture_um=80.0,
        y_aperture_um=60.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=20.0,
                waist_y_um=20.0,
            ),
        )
    )
    material = PRMaterialSpec(
        dark_intensity=0.2,
        uniform_background_intensity=0.1,
        applied_field=0.5,
        gain_length_product=0.4,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.1,
    )
    solver = PRSolverOptions(
        Nt=3,
        dt_normalized=0.01,
        optical_substeps=2,
        integrator=integrator,
    )
    return PRRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=solver,
        backend=BackendSpec(backend=backend, precision="float64", verbose=False),
        initial_A=np.ones((1, grid.Nx, grid.Ny), dtype=np.complex128),
    )


def test_small_plane_wave_workflow_matches_analytic_uniform_prediction():
    request = _plane_wave_request()
    result = run_pr_timedependent(request)
    background = request.material.background_intensity
    intensity = 1.0 + background
    multiplier = 1.0 - request.solver.dt_normalized * intensity
    E_expected = (
        request.material.applied_field
        * background
        / intensity
        * (1.0 - multiplier**request.solver.Nt)
    )
    phase_expected = np.exp(
        -2j * request.material.gain_length_product * E_expected
    )

    assert result.diagnostics["backend"]["backend"] == "numpy"
    assert np.allclose(result.E_final, E_expected, rtol=0.0, atol=1e-14)
    assert np.allclose(result.source_intensity_stack, intensity, atol=1e-13)
    assert np.allclose(result.A_final, phase_expected, rtol=1e-12, atol=1e-12)
    assert result.power_final == pytest.approx(result.power_initial, rel=2e-14)


def test_semi_implicit_plane_wave_workflow_matches_heun_prediction():
    request = _plane_wave_request(integrator=PR_SEMI_IMPLICIT_INTEGRATOR)
    result = run_pr_timedependent(request)
    background = request.material.background_intensity
    intensity = 1.0 + background
    dt_intensity = request.solver.dt_normalized * intensity
    multiplier = 1.0 - dt_intensity + 0.5 * dt_intensity**2
    equilibrium = request.material.applied_field * background / intensity
    E_expected = equilibrium * (1.0 - multiplier**request.solver.Nt)
    phase_expected = np.exp(
        -2j * request.material.gain_length_product * E_expected
    )

    assert result.diagnostics["integrator"] == PR_SEMI_IMPLICIT_INTEGRATOR
    assert np.allclose(result.E_final, E_expected, rtol=0.0, atol=2e-14)
    assert np.allclose(result.source_intensity_stack, intensity, atol=1e-13)
    assert np.allclose(result.A_final, phase_expected, rtol=1e-12, atol=1e-12)
    assert result.power_final == pytest.approx(result.power_initial, rel=2e-14)


def test_pr_primitive_functions_are_cupy_compatible_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    E = cp.zeros((8, 4), dtype=cp.float32)
    intensity = cp.ones_like(E) * 1.2
    rhs = hopping_rhs(
        E,
        intensity,
        applied_field=0.5,
        background_intensity=0.2,
        dx_normalized=0.1,
        xp=cp,
    )
    response = half_step_response_from_E(
        rhs,
        dz_substep_um=1.0,
        wavelength_um=0.633,
        interaction_length_um=10.0,
        gain_length_product=0.5,
        xp=cp,
    )

    assert isinstance(rhs, cp.ndarray)
    assert isinstance(response, cp.ndarray)
    assert cp.allclose(rhs, 0.1)
    assert cp.allclose(cp.abs(response), 1.0)

    result = run_pr_timedependent(_plane_wave_request(backend="cupy"))
    assert result.diagnostics["backend"]["backend"] == "cupy"
    assert result.power_final == pytest.approx(result.power_initial, rel=2e-6)
