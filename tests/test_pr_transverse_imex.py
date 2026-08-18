from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    run_pr_transverse_timedependent,
)
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.transport import (
    explicit_euler_step,
    imex_euler_step,
    state_from_potential,
)


def _request(*, steps: int):
    grid = GridSpec(
        Nx=12, Ny=10,
        x_aperture_um=48.0, y_aperture_um=40.0,
        dz_um=5.0, z_length_um=10.0,
    )
    return PRTransverseRunRequest(
        grid=grid,
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=8.0,
            coherence_group="transverse-pr-imex",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            gain_length_product=0.05,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps, dt_normalized=1.0e-4, optical_substeps=1,
        ),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def _smooth_fixed_source_case():
    nx, ny = 23, 19
    dx, dy = 0.35, 0.41
    x = np.arange(nx) * 2.0 * math.pi / nx
    y = np.arange(ny) * 2.0 * math.pi / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = (
        0.002 * np.cos(2.0 * X)
        + 0.001 * np.sin(3.0 * Y)
        + 0.0007 * np.cos(X - Y)
    )
    source = 0.3 + 0.2 * (1.0 + np.cos(X)) + 0.1 * (1.0 + np.sin(2.0 * Y))
    return psi, source, dx, dy


def _integrate_imex(psi, source, *, dx, dy, final_time, steps):
    state = psi.copy()
    dt = float(final_time) / int(steps)
    for _ in range(int(steps)):
        state = imex_euler_step(
            state,
            source,
            dt_normalized=dt,
            dx_normalized=dx,
            dy_normalized=dy,
        )
    return state


def test_imex_one_step_converges_quadratically_to_explicit_as_dt_shrinks():
    psi, source, dx, dy = _smooth_fixed_source_case()
    differences = []
    for dt in (4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5):
        explicit = explicit_euler_step(
            psi, source, dt_normalized=dt,
            dx_normalized=dx, dy_normalized=dy,
        )
        imex = imex_euler_step(
            psi, source, dt_normalized=dt,
            dx_normalized=dx, dy_normalized=dy,
        )
        differences.append(float(np.linalg.norm(imex - explicit)))

    assert all(
        coarse / fine > 3.9
        for coarse, fine in zip(differences, differences[1:])
    )


def test_imex_has_first_order_global_temporal_convergence():
    psi, source, dx, dy = _smooth_fixed_source_case()
    final_time = 0.02
    reference = _integrate_imex(
        psi, source, dx=dx, dy=dy, final_time=final_time, steps=512
    )
    errors = []
    for steps in (16, 32, 64, 128):
        candidate = _integrate_imex(
            psi, source, dx=dx, dy=dy, final_time=final_time, steps=steps
        )
        errors.append(
            float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
        )

    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:])]
    assert all(1.8 < ratio < 2.6 for ratio in ratios)


def test_uniform_source_modal_decay_uses_per_plane_istar():
    nx, ny = 31, 29
    dx = dy = 0.2
    mode = 3
    amplitude = 1.0e-4
    profile = amplitude * np.cos(
        2.0 * math.pi * mode * np.arange(nx) / nx
    )[:, None] * np.ones((1, ny))
    psi = np.stack((profile, profile))
    source = np.empty_like(psi)
    source[0] = 0.4
    source[1] = 0.9
    dt = 0.03

    candidate = imex_euler_step(
        psi, source, dt_normalized=dt,
        dx_normalized=dx, dy_normalized=dy,
    )
    initial_hat = np.fft.fft2(psi, axes=(-2, -1))
    candidate_hat = np.fft.fft2(candidate, axes=(-2, -1))
    k = 2.0 * math.pi * mode / (nx * dx)
    measured = np.abs(candidate_hat[:, mode, 0] / initial_hat[:, mode, 0])
    expected = 1.0 / (1.0 + dt * np.array((0.4, 0.9)) * (1.0 + k * k))
    np.testing.assert_allclose(measured, expected, rtol=0.0, atol=5.0e-13)


def test_imex_preserves_carrier_curl_gauss_and_gauge():
    psi, source, dx, dy = _smooth_fixed_source_case()
    initial = state_from_potential(psi, dx_normalized=dx, dy_normalized=dy)
    candidate_psi = imex_euler_step(
        psi, source, dt_normalized=0.002,
        dx_normalized=dx, dy_normalized=dy,
    )
    candidate = state_from_potential(
        candidate_psi, dx_normalized=dx, dy_normalized=dy
    )
    diagnostics = state_diagnostics(
        candidate, dx_normalized=dx, dy_normalized=dy
    )
    np.testing.assert_allclose(
        np.sum(candidate.carrier_density),
        np.sum(initial.carrier_density),
        rtol=0.0,
        atol=2.0e-13,
    )
    assert diagnostics["curl_max"] < 2.0e-16
    assert diagnostics["gauss_max"] < 3.0e-15
    assert diagnostics["potential_mean_max_abs"] < 1.0e-18


def test_imex_is_stable_beyond_explicit_high_mode_limit():
    nx, ny = 31, 29
    dx = dy = 0.2
    mode = 14
    initial = 1.0e-10 * np.cos(
        2.0 * math.pi * mode * np.arange(nx) / nx
    )[:, None] * np.ones((1, ny))
    source = np.ones_like(initial)
    explicit = initial.copy()
    imex = initial.copy()
    for _ in range(8):
        explicit = explicit_euler_step(
            explicit, source, dt_normalized=0.02,
            dx_normalized=dx, dy_normalized=dy,
        )
        imex = imex_euler_step(
            imex, source, dt_normalized=0.02,
            dx_normalized=dx, dy_normalized=dy,
        )

    initial_norm = np.linalg.norm(initial)
    assert np.linalg.norm(explicit) > 1000.0 * initial_norm
    assert np.linalg.norm(imex) < initial_norm
    assert np.all(np.isfinite(imex))


def test_workflow_defaults_to_imex_and_retains_explicit_reference():
    default_options = PRTransverseSolverOptions(Nt=1, dt_normalized=1.0e-4)
    assert default_options.integrator == PR_TRANSVERSE_IMEX_EULER
    request = _request(steps=1)
    imex = run_pr_transverse_timedependent(request)
    explicit = run_pr_transverse_timedependent(
        replace(
            request,
            solver=replace(
                request.solver,
                integrator=PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
            ),
        )
    )
    assert imex.diagnostics["integrator_policy"] == "production_first_order_spectral_imex"
    assert explicit.diagnostics["integrator_policy"] == (
        "transparent_reference_not_production_default"
    )
    assert np.all(np.isfinite(imex.psi_final))
    assert np.all(np.isfinite(explicit.psi_final))
