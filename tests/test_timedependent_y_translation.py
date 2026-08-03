from __future__ import annotations

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.grid import make_grid
from lcprop.core.requests import OutputOptions, TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.optics.splitstep import total_intensity
import lcprop.optics.splitstep as splitstep
import lcprop.workflows.timedependent as timedependent


WAVELENGTH_UM = 0.633
Y_APERTURE_UM = 100.0
TILT_Y_RAD_PER_UM = -0.34


def _request(*, y0_um: float = 0.0, power_mW: float = 0.05):
    grid = GridSpec(
        Nx=64,
        Ny=128,
        dz_um=20.0,
        x_aperture_um=75.0,
        y_aperture_um=Y_APERTURE_UM,
        z_length_um=600.0,
    )
    return TimeDependentRunRequest(
        grid=grid,
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=WAVELENGTH_UM,
                    power_mW=power_mW,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                    y0_um=y0_um,
                    tilt_y_rad_per_um=TILT_Y_RAD_PER_UM,
                    coherence_group="A",
                ),
            )
        ),
        solver=TimeDependentSolverOptions(
            Nt=3,
            dt=7.5e-4,
            gamma_z=0.0,
            max_picard_iter=4,
        ),
        output=OutputOptions(),
        initial_theta=np.full((grid.Nx, grid.Ny), 0.5),
    )


def _circular_y_centroid(intensity, y_um) -> float:
    marginal = np.asarray(intensity).sum(axis=0)
    phase = np.exp(2j * np.pi * np.asarray(y_um) / Y_APERTURE_UM)
    return float(np.angle(np.sum(marginal * phase)) * Y_APERTURE_UM / (2 * np.pi))


def _unwrapped_midpoint_trajectory(result, grid, *, y0_um: float) -> tuple[np.ndarray, np.ndarray]:
    centroids = [y0_um]
    centroids.extend(
        _circular_y_centroid(intensity, grid.y_um)
        for intensity in np.asarray(result.final_intensity_stack)
    )
    unwrapped = (
        np.unwrap(np.asarray(centroids) * 2 * np.pi / Y_APERTURE_UM)
        * Y_APERTURE_UM
        / (2 * np.pi)
    )
    z_um = np.concatenate(
        ([0.0], (np.arange(grid.Nz, dtype=float) + 0.5) * grid.dz_um)
    )
    return z_um, unwrapped


def _expected_slope(n_ref: float = 1.5) -> float:
    k_medium = 2 * np.pi * n_ref / WAVELENGTH_UM
    return TILT_Y_RAD_PER_UM / k_medium


def test_frozen_director_td_uses_shared_advance_and_preserves_tilt(monkeypatch):
    def frozen_theta_factory(*args, **kwargs):
        def frozen_theta_step(theta_k, intensity, theta_prev, theta_next, k):
            return theta_k.copy()

        return frozen_theta_step

    monkeypatch.setattr(
        timedependent,
        "make_zcoupled_theta_step",
        frozen_theta_factory,
    )
    shared_advance_calls = 0
    original_advance = splitstep.advance_slice

    def observed_advance(*args, **kwargs):
        nonlocal shared_advance_calls
        shared_advance_calls += 1
        return original_advance(*args, **kwargs)

    monkeypatch.setattr(splitstep, "advance_slice", observed_advance)

    results = []
    trajectories = []
    for y0_um in (0.0, 12.5):
        request = _request(y0_um=y0_um)
        result = timedependent.run_timedependent(request)
        grid = make_grid(request.grid, real_dtype=np.float64)
        z_um, y_um = _unwrapped_midpoint_trajectory(
            result,
            grid,
            y0_um=y0_um,
        )
        results.append(result)
        trajectories.append(y_um - y0_um)

        measured_slope = float(np.polyfit(z_um, y_um, 1)[0])
        assert np.all(np.diff(y_um) < 0.0)
        assert measured_slope == pytest.approx(_expected_slope(), rel=5e-4)
        assert result.power_final == pytest.approx(result.power_initial, abs=2e-12)

    # Each run performs an initial reconstruction, one accepted and one
    # display reconstruction per TD update, and one final reconstruction.
    expected_calls_per_run = (2 * 3 + 2) * 30
    assert shared_advance_calls == 2 * expected_calls_per_run

    reference, shifted = results
    np.testing.assert_allclose(
        trajectories[1], trajectories[0], rtol=0.0, atol=5e-7
    )
    for field_name in (
        "initial_intensity_stack",
        "final_intensity_stack",
        "initial_source_intensity_stack",
        "final_source_intensity_stack",
        "theta_final",
    ):
        np.testing.assert_allclose(
            np.roll(np.asarray(getattr(shifted, field_name)), -16, axis=-1),
            np.asarray(getattr(reference, field_name)),
            rtol=1e-11,
            atol=1e-13,
        )


def test_weakly_self_consistent_td_has_no_periodic_y_restoring_force():
    request = _request(power_mW=0.05)
    trajectories = []

    def observe(progress):
        intensity_stack = progress.latest_field_state["current_intensity_stack"]
        probe = type("Probe", (), {"final_intensity_stack": intensity_stack})
        grid = make_grid(request.grid, real_dtype=np.float64)
        _, y_um = _unwrapped_midpoint_trajectory(probe, grid, y0_um=0.0)
        trajectories.append(y_um)

    result = timedependent.run_timedependent(
        request,
        progress_callback=observe,
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    z_um, final_y = _unwrapped_midpoint_trajectory(result, grid, y0_um=0.0)
    trajectories.append(final_y)

    assert len(trajectories) == request.solver.Nt + 1
    for y_um in trajectories:
        measured_slope = float(np.polyfit(z_um, y_um, 1)[0])
        assert np.all(np.diff(y_um) < 0.0)
        assert measured_slope == pytest.approx(_expected_slope(), rel=0.02)
    assert result.power_final == pytest.approx(result.power_initial, abs=2e-12)
