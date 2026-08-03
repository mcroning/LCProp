from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.grid import make_grid
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.optics.splitstep import advance_slice, total_intensity
from lcprop.workflows.runtime import build_runtime_components
from lcprop.workflows.static import run_static


WAVELENGTH_UM = 0.633
Y_APERTURE_UM = 100.0
TILT_Y_RAD_PER_UM = -0.34


def _circular_y_centroid(intensity, y_um) -> float:
    marginal = np.asarray(intensity).sum(axis=0)
    phase = np.exp(2j * np.pi * np.asarray(y_um) / Y_APERTURE_UM)
    return float(np.angle(np.sum(marginal * phase)) * Y_APERTURE_UM / (2 * np.pi))


def _unwrap_y(values) -> np.ndarray:
    return (
        np.unwrap(np.asarray(values) * 2 * np.pi / Y_APERTURE_UM)
        * Y_APERTURE_UM
        / (2 * np.pi)
    )


def _request(*, strategy: str, y0_um: float = 0.0, z_length_um: float = 600.0):
    self_consistent = strategy == "local_self_consistent"
    return StaticRunRequest(
        grid=GridSpec(
            Nx=64,
            Ny=128,
            dz_um=20.0,
            x_aperture_um=75.0,
            y_aperture_um=Y_APERTURE_UM,
            z_length_um=z_length_um,
        ),
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=WAVELENGTH_UM,
                    power_mW=1.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                    y0_um=y0_um,
                    tilt_y_rad_per_um=TILT_Y_RAD_PER_UM,
                    coherence_group="A",
                ),
            )
        ),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy=strategy,
                theta_solver="picard_cn" if self_consistent else "none",
                optics_solver="splitstep",
                coupling="self_consistent" if self_consistent else "frozen",
            ),
            static_max_coupled_passes=10,
        ),
        output=OutputOptions(),
    )


@pytest.mark.parametrize("director", ("uniform", "frozen_bias"))
def test_tilted_beam_has_linear_periodic_y_trajectory(director):
    request = _request(strategy="fixed_theta", z_length_um=3000.0)
    components = build_runtime_components(request)
    theta = (
        np.full_like(components.bias.theta_2d, 0.5)
        if director == "uniform"
        else components.bias.theta_2d
    )
    A = components.launch.A0.copy()
    centroids = [
        _circular_y_centroid(total_intensity(A), components.grid.y_um)
    ]
    for _ in range(components.grid.Nz):
        advance_slice(
            A,
            theta,
            kernel=components.kernel,
            dz=components.grid.dz_um,
            wavelength=components.wavelength_um,
            n_ref=components.n_ref,
            ne=request.material.ne,
            no=request.material.no,
            Nsub=components.optical_substeps.Nsub,
        )
        centroids.append(
            _circular_y_centroid(total_intensity(A), components.grid.y_um)
        )

    y_unwrapped = _unwrap_y(centroids)
    z_um = np.arange(components.grid.Nz + 1) * components.grid.dz_um
    measured_slope = float(np.polyfit(z_um, y_unwrapped, 1)[0])
    k_medium = 2 * np.pi * components.n_ref / WAVELENGTH_UM
    angular_spectrum_slope = TILT_Y_RAD_PER_UM / np.sqrt(
        k_medium**2 - TILT_Y_RAD_PER_UM**2
    )

    assert np.all(np.diff(y_unwrapped) < 0.0)
    assert y_unwrapped[-1] < -Y_APERTURE_UM / 2
    assert measured_slope == pytest.approx(angular_spectrum_slope, rel=5e-4)


def test_local_static_y_motion_is_monotonic_and_translation_equivariant():
    results = []
    trajectories = []
    for y0_um in (0.0, 12.5):
        request = _request(
            strategy="local_self_consistent",
            y0_um=y0_um,
        )
        grid = make_grid(request.grid, real_dtype=np.float64)
        centroids = [y0_um]

        def observe(progress):
            A = np.asarray(progress.latest_field_state["A_current"])
            centroids.append(
                _circular_y_centroid(total_intensity(A), grid.y_um)
            )

        results.append(run_static(request, progress_callback=observe))
        trajectories.append(_unwrap_y(centroids) - y0_um)

    reference, shifted = results
    reference_y, shifted_y = trajectories
    z_um = np.arange(reference_y.size) * 20.0
    measured_slope = float(np.polyfit(z_um, reference_y, 1)[0])
    k_medium = 2 * np.pi * 1.5 / WAVELENGTH_UM
    expected_slope = TILT_Y_RAD_PER_UM / np.sqrt(
        k_medium**2 - TILT_Y_RAD_PER_UM**2
    )

    assert np.all(np.diff(reference_y) < 0.0)
    assert measured_slope == pytest.approx(expected_slope, rel=0.05)
    np.testing.assert_allclose(shifted_y, reference_y, rtol=0.0, atol=5e-7)

    # 12.5 um is exactly 16 samples on this periodic y grid.
    inverse_shift = -16
    for field_name in (
        "intensity_stack",
        "theta_intensity_stack",
        "theta_final",
    ):
        np.testing.assert_allclose(
            np.roll(np.asarray(getattr(shifted, field_name)), inverse_shift, axis=-1),
            np.asarray(getattr(reference, field_name)),
            rtol=1e-11,
            atol=1e-13,
        )

    assert reference.power_final == pytest.approx(reference.power_initial, abs=1e-11)
    assert shifted.power_final == pytest.approx(shifted.power_initial, abs=1e-11)


def test_static_warns_when_y_waist_is_underresolved():
    request = _request(strategy="fixed_theta")
    request = replace(
        request,
        grid=GridSpec(
            Nx=64,
            Ny=64,
            dz_um=20.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=20.0,
        ),
    )

    result = run_static(request)

    assert any("minimum waist_y/dy=1.92" in warning for warning in result.warnings)
