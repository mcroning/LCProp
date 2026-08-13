from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.lc.optical_response import compute_neff
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.lc.propagation import advance_slice
from lcprop.workflows.runtime import (
    build_runtime_components,
    initial_A_field,
)
import lcprop.workflows.static as static_workflow


@dataclass(frozen=True)
class IndexProbe:
    grid_x_um: float
    grid_y_um: float
    theta_rad: float
    delta_theta_rad: float
    delta_n: float
    delta_n_from_bias: float
    d_delta_n_dx_per_um: float


def _offaxis_request(*, strategy: str, max_iterations: int = 4) -> StaticRunRequest:
    return StaticRunRequest(
        grid=GridSpec(
            Nx=256,
            Ny=64,
            dz_um=20.0,
            x_aperture_um=200.0,
            y_aperture_um=100.0,
            z_length_um=3000.0,
        ),
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    name="off-axis",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=12.0,
                    waist_y_um=12.0,
                    x0_um=-20.0,
                    y0_um=0.0,
                    coherence_group="probe",
                ),
            )
        ),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy=strategy,
                theta_solver=(
                    "picard_cn" if strategy == "local_self_consistent" else "none"
                ),
                optics_solver="splitstep",
                coupling=(
                    "self_consistent"
                    if strategy == "local_self_consistent"
                    else "frozen"
                ),
            ),
            max_iterations=max_iterations,
        ),
        output=OutputOptions(save_slices=False),
    )


def _centroid_x(A: np.ndarray, x_um: np.ndarray) -> float:
    intensity = np.sum(np.abs(A) ** 2, axis=0)
    return float(np.sum(intensity * x_um[:, None]) / np.sum(intensity))


def _centroid_y(A: np.ndarray, y_um: np.ndarray) -> float:
    intensity = np.sum(np.abs(A) ** 2, axis=0)
    return float(np.sum(intensity * y_um[None, :]) / np.sum(intensity))


def _propagate_trajectory(components, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = initial_A_field(components)
    centroids = [_centroid_x(A, components.grid.x_um)]
    for _ in range(components.grid.Nz):
        advance_slice(
            A,
            theta,
            kernel=components.kernel,
            dz=components.grid.dz_um,
            wavelength=components.wavelength_um,
            n_ref=components.n_ref,
            ne=components.request.material.ne,
            no=components.request.material.no,
            xp=components.grid.xp,
        )
        centroids.append(_centroid_x(A, components.grid.x_um))
    z_um = np.arange(components.grid.Nz + 1) * components.grid.dz_um
    return z_um, np.asarray(centroids)


def _index_probe(
    theta: np.ndarray,
    theta_bias: np.ndarray,
    components,
    *,
    x0_um: float,
    y0_um: float,
) -> IndexProbe:
    ix = int(np.argmin(np.abs(components.grid.x_um - x0_um)))
    iy = int(np.argmin(np.abs(components.grid.y_um - y0_um)))
    delta_n = compute_neff(
        theta,
        ne=components.request.material.ne,
        no=components.request.material.no,
    ) - components.n_ref
    bias_delta_n = compute_neff(
        theta_bias,
        ne=components.request.material.ne,
        no=components.request.material.no,
    ) - components.n_ref
    gradient = np.gradient(delta_n, components.grid.dx_um, axis=0)
    return IndexProbe(
        grid_x_um=float(components.grid.x_um[ix]),
        grid_y_um=float(components.grid.y_um[iy]),
        theta_rad=float(theta[ix, iy]),
        delta_theta_rad=float(theta[ix, iy] - theta_bias[ix, iy]),
        delta_n=float(delta_n[ix, iy]),
        delta_n_from_bias=float(delta_n[ix, iy] - bias_delta_n[ix, iy]),
        d_delta_n_dx_per_um=float(gradient[ix, iy]),
    )


def test_offaxis_beam_receives_bias_and_updated_self_consistent_theta(
    monkeypatch,
):
    fixed_request = _offaxis_request(strategy="fixed_theta")
    components = build_runtime_components(fixed_request)
    launch_centroid = _centroid_x(
        components.launch.A0,
        components.grid.x_um,
    )
    assert abs(launch_centroid - (-20.0)) < components.grid.dx_um
    assert (
        abs(_centroid_y(components.launch.A0, components.grid.y_um))
        < components.grid.dy_um
    )

    theta_zero = np.zeros_like(components.bias.theta_2d)
    z_um, homogeneous_x = _propagate_trajectory(components, theta_zero)
    _, frozen_bias_x = _propagate_trajectory(
        components,
        components.bias.theta_2d,
    )
    fixed_result = static_workflow.run_static(fixed_request)
    assert np.isclose(
        frozen_bias_x[-1],
        _centroid_x(fixed_result.A_final, components.grid.x_um),
    )

    original_advance = static_workflow.advance_slice_with_midpoint_source
    trial_centroids: list[float] = []

    def observed_advance(A, theta, **kwargs):
        result = original_advance(A, theta, **kwargs)
        trial_centroids.append(_centroid_x(result[0], components.grid.x_um))
        return result

    monkeypatch.setattr(
        static_workflow,
        "advance_slice_with_midpoint_source",
        observed_advance,
    )
    self_request = _offaxis_request(
        strategy="local_self_consistent",
        max_iterations=4,
    )
    self_result = static_workflow.run_static(self_request)
    theta_sources = np.asarray(self_result.theta_final)
    accepted_centroids = []
    call_index = 0
    for summary in self_result.slice_summaries:
        call_index += 1 + summary.optical_passes
        accepted_centroids.append(trial_centroids[call_index - 1])
    self_consistent_x = np.asarray([launch_centroid, *accepted_centroids])

    assert len(trial_centroids) == (
        sum(1 + summary.optical_passes for summary in self_result.slice_summaries)
    )
    assert theta_sources.shape == (
        components.grid.Nz,
        components.grid.Nx,
        components.grid.Ny,
    )
    source_updates = [
        float(np.max(np.abs(theta_sources[0] - components.bias.theta_2d))),
        *(
            float(np.max(np.abs(current - previous)))
            for previous, current in zip(theta_sources, theta_sources[1:])
        ),
    ]
    assert all(update > 1.0e-10 for update in source_updates)
    assert np.allclose(
        self_consistent_x[-1],
        _centroid_x(self_result.A_final, components.grid.x_um),
    )

    homogeneous_probe = _index_probe(
        theta_zero,
        components.bias.theta_2d,
        components,
        x0_um=-20.0,
        y0_um=0.0,
    )
    frozen_probe = _index_probe(
        components.bias.theta_2d,
        components.bias.theta_2d,
        components,
        x0_um=-20.0,
        y0_um=0.0,
    )
    self_probe = _index_probe(
        theta_sources[0],
        components.bias.theta_2d,
        components,
        x0_um=-20.0,
        y0_um=0.0,
    )

    # Homogeneous propagation must not masquerade as the frozen bias/GRIN
    # response, and the self-consistent source must alter that response.
    assert np.max(np.abs(homogeneous_x - launch_centroid)) < 0.1
    assert np.max(np.abs(frozen_bias_x - homogeneous_x)) > 0.1
    assert np.max(np.abs(self_consistent_x - frozen_bias_x)) > 0.01

    sample_indices = np.linspace(0, components.grid.Nz, 7, dtype=int)
    print("\nz_um homogeneous_x_um frozen_bias_x_um self_consistent_x_um")
    for index in sample_indices:
        print(
            f"{z_um[index]:.0f} "
            f"{homogeneous_x[index]:.9f} "
            f"{frozen_bias_x[index]:.9f} "
            f"{self_consistent_x[index]:.9f}"
        )
    print(f"homogeneous probe: {homogeneous_probe}")
    print(f"frozen probe: {frozen_probe}")
    print(f"self-consistent probe: {self_probe}")
    print(f"theta-source max|delta theta| per pass: {source_updates}")
