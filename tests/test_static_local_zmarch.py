from __future__ import annotations

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.grid import make_grid
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.products.data_model import from_static_result
from lcprop.workflows.timedependent import run_timedependent
import lcprop.workflows.static as static_workflow


def _workflow() -> StaticWorkflowOptions:
    return StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )


def _request_parts() -> dict:
    return {
        "grid": GridSpec(
            Nx=128,
            Ny=128,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=500.0,
        ),
        "material": LCMaterial(),
        "bias": BiasSpec(),
        "beams": BeamStack(
            channels=(
                BeamChannel(
                    name="off-axis",
                    power_mW=1.0,
                    waist_x_um=10.0,
                    waist_y_um=10.0,
                    x0_um=-20.0,
                    y0_um=0.0,
                    theta_weight=1.0,
                    coherence_group="A",
                ),
            )
        ),
        "output": OutputOptions(save_slices=True),
    }


def _centroid_x(intensity: np.ndarray, x_um: np.ndarray) -> float:
    return float(np.sum(intensity * x_um[:, None]) / np.sum(intensity))


def _delta_peak_x(delta_theta: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    peak_ix = np.argmax(np.max(delta_theta, axis=2), axis=1)
    return x_um[peak_ix]


def test_static_local_self_consistent_marches_theta_and_field_along_z(monkeypatch):
    parts = _request_parts()
    request = StaticRunRequest(
        **parts,
        solver=StaticSolverOptions(workflow=_workflow(), max_iterations=3),
    )
    grid = make_grid(request.grid)
    x_um = np.asarray(grid.x_um)
    calls: list[dict[str, np.ndarray | tuple[str, ...]]] = []
    original_advance = static_workflow.advance_slice_with_midintensity

    def observed_advance(A, theta, **kwargs):
        A_in = np.asarray(A).copy()
        result = original_advance(A, theta, **kwargs)
        calls.append(
            {
                "A_in": A_in,
                "A_out": np.asarray(result[0]).copy(),
                "I_mid": np.asarray(result[3]).copy(),
                "coherence_groups": tuple(kwargs["coherence_groups"]),
                "theta_weights": np.asarray(kwargs["theta_weights"]).copy(),
            }
        )
        return result

    monkeypatch.setattr(
        static_workflow,
        "advance_slice_with_midintensity",
        observed_advance,
    )
    result = static_workflow.run_static(request)

    theta = np.asarray(result.theta_final)
    theta_bias = np.asarray(result.theta_bias)
    delta_theta = theta - theta_bias[None, :, :]
    assert theta.shape == (grid.Nz, grid.Nx, grid.Ny)
    assert np.max(np.abs(np.diff(theta, axis=0))) > 1.0e-4
    assert not np.array_equal(theta[0], theta[-1])

    assert len(calls) == grid.Nz * request.solver.max_iterations
    accepted = [
        calls[(k + 1) * request.solver.max_iterations - 1]
        for k in range(grid.Nz)
    ]
    first_trial = [
        calls[k * request.solver.max_iterations]
        for k in range(grid.Nz)
    ]
    for k in range(grid.Nz - 1):
        assert np.array_equal(accepted[k]["A_out"], first_trial[k + 1]["A_in"])
    assert not np.array_equal(first_trial[1]["A_in"], first_trial[0]["A_in"])

    assert all(call["coherence_groups"] == ("A",) for call in calls)
    assert all(np.array_equal(call["theta_weights"], [1.0]) for call in calls)

    optical_x = np.asarray(
        [_centroid_x(call["I_mid"], x_um) for call in accepted]
    )
    static_peak_x = _delta_peak_x(delta_theta, x_um)
    assert np.mean(np.abs(static_peak_x - optical_x)) < 2.0
    assert np.max(np.abs(static_peak_x - optical_x)) < 3.0

    run_data = from_static_result(result)
    assert np.array_equal(run_data.fields["theta_stack"].data, theta)
    assert np.array_equal(run_data.fields["theta"].data, theta[-1])

    td = run_timedependent(
        TimeDependentRunRequest(
            **parts,
            solver=TimeDependentSolverOptions(
                workflow=_workflow(),
                Nt=3,
                dt=0.01,
                gamma_z=0.0,
            ),
        )
    )
    td_delta_theta = np.asarray(td.theta_final) - np.asarray(td.theta_bias)[None]
    td_peak_x = _delta_peak_x(td_delta_theta, x_um)
    correlation = float(
        np.corrcoef(delta_theta.ravel(), td_delta_theta.ravel())[0, 1]
    )
    assert correlation > 0.75
    assert np.mean(np.abs(static_peak_x - td_peak_x)) < 2.0
