from __future__ import annotations

import numpy as np
import pytest

from lcprop.algorithms.theta_cn import (
    static_director_residual,
    static_director_residual_metrics,
)
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.products.data_model import from_static_result
from lcprop.workflows.static import run_static


def _workflow() -> StaticWorkflowOptions:
    return StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )


def _request(**solver_updates) -> StaticRunRequest:
    solver = StaticSolverOptions(
        workflow=_workflow(),
        max_iterations=2,
        **solver_updates,
    )
    return StaticRunRequest(
        grid=GridSpec(
            Nx=16,
            Ny=12,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=15.0,
        ),
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    power_mW=1.0,
                    waist_x_um=6.0,
                    waist_y_um=6.0,
                    x0_um=-10.0,
                    coherence_group="A",
                ),
            )
        ),
        solver=solver,
        output=OutputOptions(),
    )


def test_static_residual_zero_solution_and_perturbation():
    theta = np.zeros((18, 14), dtype=np.float64)
    intensity = np.ones_like(theta)

    zero = static_director_residual_metrics(
        theta,
        intensity,
        b=2.0,
        bi=3.0,
        dx=0.1,
        dy=0.15,
    )
    assert zero == {"residual_rms": 0.0, "residual_max": 0.0}

    x = np.linspace(0.0, np.pi, theta.shape[0])[:, None]
    y = np.arange(theta.shape[1])[None, :]
    perturbed = 1.0e-3 * np.sin(x) * np.cos(2.0 * np.pi * y / theta.shape[1])
    residual = static_director_residual(
        perturbed,
        intensity,
        b=2.0,
        bi=3.0,
        dx=0.1,
        dy=0.15,
    )
    metrics = static_director_residual_metrics(
        perturbed,
        intensity,
        b=2.0,
        bi=3.0,
        dx=0.1,
        dy=0.15,
    )

    assert np.all(residual[[0, -1]] == 0.0)
    assert metrics["residual_rms"] > zero["residual_rms"]
    assert metrics["residual_max"] > zero["residual_max"]


def test_legacy_static_tolerances_map_only_to_update_criteria():
    options = StaticSolverOptions(
        tolerance_rms=1.0e-4,
        tolerance_max=2.0e-4,
        max_iterations=7,
    )

    assert options.resolved_delta_theta_rms_tol == 1.0e-4
    assert options.resolved_delta_theta_max_tol == 2.0e-4
    assert options.resolved_static_max_iterations == 7
    assert options.static_residual_rms_tol is None
    assert options.static_residual_max_tol is None


def test_static_iteration_records_are_ordered_and_summarized():
    result = run_static(_request())

    assert len(result.slice_summaries) == 3
    assert [item.z_index for item in result.slice_summaries] == [0, 1, 2]
    assert result.iteration_records
    ordering = [
        (item.z_index, item.optical_pass, item.relax_iteration)
        for item in result.iteration_records
    ]
    assert ordering == sorted(ordering)

    for z_index, summary in enumerate(result.slice_summaries):
        records = [item for item in result.iteration_records if item.z_index == z_index]
        assert summary.relaxation_iterations == len(records)
        assert summary.optical_passes == 2
        assert summary.converged is False
        assert summary.termination_reason == "maximum_iterations"


def test_static_residual_tolerance_is_explicit_and_can_converge():
    result = run_static(
        _request(
            static_residual_rms_tol=1.0e9,
            static_residual_max_tol=1.0e9,
        )
    )

    assert result.all_slices_converged is True
    assert all(item.converged for item in result.slice_summaries)
    assert all(
        item.termination_reason == "residual_tolerance"
        for item in result.slice_summaries
    )
    assert all(item.optical_passes == 1 for item in result.slice_summaries)


def test_static_update_tolerance_is_distinct_from_residual_tolerance():
    result = run_static(
        _request(
            static_delta_theta_rms_tol=1.0e9,
            static_delta_theta_max_tol=1.0e9,
        )
    )

    assert result.all_slices_converged is True
    assert all(
        item.termination_reason == "update_tolerance"
        for item in result.slice_summaries
    )


def test_static_maximum_iterations_above_tolerance_is_not_converged():
    result = run_static(
        _request(
            static_max_iterations=1,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        )
    )

    assert result.all_slices_converged is False
    assert all(not item.converged for item in result.slice_summaries)
    assert all(
        item.termination_reason == "maximum_iterations"
        for item in result.slice_summaries
    )


def test_static_aggregate_residuals_match_slice_records():
    result = run_static(_request())
    rms = np.asarray([item.final_residual_rms for item in result.slice_summaries])
    maximum = np.asarray([item.final_residual_max for item in result.slice_summaries])

    assert result.max_final_residual_rms == pytest.approx(np.max(rms))
    assert result.median_final_residual_rms == pytest.approx(np.median(rms))
    assert result.rms_over_z_final_residual == pytest.approx(
        np.sqrt(np.mean(rms * rms))
    )
    assert result.max_final_residual_max == pytest.approx(np.max(maximum))
    assert result.worst_slice_index == int(np.argmax(rms))


def test_static_history_disabled_keeps_slice_summaries():
    result = run_static(_request(record_iteration_history=False))
    data = from_static_result(result)

    assert result.iteration_records == ()
    assert len(result.slice_summaries) == 3
    assert all(item.relaxation_iterations > 0 for item in result.slice_summaries)
    assert "static_convergence" in data.diagnostics
    assert "static_iteration_history" not in data.diagnostics


def test_static_results_diagnostics_and_curves_match_summaries():
    result = run_static(_request())
    data = from_static_result(result)
    rows = data.diagnostics["static_convergence"].values["rows"]

    assert len(rows) == len(result.slice_summaries)
    assert np.allclose(
        [row["final_residual_rms"] for row in rows],
        [item.final_residual_rms for item in result.slice_summaries],
    )
    assert [row["termination_reason"] for row in rows] == [
        item.termination_reason for item in result.slice_summaries
    ]
    assert "static_iteration_history" in data.diagnostics

    assert list(data.curves.keys()) == [
        "static_final_residual_rms",
        "static_final_residual_max",
        "static_relaxation_iterations",
        "static_theta_max",
    ]
    assert np.allclose(
        data.curves["static_final_residual_rms"].y,
        [item.final_residual_rms for item in result.slice_summaries],
    )
    assert data.curves["static_final_residual_rms"].y_scale == "log"
    assert np.array_equal(
        data.curves["static_relaxation_iterations"].y,
        [item.relaxation_iterations for item in result.slice_summaries],
    )


def test_static_residual_cupy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        theta = cp.zeros((8, 6), dtype=cp.float64)
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        pytest.skip(f"CuPy runtime unavailable: {exc}")
    intensity = cp.ones_like(theta)

    metrics = static_director_residual_metrics(
        theta,
        intensity,
        b=1.0,
        bi=2.0,
        dx=0.2,
        dy=0.3,
        xp=cp,
    )
    assert metrics == {"residual_rms": 0.0, "residual_max": 0.0}
