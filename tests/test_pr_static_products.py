from dataclasses import replace

import numpy as np
import pytest

from lcprop.pr.products import pr_static_result_to_run_data
from lcprop.pr.static_workflow import (
    PRCoupledStaticIterationRecord,
    PRCoupledStaticSliceSummary,
    PRStaticRunResult,
)


def _synthetic_static_result(*, completed_slices: int = 2, status="converged"):
    nx, ny, nz = 4, 3, 2
    A_initial = np.ones((1, nx, ny), dtype=np.complex128)
    A_final = np.full((1, nx, ny), 0.5j, dtype=np.complex128)
    shape = (completed_slices, nx, ny)
    E_initial = np.zeros(shape, dtype=float)
    E_final = np.full(shape, 0.25, dtype=float)
    source = np.full(shape, 1.3, dtype=float)
    residual = np.full(shape, 1e-10, dtype=float)
    summaries = tuple(
        PRCoupledStaticSliceSummary(
            z_index=index,
            z_um=5.0 * index,
            coupled_passes=2,
            final_residual_rms=1e-10,
            final_residual_max=2e-10,
            final_delta_E_rms=3e-9,
            final_delta_E_max=4e-9,
            converged=True,
            termination_reason="residual_tolerance",
        )
        for index in range(completed_slices)
    )
    records = tuple(
        PRCoupledStaticIterationRecord(
            z_index=index,
            coupled_pass=1,
            residual_before_rms=1e-4,
            residual_before_max=2e-4,
            residual_after_rms=1e-10,
            residual_after_max=2e-10,
            delta_E_rms=3e-9,
            delta_E_max=4e-9,
            step_scale=1.0,
            material_status="converged",
            accepted=True,
        )
        for index in range(completed_slices)
    )
    return PRStaticRunResult(
        A_initial=A_initial,
        A_final=A_final,
        E_initial=E_initial,
        E_final=E_final,
        source_intensity_stack=source,
        residual_stack=residual,
        power_initial=1.0,
        power_final=1.0 - 2e-15,
        converged=status == "converged",
        completed_slices=completed_slices,
        iteration_records=records,
        slice_summaries=summaries,
        grid_summary={
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx_um": 2.0,
            "dy_um": 4.0,
            "dz_um": 5.0,
        },
        launch_summary={
            "Nch": 1,
            "coherence": "incoherent",
            "coherence_groups": ["static"],
        },
        backend_summary={"backend": "numpy", "precision": "float64"},
        tolerance_provenance={
            "precision": "float64",
            "material_solver": {"source": "precision_default"},
        },
        replay_diagnostics={
            "performed_slices": completed_slices,
            "requested_slices": nz,
            "field_consistent": True,
            "source_consistent": True,
            "residual_consistent": True,
            "residual_converged": status == "converged",
        },
        status=status,
    )


def test_static_product_preserves_fields_convergence_and_replay_evidence():
    result = _synthetic_static_result()

    data = pr_static_result_to_run_data(result)

    assert data.workflow == "pr_static"
    np.testing.assert_array_equal(data.geometry.x, [-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_array_equal(data.geometry.y, [-4.0, 0.0, 4.0])
    np.testing.assert_array_equal(data.geometry.z, [0.0, 5.0])
    assert tuple(data.fields.keys()) == (
        "input_intensity",
        "output_intensity",
        "initial_E",
        "final_E",
        "initial_E_stack",
        "final_E_stack",
        "pr_driving_intensity_stack",
        "pr_static_residual_stack",
    )
    np.testing.assert_array_equal(
        data.fields["final_E_stack"].data,
        result.E_final,
    )
    np.testing.assert_array_equal(
        data.fields["pr_static_residual_stack"].data,
        result.residual_stack,
    )
    assert tuple(data.curves.keys()) == (
        "static_final_residual_rms",
        "static_final_residual_max",
        "static_coupled_passes",
        "static_delta_E_max",
    )

    summary = data.diagnostics["summary"].values
    assert summary["status"] == "converged"
    assert summary["converged"] is True
    assert summary["completed_slices"] == 2
    assert summary["total_slices"] == 2
    assert summary["z_reached_um"] == 10.0
    assert summary["resolved_tolerances"] == result.tolerance_provenance
    assert summary["replay"] == result.replay_diagnostics
    rows = data.diagnostics["static_convergence"].values["rows"]
    assert len(rows) == 2
    assert rows[0]["termination_reason"] == "residual_tolerance"
    assert len(
        data.diagnostics["static_iteration_history"].values["rows"]
    ) == 2


def test_cancelled_static_product_uses_only_completed_prefix():
    result = _synthetic_static_result(
        completed_slices=1,
        status="cancelled",
    )

    data = pr_static_result_to_run_data(result)

    np.testing.assert_array_equal(data.geometry.z, [0.0])
    assert data.fields["final_E_stack"].data.shape == (1, 4, 3)
    assert data.fields["output_intensity"].display_name == (
        "Intensity at Current z"
    )
    summary = data.diagnostics["summary"].values
    assert summary["status"] == "cancelled"
    assert summary["converged"] is False
    assert summary["completed_slices"] == 1
    assert summary["total_slices"] == 2
    assert summary["z_reached_um"] == 5.0


def test_static_product_rejects_wrong_type_and_inconsistent_prefix_shape():
    with pytest.raises(TypeError, match="PRStaticRunResult"):
        pr_static_result_to_run_data(object())

    malformed = replace(
        _synthetic_static_result(),
        residual_stack=np.zeros((1, 4, 3)),
    )
    with pytest.raises(ValueError, match="residual_stack"):
        pr_static_result_to_run_data(malformed)
