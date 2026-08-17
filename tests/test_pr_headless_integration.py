from __future__ import annotations

import json

import numpy as np

from lcprop.core.execution import CancellationToken
from lcprop.lc.operations import LC_MATERIAL_ID, LC_STATIC_OPERATION
from lcprop.persistence import load_run_checkpoint, save_run_checkpoint
from lcprop.persistence.static import StaticCheckpoint
from lcprop.pr.checkpoint import PRTimeDependentCheckpoint
from lcprop.pr.operations import (
    PR_MATERIAL_ID,
    PR_STATIC_OPERATION,
    PR_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.specs import PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.workflow import continue_pr_timedependent
from lcprop.runners.local import LocalRunner
from tests.test_all_workflows import make_base_static_request
from tests.test_pr_execution import _assert_same_physical_result, _request


def _assert_same_pr_run_data(actual, expected) -> None:
    assert actual.workflow == expected.workflow
    assert actual.longitudinal_enabled == expected.longitudinal_enabled
    assert actual.longitudinal_message == expected.longitudinal_message
    assert actual.geometry.units == expected.geometry.units
    for axis in ("x", "y", "z"):
        np.testing.assert_array_equal(
            actual.geometry.coord(axis),
            expected.geometry.coord(axis),
        )

    assert tuple(actual.fields.keys()) == tuple(expected.fields.keys())
    for key in actual.fields:
        actual_field = actual.fields[key]
        expected_field = expected.fields[key]
        assert actual_field.key == expected_field.key
        assert actual_field.display_name == expected_field.display_name
        assert actual_field.axes == expected_field.axes
        assert actual_field.kind == expected_field.kind
        assert actual_field.units == expected_field.units
        assert actual_field.default_view == expected_field.default_view
        assert actual_field.quantity == expected_field.quantity
        assert actual_field.value_unit == expected_field.value_unit
        assert actual_field.colormap == expected_field.colormap
        assert actual_field.source_volume_key == expected_field.source_volume_key
        np.testing.assert_array_equal(actual_field.data, expected_field.data)

    assert tuple(actual.curves.keys()) == tuple(expected.curves.keys())
    assert tuple(actual.diagnostics.keys()) == tuple(expected.diagnostics.keys())
    for key in actual.diagnostics:
        actual_diagnostic = actual.diagnostics[key]
        expected_diagnostic = expected.diagnostics[key]
        assert actual_diagnostic.key == expected_diagnostic.key
        assert actual_diagnostic.display_name == expected_diagnostic.display_name
        assert actual_diagnostic.values == expected_diagnostic.values


def test_registered_pr_stop_disk_resume_and_products_match_uninterrupted(
    tmp_path,
):
    runner = LocalRunner(
        operations=(
            LC_STATIC_OPERATION,
            PR_TIMEDEPENDENT_OPERATION,
            PR_STATIC_OPERATION,
        )
    )
    assert tuple(operation.key for operation in runner.registered_operations) == (
        (LC_MATERIAL_ID, "static"),
        (PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW),
        (PR_MATERIAL_ID, PR_STATIC_WORKFLOW),
    )

    request = _request(steps=4)
    token = CancellationToken()
    progress = []

    def stop_after_first(item) -> None:
        progress.append(item)
        token.cancel()

    stopped = runner.run_registered(
        PR_MATERIAL_ID,
        PR_TIMEDEPENDENT_WORKFLOW,
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )

    assert stopped.kind == PR_TIMEDEPENDENT_WORKFLOW
    assert stopped.material_id == PR_MATERIAL_ID
    assert stopped.message == "Cancelled locally"
    assert stopped.result.status == "cancelled"
    assert stopped.result.completed_steps == 1
    assert stopped.result.requested_steps == 4
    assert stopped.result.time_normalized == request.solver.dt_normalized
    assert isinstance(stopped.result.checkpoint, PRTimeDependentCheckpoint)
    assert stopped.run_data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    assert len(progress) == 1
    assert progress[0].workflow == PR_TIMEDEPENDENT_WORKFLOW

    pr_directory = tmp_path / "pr"
    save_run_checkpoint(stopped.result.checkpoint, pr_directory)
    provenance = json.loads(
        (pr_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["material"] == PR_MATERIAL_ID
    assert provenance["workflow"] == PR_TIMEDEPENDENT_WORKFLOW

    loaded = load_run_checkpoint(pr_directory)
    assert isinstance(loaded, PRTimeDependentCheckpoint)
    assert loaded.completed_steps == 1
    assert loaded.time_normalized == request.solver.dt_normalized

    resumed = continue_pr_timedependent(
        request,
        loaded,
        additional_steps=3,
    )
    resumed_data = PR_TIMEDEPENDENT_OPERATION.to_run_data(resumed)
    uninterrupted = runner.run_registered(
        PR_MATERIAL_ID,
        PR_TIMEDEPENDENT_WORKFLOW,
        request,
    )

    _assert_same_physical_result(resumed, uninterrupted.result)
    assert resumed.status == uninterrupted.result.status == "completed"
    assert resumed.completed_steps == uninterrupted.result.completed_steps == 4
    assert resumed.requested_steps == uninterrupted.result.requested_steps == 4
    assert resumed.time_normalized == uninterrupted.result.time_normalized
    assert resumed.diagnostics == uninterrupted.result.diagnostics
    _assert_same_pr_run_data(resumed_data, uninterrupted.run_data)

    static_request = PRStaticRunRequest(
        grid=request.grid,
        beams=request.beams,
        material=request.material,
        solver=PRStaticWorkflowOptions(),
        backend=request.backend,
        initial_A=request.initial_A,
    )
    static = runner.run_registered(
        PR_MATERIAL_ID,
        PR_STATIC_WORKFLOW,
        static_request,
    )
    assert static.kind == PR_STATIC_WORKFLOW
    assert static.material_id == PR_MATERIAL_ID
    assert static.result.status == "converged"
    assert static.run_data.workflow == PR_STATIC_WORKFLOW
    assert static.run_data.diagnostics["summary"].values["replay"][
        "field_consistent"
    ]

    lc_result = runner.run_registered(
        LC_MATERIAL_ID,
        "static",
        make_base_static_request(),
    )
    assert lc_result.material_id == LC_MATERIAL_ID
    assert lc_result.run_data.workflow == "static"
    lc_directory = tmp_path / "lc"
    save_run_checkpoint(lc_result.result.checkpoint, lc_directory)
    loaded_lc = load_run_checkpoint(lc_directory)
    assert isinstance(loaded_lc, StaticCheckpoint)
    np.testing.assert_array_equal(
        loaded_lc.A_next,
        lc_result.result.checkpoint.A_next,
    )
    lc_provenance = json.loads(
        (lc_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert lc_provenance["workflow"] == "static"
    assert "material" not in lc_provenance
