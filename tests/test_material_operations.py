import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.lc.operations import (
    LC_CONTINUE_STATIC_OPERATION,
    LC_CONTINUE_TIMEDEPENDENT_OPERATION,
    LC_MATERIAL_ID,
    LC_PARAMETER_SWEEP_OPERATION,
    LC_SOLITON_EXISTENCE_OPERATION,
    LC_SOLITON_OPERATION,
    LC_STATIC_OPERATION,
    LC_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.operations import PR_MATERIAL_ID, PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.runners.base import WorkflowOperation
from lcprop.runners.local import LocalRunner
from tests.test_all_workflows import make_base_static_request


def _pr_request(*, steps: int = 2) -> PRRunRequest:
    grid = GridSpec(
        Nx=8,
        Ny=6,
        x_aperture_um=80.0,
        y_aperture_um=60.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    return PRRunRequest(
        grid=grid,
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=20.0,
                    waist_y_um=20.0,
                    coherence_group="pr-operation",
                ),
            ),
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            applied_field=0.5,
            gain_length_product=0.1,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(
            Nt=steps,
            dt_normalized=0.01,
            optical_substeps=1,
        ),
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
        initial_A=np.ones((1, grid.Nx, grid.Ny), dtype=np.complex128),
    )


def test_workflow_operation_is_a_small_validated_callable_descriptor():
    operation = WorkflowOperation(
        material_id="material",
        workflow_id="workflow",
        run=lambda request: request,
        to_run_data=lambda result: result,
    )

    assert operation.key == ("material", "workflow")
    with pytest.raises(ValueError, match="material_id"):
        WorkflowOperation("", "workflow", lambda x: x, lambda x: x)
    with pytest.raises(ValueError, match="workflow_id"):
        WorkflowOperation("material", " workflow ", lambda x: x, lambda x: x)
    with pytest.raises(TypeError, match="run must be callable"):
        WorkflowOperation("material", "workflow", None, lambda x: x)
    with pytest.raises(TypeError, match="to_run_data must be callable"):
        WorkflowOperation("material", "workflow", lambda x: x, None)


def test_local_runner_registration_is_explicit_and_rejects_duplicate_keys():
    empty = LocalRunner()
    assert empty.registered_operations == ()
    with pytest.raises(KeyError, match="no operation registered"):
        empty.run_registered("lc", "static", make_base_static_request())

    runner = LocalRunner(operations=(LC_STATIC_OPERATION,))
    assert runner.registered_operations == (LC_STATIC_OPERATION,)
    with pytest.raises(ValueError, match="already registered"):
        runner.register_operation(LC_STATIC_OPERATION)


def test_shared_execution_does_not_mask_workflow_or_product_errors():
    def failed_workflow(_request):
        raise RuntimeError("workflow failed")

    workflow_failure = WorkflowOperation(
        "material",
        "failed_workflow",
        failed_workflow,
        lambda result: result,
    )
    with pytest.raises(RuntimeError, match="workflow failed"):
        LocalRunner().run_operation(workflow_failure, object())

    def failed_adapter(_result):
        raise ValueError("product conversion failed")

    adapter_failure = WorkflowOperation(
        "material",
        "failed_adapter",
        lambda _request: object(),
        failed_adapter,
    )
    with pytest.raises(ValueError, match="product conversion failed"):
        LocalRunner().run_operation(adapter_failure, object())


def test_lc_static_operation_uses_shared_execution_without_changing_legacy_method():
    request = make_base_static_request()
    runner = LocalRunner(operations=(LC_STATIC_OPERATION,))

    shared = runner.run_registered(LC_MATERIAL_ID, "static", request)
    legacy = LocalRunner().run_static(request)

    assert shared.kind == "static"
    assert shared.material_id == LC_MATERIAL_ID
    assert shared.message == "Completed locally"
    assert shared.run_data.workflow == "static"
    np.testing.assert_array_equal(shared.result.A_final, legacy.result.A_final)
    np.testing.assert_array_equal(
        shared.result.theta_final,
        legacy.result.theta_final,
    )
    assert legacy.kind == "static"
    assert legacy.run_data is None
    assert legacy.material_id is None


def test_pr_operation_uses_same_execution_path_with_progress_and_cancellation():
    runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
    token = CancellationToken()
    progress = []

    def stop_after_first(item) -> None:
        progress.append(item)
        token.cancel()

    shared = runner.run_registered(
        PR_MATERIAL_ID,
        PR_TIMEDEPENDENT_WORKFLOW,
        _pr_request(),
        cancellation_token=token,
        progress_callback=stop_after_first,
    )

    assert shared.kind == PR_TIMEDEPENDENT_WORKFLOW
    assert shared.material_id == PR_MATERIAL_ID
    assert shared.message == "Cancelled locally"
    assert shared.result.status == "cancelled"
    assert shared.result.completed_steps == 1
    assert shared.result.requested_steps == 2
    assert len(progress) == 1
    assert progress[0].workflow == PR_TIMEDEPENDENT_WORKFLOW
    assert shared.run_data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    summary = shared.run_data.diagnostics["summary"].values
    assert summary["status"] == "cancelled"
    assert summary["completed_material_steps"] == 1


def test_material_packages_expose_operations_without_a_global_registry():
    assert LC_STATIC_OPERATION.key == (LC_MATERIAL_ID, "static")
    assert LC_TIMEDEPENDENT_OPERATION.key == (
        LC_MATERIAL_ID,
        "timedependent",
    )
    assert PR_TIMEDEPENDENT_OPERATION.key == (
        PR_MATERIAL_ID,
        PR_TIMEDEPENDENT_WORKFLOW,
    )


def test_shared_local_runner_import_does_not_import_material_packages():
    script = """
import sys
import lcprop.runners.local
assert 'lcprop.lc' not in sys.modules
assert 'lcprop.pr' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_material_operation_callbacks_have_material_owned_provenance():
    for operation in (
        LC_STATIC_OPERATION,
        LC_TIMEDEPENDENT_OPERATION,
        LC_SOLITON_OPERATION,
        LC_SOLITON_EXISTENCE_OPERATION,
        LC_PARAMETER_SWEEP_OPERATION,
        LC_CONTINUE_STATIC_OPERATION,
        LC_CONTINUE_TIMEDEPENDENT_OPERATION,
    ):
        assert operation.material_id == LC_MATERIAL_ID
        assert operation.run.__module__.startswith("lcprop.lc.")
        assert operation.to_run_data.__module__ == "lcprop.lc.products"

    assert PR_TIMEDEPENDENT_OPERATION.run.__module__.startswith("lcprop.pr.")
    assert PR_TIMEDEPENDENT_OPERATION.to_run_data.__module__ == (
        "lcprop.pr.products"
    )


def test_lc_compatibility_method_delegates_through_operation_execution(
    monkeypatch,
):
    runner = LocalRunner()
    observed = []
    original = runner.run_operation

    def capture(operation, request, *args, **kwargs):
        observed.append((operation, kwargs["_prepare_products"]))
        return original(operation, request, *args, **kwargs)

    monkeypatch.setattr(runner, "run_operation", capture)
    legacy = runner.run_static(make_base_static_request())

    assert observed == [(LC_STATIC_OPERATION, False)]
    assert legacy.kind == "static"
    assert legacy.message == "Completed locally"
    assert legacy.run_data is None
    assert legacy.material_id is None


@pytest.mark.parametrize(
    ("method_name", "operation", "args", "kwargs"),
    [
        ("run_static", LC_STATIC_OPERATION, (), {}),
        ("continue_static", LC_CONTINUE_STATIC_OPERATION, (object(),), {}),
        ("run_timedependent", LC_TIMEDEPENDENT_OPERATION, (), {}),
        (
            "continue_timedependent",
            LC_CONTINUE_TIMEDEPENDENT_OPERATION,
            (object(), 3),
            {},
        ),
        ("run_soliton", LC_SOLITON_OPERATION, (), {}),
        ("run_soliton_existence", LC_SOLITON_EXISTENCE_OPERATION, (), {}),
        ("run_parameter_sweep", LC_PARAMETER_SWEEP_OPERATION, (), {}),
    ],
)
def test_all_lc_execution_compatibility_methods_delegate_to_canonical_operation(
    monkeypatch,
    method_name,
    operation,
    args,
    kwargs,
):
    runner = LocalRunner()
    request = SimpleNamespace(refine_transverse=False)
    workflow_result = SimpleNamespace(status="completed")
    observed = []

    def capture(selected, selected_request, *selected_args, **selected_kwargs):
        observed.append(
            (
                selected,
                selected_request,
                selected_args,
                selected_kwargs,
            )
        )
        return SimpleNamespace(
            kind=selected.workflow_id,
            result=workflow_result,
            message="Completed locally",
        )

    monkeypatch.setattr(runner, "run_operation", capture)
    result = getattr(runner, method_name)(request, *args, **kwargs)

    assert observed == [
        (
            operation,
            request,
            args,
            {"_prepare_products": False},
        )
    ]
    assert result.kind == operation.workflow_id
    assert result.result is workflow_result
    assert result.run_data is None
    assert result.material_id is None
