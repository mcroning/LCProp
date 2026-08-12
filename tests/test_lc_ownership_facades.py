"""Compatibility tests for the LC-owned public facade."""

from __future__ import annotations

import importlib

import pytest

import lcprop.lc as lc
from lcprop.core import context as legacy_context
from lcprop.core import requests as legacy_requests
from lcprop.core import results as legacy_results
from lcprop.lc import persistence as lc_persistence
from lcprop.lc import diagnostics as lc_diagnostics
from lcprop.lc import operations as lc_operations
from lcprop.lc import products as lc_products
from lcprop.lc import requests as lc_requests
from lcprop.lc import results as lc_results
from lcprop.lc import specs as lc_specs
from lcprop.lc import static_torque_balance as lc_torque
from lcprop.lc import workflows as lc_workflows
from lcprop.lc.workflows import runtime as lc_runtime_workflow
from lcprop.lc.workflows import soliton as lc_soliton_workflow
from lcprop.lc.workflows import soliton_existence as lc_existence_workflow
from lcprop.lc.workflows import soliton_trans as lc_soliton_trans_workflow
from lcprop.lc.workflows import static as lc_static_workflow
from lcprop.lc.workflows import sweep as lc_sweep_workflow
from lcprop.lc.workflows import timedependent as lc_td_workflow
from lcprop.persistence import static as legacy_static_persistence
from lcprop.persistence import timedependent as legacy_td_persistence
from lcprop.products import data_model as legacy_products
from lcprop.products import diagnostics as legacy_diagnostics
from lcprop.products import static_torque_balance as legacy_torque
from lcprop.workflows import soliton as legacy_soliton
from lcprop.workflows import soliton_existence as legacy_existence
from lcprop.workflows import runtime as legacy_runtime_workflow
from lcprop.workflows import soliton_trans as legacy_soliton_trans_workflow
from lcprop.workflows import static as legacy_static_workflow
from lcprop.workflows import sweep as legacy_sweep
from lcprop.workflows import timedependent as legacy_td_workflow
from lcprop.workflows import (
    continue_static as legacy_continue_static,
    continue_timedependent as legacy_continue_timedependent,
    run_parameter_sweep as legacy_run_parameter_sweep,
    run_soliton as legacy_run_soliton,
    run_soliton_existence as legacy_run_soliton_existence,
    run_static as legacy_run_static,
    run_timedependent as legacy_run_timedependent,
    timedependent_state_from_static_result as legacy_td_state_from_static,
    validate_static_continuation as legacy_validate_static_continuation,
    validate_timedependent_continuation as legacy_validate_td_continuation,
)
from lcprop.workflows.soliton_trans import polish_soliton as legacy_polish_soliton


@pytest.mark.parametrize(
    ("facade", "legacy"),
    [
        (lc_specs.LCMaterial, legacy_context.LCMaterial),
        (lc_specs.BiasSpec, legacy_context.BiasSpec),
        (lc_specs.LCContext, legacy_context.LCContext),
        (lc_requests.StaticRunRequest, legacy_requests.StaticRunRequest),
        (lc_requests.RuntimeOptions, legacy_requests.RuntimeOptions),
        (lc_requests.OutputOptions, legacy_requests.OutputOptions),
        (lc_requests.StaticSolverOptions, legacy_requests.StaticSolverOptions),
        (lc_requests.StaticWorkflowOptions, legacy_requests.StaticWorkflowOptions),
        (
            lc_requests.TimeDependentRunRequest,
            legacy_requests.TimeDependentRunRequest,
        ),
        (
            lc_requests.TimeDependentSolverOptions,
            legacy_requests.TimeDependentSolverOptions,
        ),
        (lc_requests.SolitonRequest, legacy_soliton.SolitonRequest),
        (
            lc_requests.SolitonExistenceRequest,
            legacy_existence.SolitonExistenceRequest,
        ),
        (lc_requests.ParameterSweepRequest, legacy_sweep.ParameterSweepRequest),
        (lc_results.StaticIterationRecord, legacy_results.StaticIterationRecord),
        (lc_results.StaticSliceSummary, legacy_results.StaticSliceSummary),
        (lc_results.StaticRunResult, legacy_results.StaticRunResult),
        (
            lc_results.TimeDependentRunResult,
            legacy_results.TimeDependentRunResult,
        ),
        (lc_results.SolitonResult, legacy_soliton.SolitonResult),
        (
            lc_results.SolitonExistenceResult,
            legacy_existence.SolitonExistenceResult,
        ),
        (lc_results.ParameterSweepResult, legacy_sweep.ParameterSweepResult),
        (lc_results.SolitonSweepMember, legacy_sweep.SolitonSweepMember),
    ],
)
def test_lc_owned_type_facades_preserve_identity(facade, legacy):
    assert facade is legacy


@pytest.mark.parametrize(
    ("facade", "legacy"),
    [
        (lc_workflows.run_static, legacy_run_static),
        (lc_workflows.continue_static, legacy_continue_static),
        (
            lc_workflows.validate_static_continuation,
            legacy_validate_static_continuation,
        ),
        (lc_workflows.run_timedependent, legacy_run_timedependent),
        (lc_workflows.continue_timedependent, legacy_continue_timedependent),
        (
            lc_workflows.validate_timedependent_continuation,
            legacy_validate_td_continuation,
        ),
        (
            lc_workflows.timedependent_state_from_static_result,
            legacy_td_state_from_static,
        ),
        (lc_workflows.run_soliton, legacy_run_soliton),
        (lc_workflows.polish_soliton, legacy_polish_soliton),
        (lc_workflows.run_soliton_existence, legacy_run_soliton_existence),
        (lc_workflows.run_parameter_sweep, legacy_run_parameter_sweep),
        (lc_products.from_static_result, legacy_products.from_static_result),
        (
            lc_products.from_timedependent_result,
            legacy_products.from_timedependent_result,
        ),
        (lc_products.from_soliton_result, legacy_products.from_soliton_result),
        (
            lc_products.from_soliton_existence_result,
            legacy_products.from_soliton_existence_result,
        ),
        (
            lc_products.from_parameter_sweep_result,
            legacy_products.from_parameter_sweep_result,
        ),
        (lc_products.to_run_data, legacy_products.to_run_data),
        (lc_products.theta_metrics, legacy_diagnostics.theta_metrics),
        (lc_diagnostics.theta_metrics, legacy_diagnostics.theta_metrics),
        (
            lc_products.theta_update_metrics,
            legacy_diagnostics.theta_update_metrics,
        ),
        (
            lc_products.residual_theta_static,
            legacy_diagnostics.residual_theta_static,
        ),
        (
            lc_products.StaticTorqueBalanceData,
            legacy_torque.StaticTorqueBalanceData,
        ),
        (
            lc_torque.StaticTorqueBalanceData,
            legacy_torque.StaticTorqueBalanceData,
        ),
        (
            lc_products.build_static_torque_balance_data,
            legacy_torque.build_static_torque_balance_data,
        ),
        (
            lc_products.plot_static_torque_balance,
            legacy_torque.plot_static_torque_balance,
        ),
    ],
)
def test_lc_owned_callable_and_product_facades_preserve_identity(facade, legacy):
    assert facade is legacy


@pytest.mark.parametrize(
    ("facade", "legacy"),
    [
        (
            lc_persistence.StaticCheckpoint,
            legacy_static_persistence.StaticCheckpoint,
        ),
        (
            lc_persistence.static_request_fingerprint,
            legacy_static_persistence.static_request_fingerprint,
        ),
        (
            lc_persistence.validate_static_checkpoint,
            legacy_static_persistence.validate_static_checkpoint,
        ),
        (
            lc_persistence.save_static_checkpoint,
            legacy_static_persistence.save_static_checkpoint,
        ),
        (
            lc_persistence.load_static_checkpoint,
            legacy_static_persistence.load_static_checkpoint,
        ),
        (
            lc_persistence.TimeDependentCheckpoint,
            legacy_td_persistence.TimeDependentCheckpoint,
        ),
        (
            lc_persistence.save_timedependent_checkpoint,
            legacy_td_persistence.save_timedependent_checkpoint,
        ),
        (
            lc_persistence.load_timedependent_checkpoint,
            legacy_td_persistence.load_timedependent_checkpoint,
        ),
    ],
)
def test_lc_checkpoint_facade_preserves_identity(facade, legacy):
    assert facade is legacy


def test_top_level_lc_exports_are_the_facade_objects():
    for module in (
        lc_specs,
        lc_requests,
        lc_results,
        lc_workflows,
        lc_products,
        lc_persistence,
    ):
        for name in module.__all__:
            assert getattr(lc, name) is getattr(module, name)

    assert lc.LC_STATIC_OPERATION is lc_operations.LC_STATIC_OPERATION
    assert lc.LC_TIMEDEPENDENT_OPERATION is lc_operations.LC_TIMEDEPENDENT_OPERATION


def test_lc_facade_does_not_claim_shared_platform_types():
    for name in ("BackendSpec", "BeamStack", "GridSpec", "RunData"):
        assert name not in lc.__all__
        assert not hasattr(lc, name)


def test_relocated_lc_types_report_lc_as_their_canonical_module():
    for value in (
        lc_specs.LCMaterial,
        lc_specs.BiasSpec,
        lc_specs.LCContext,
        lc_requests.RuntimeOptions,
        lc_requests.OutputOptions,
        lc_requests.StaticRunRequest,
        lc_requests.TimeDependentRunRequest,
        lc_requests.SolitonRequest,
        lc_requests.SolitonExistenceRequest,
        lc_requests.ParameterSweepRequest,
        lc_results.StaticRunResult,
        lc_results.TimeDependentRunResult,
        lc_results.SolitonResult,
        lc_results.SolitonExistenceResult,
        lc_results.ParameterSweepResult,
    ):
        assert value.__module__ in {
            "lcprop.lc.specs",
            "lcprop.lc.requests",
            "lcprop.lc.results",
        }


@pytest.mark.parametrize(
    ("canonical", "legacy"),
    [
        (lc_runtime_workflow, legacy_runtime_workflow),
        (lc_static_workflow, legacy_static_workflow),
        (lc_td_workflow, legacy_td_workflow),
        (lc_soliton_workflow, legacy_soliton),
        (lc_soliton_trans_workflow, legacy_soliton_trans_workflow),
        (lc_existence_workflow, legacy_existence),
        (lc_sweep_workflow, legacy_sweep),
    ],
)
def test_legacy_workflow_modules_alias_canonical_lc_modules(canonical, legacy):
    assert legacy is canonical


def test_lc_workflow_entry_points_report_canonical_modules():
    expected = {
        lc_workflows.run_static: "lcprop.lc.workflows.static",
        lc_workflows.continue_static: "lcprop.lc.workflows.static",
        lc_workflows.run_timedependent: "lcprop.lc.workflows.timedependent",
        lc_workflows.continue_timedependent: "lcprop.lc.workflows.timedependent",
        lc_workflows.run_soliton: "lcprop.lc.workflows.soliton",
        lc_workflows.polish_soliton: "lcprop.lc.workflows.soliton_trans",
        lc_workflows.run_soliton_existence: (
            "lcprop.lc.workflows.soliton_existence"
        ),
        lc_workflows.run_parameter_sweep: "lcprop.lc.workflows.sweep",
    }
    for entry_point, module_name in expected.items():
        assert entry_point.__module__ == module_name


def test_lc_workflow_operation_ids_are_unchanged():
    assert lc_operations.LC_STATIC_OPERATION.material_id == "lc"
    assert lc_operations.LC_STATIC_OPERATION.workflow_id == "static"
    assert lc_operations.LC_TIMEDEPENDENT_OPERATION.material_id == "lc"
    assert lc_operations.LC_TIMEDEPENDENT_OPERATION.workflow_id == "timedependent"


def test_product_ownership_is_split_between_shared_records_and_lc_adapters():
    assert legacy_products.RunData.__module__ == "lcprop.products.data_model"
    assert legacy_products.FieldData.__module__ == "lcprop.products.data_model"
    assert lc_products.from_static_result.__module__ == "lcprop.lc.products"
    assert lc_products.to_run_data.__module__ == "lcprop.lc.products"
    assert lc_diagnostics.theta_metrics.__module__ == "lcprop.lc.diagnostics"
    assert lc_torque.StaticTorqueBalanceData.__module__ == (
        "lcprop.lc.static_torque_balance"
    )


def test_persistence_and_lc_facades_support_reverse_import_order():
    persistence = importlib.import_module("lcprop.persistence")
    reloaded_lc = importlib.import_module("lcprop.lc")

    assert persistence.StaticCheckpoint is reloaded_lc.StaticCheckpoint
    assert persistence.TimeDependentCheckpoint is reloaded_lc.TimeDependentCheckpoint


def test_pr_public_ownership_remains_importable_and_distinct():
    pr = importlib.import_module("lcprop.pr")

    assert pr.PR_MATERIAL_ID == "pr"
    assert lc.LC_MATERIAL_ID == "lc"
    assert not hasattr(lc, "PRMaterialSpec")
