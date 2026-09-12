from __future__ import annotations

import inspect
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

import lcprop.pr.transverse.static_workflow as transverse_static_module
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.image_amplification import image_amplification_base_capabilities
from lcprop.pr.gui.request_adapter import (
    validate_pr_transverse_static_gui_request,
)
from lcprop.pr.operations import (
    PR_IMAGE_AMPLIFICATION_OPERATION,
    PR_STATIC_OPERATION,
    PR_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.specs import PR_MATERIAL_ID
from lcprop.pr.transverse.operations import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)
from lcprop.runners.local import LocalRunner


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _configured_window(app, *, gain_length_product: float = 1.0e-3):
    window = PRMainWindow()
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    window.grid_panel.Nx.setValue(12)
    window.grid_panel.Ny.setValue(12)
    window.grid_panel.x_aperture_um.setValue(40.0)
    window.grid_panel.y_aperture_um.setValue(40.0)
    window.grid_panel.dz_um.setValue(5.0)
    window.grid_panel.z_length_um.setValue(10.0)
    window.material_panel.dark_intensity.setValue(0.4)
    window.material_panel.uniform_background_intensity.setValue(0.1)
    window.material_panel.applied_field.setValue(0.0)
    window.material_panel.gain_length_product.setValue(gain_length_product)
    window.material_panel.use_characteristic_wavenumber_override.setChecked(True)
    window.material_panel.characteristic_wavenumber_per_um_override.setValue(0.1)
    return window


def _wait_for(app, predicate, *, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for transverse PR GUI run")
        time.sleep(0.002)
    app.processEvents()


def test_gui_static_choice_identifies_canonical_2d_and_legacy_paths(app):
    window = PRMainWindow()
    canonical_index = window.evolution_panel.workflow.findData(
        PR_TRANSVERSE_STATIC_WORKFLOW
    )
    legacy_index = window.evolution_panel.workflow.findData(
        PR_STATIC_OPERATION.workflow_id
    )

    assert canonical_index >= 0
    assert legacy_index >= 0
    assert canonical_index < legacy_index
    assert "Full transverse PR transport" in (
        window.evolution_panel.workflow.itemText(canonical_index)
    )
    assert "Reduced x-only PR transport" in window.evolution_panel.workflow.itemText(
        legacy_index
    )
    assert window.runner.registered_operations == (
        PR_TIMEDEPENDENT_OPERATION,
        PR_IMAGE_AMPLIFICATION_OPERATION,
        PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
        PR_TRANSVERSE_STATIC_OPERATION,
        PR_STATIC_OPERATION,
    )

    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    label = window.evolution_panel._form.labelForField(
        window.evolution_panel.max_coupled_passes
    )
    assert label.text() == "Maximum coupled iterations"

    window.evolution_panel.set_workflow_id(PR_STATIC_OPERATION.workflow_id)
    assert label.text() == "Maximum coupled passes per slice"
    window.close()


def test_gui_builds_valid_canonical_transverse_static_request(app):
    window = _configured_window(app)
    request = window.build_request()
    preflight = validate_pr_transverse_static_gui_request(request)
    summary = window.describe_request(request)

    assert isinstance(request, PRTransverseStaticRunRequest)
    assert request.initial_A is None
    assert request.initial_psi is None
    assert request.transport.m_y == 1.0
    assert request.dielectric.h_y == 1.0
    assert request.projection.g_x == 1.0
    assert request.projection.g_y == 0.0
    assert request.boundary.x_boundary == "periodic"
    assert request.boundary.y_boundary == "periodic"
    assert request.solver.max_coupled_iterations == 20
    assert request.solver.optical_substeps == 1
    assert preflight.aperture is not None
    assert "Workflow: pr_transverse_static" in summary
    assert "Static material model: Fully nonlinear" in summary
    assert "Transport: Full transverse PR transport" in summary
    assert "Solved material fields: E_x and E_y" in summary
    assert "Scalar optical projection: E_active = E_x" in summary
    assert window.evolution_panel.Nt.isHidden()
    assert window.evolution_panel.dt_normalized.isHidden()
    assert window.evolution_panel.integrator.isHidden()
    window.close()


def test_gui_preserves_response_control_when_switching_to_transverse_td(app):
    window = _configured_window(app)
    panel = window.evolution_panel
    linearized_index = panel.material_response.findData(
        PR_MATERIAL_RESPONSE_LINEARIZED
    )
    panel.material_response.setCurrentIndex(linearized_index)
    panel.reference_intensity.setValue(1.5)
    panel.transverse_applied_field.setValue(0.25)

    request = window.build_request()
    summary = window.describe_request(request)
    assert request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
    assert request.material_response.reference_intensity == 1.5
    assert request.boundary.profile_id == (
        PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
    )
    assert request.boundary.applied_field_x == 0.25
    assert "Linearized material response [Experimental]" in summary
    assert "Linearization intensity I₀: 1.5" in summary
    assert "fixed harmonic mean field" in summary
    assert not panel.reference_intensity.isHidden()
    assert not panel.transverse_applied_field.isHidden()

    panel.set_workflow_id(PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.workflow_id)
    assert panel.material_response.currentData() == PR_MATERIAL_RESPONSE_LINEARIZED
    assert not panel.material_response.isHidden()
    assert not panel.reference_intensity.isHidden()
    assert not panel.transverse_applied_field.isHidden()
    window.close()


@pytest.mark.parametrize(
    ("material_response", "expected_badge", "expected_status", "warns"),
    (
        (
            PR_MATERIAL_RESPONSE_NONLINEAR,
            "[Validated]",
            "Validated for the Image Amplification experiment.",
            False,
        ),
        (
            PR_MATERIAL_RESPONSE_LINEARIZED,
            "[Experimental]",
            "Experimental: compatible architecture",
            True,
        ),
    ),
)
def test_image_amplification_status_tracks_transverse_material_response(
    app,
    material_response,
    expected_badge,
    expected_status,
    warns,
):
    window = _configured_window(app)
    panel = window.evolution_panel
    panel.set_image_amplification_mode(True)
    response_index = panel.material_response.findData(material_response)
    panel.material_response.setCurrentIndex(response_index)

    workflow_index = panel.workflow.findData(PR_TRANSVERSE_STATIC_WORKFLOW)
    assert expected_badge in panel.workflow.itemText(workflow_index)
    assert expected_status in panel.algorithm_status.text()

    request = window.build_request()
    capability = next(
        capability
        for capability in image_amplification_base_capabilities()
        if capability.workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW
    )
    before = window.results_panel.workspace.console.toPlainText()
    window._append_image_amplification_validation_warning(capability, request)
    after = window.results_panel.workspace.console.toPlainText()
    warning = "experimental; specialized validation is pending"
    assert (warning in after[len(before):]) is warns
    window.close()


def test_gui_preflight_rejects_unsupported_profile_v1_execution_settings(app):
    window = _configured_window(app)
    window.material_panel.applied_field.setValue(0.1)
    with pytest.raises(ValueError, match="zero normalized applied field"):
        window.build_request()

    window.material_panel.applied_field.setValue(0.0)
    window.evolution_panel.backend.setCurrentText("auto")
    with pytest.raises(ValueError, match="explicit NumPy or CuPy"):
        window.build_request()

    window.evolution_panel.backend.setCurrentText("numpy")
    window.evolution_panel.precision.setCurrentText("float32")
    with pytest.raises(ValueError, match="requires float64"):
        window.build_request()
    window.close()


def test_normal_gui_dispatch_runs_2d_zero_flux_and_presents_material_state(
    app,
    monkeypatch,
):
    window = _configured_window(app)
    gui_thread = QThread.currentThread()
    material_calls = []
    dispatch_calls = []
    original_material_solve = (
        transverse_static_module._solve_pr_transverse_static_intensity_host_volume
    )
    original_dispatch = window.runner.run_registered

    def observed_material_solve(*args, **kwargs):
        material_calls.append((args, kwargs))
        return original_material_solve(*args, **kwargs)

    def observed_dispatch(material_id, workflow_id, request, **kwargs):
        dispatch_calls.append(
            (material_id, workflow_id, type(request), QThread.currentThread())
        )
        return original_dispatch(material_id, workflow_id, request, **kwargs)

    monkeypatch.setattr(
        transverse_static_module,
        "_solve_pr_transverse_static_intensity_host_volume",
        observed_material_solve,
    )
    window.runner.run_registered = observed_dispatch
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert material_calls
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][:3] == (
        PR_MATERIAL_ID,
        PR_TRANSVERSE_STATIC_WORKFLOW,
        PRTransverseStaticRunRequest,
    )
    assert dispatch_calls[0][3] is not gui_thread
    assert window.last_runner_result.kind == PR_TRANSVERSE_STATIC_WORKFLOW
    assert window.last_runner_result.material_id == PR_MATERIAL_ID
    assert window.last_result.status == "converged"
    assert window.run_status == "completed"

    result = window.last_result
    assert result.psi_final.shape == (2, 12, 12)
    assert np.isfinite(result.psi_final).all()
    assert result.diagnostics["finite_material_state"] is True
    assert result.diagnostics["equilibrium_residual_rms"] <= (
        result.resolved_profile["solver"]["equilibrium_rms_tolerance"]
    )
    assert result.diagnostics["equilibrium_residual_max"] <= (
        result.resolved_profile["solver"]["equilibrium_max_tolerance"]
    )
    assert result.replay_diagnostics["complete_independent_replay"] is True

    run_data = window.last_runner_result.run_data
    for key in (
        "psi",
        "P",
        "E_x",
        "E_y",
        "E_active",
        "transport_intensity",
        "equilibrium_residual",
    ):
        assert key in run_data.fields
        assert run_data.fields[key].data.shape == (2, 12, 12)
        assert np.isfinite(run_data.fields[key].data).all()
    np.testing.assert_array_equal(
        run_data.fields["E_active"].data,
        run_data.fields["E_x"].data,
    )
    console = window.results_panel.workspace.console.toPlainText()
    assert "Authoritative zero-flux residual" in console
    assert "max|E_x|" in console
    assert "max|E_y|" in console
    assert "Material solve work" in console
    assert "2D zero-flux static solve converged" in console
    window.close()


def test_transverse_static_nonconvergence_is_not_presented_as_success(app):
    window = _configured_window(app, gain_length_product=0.2)
    window.evolution_panel.max_coupled_passes.setValue(1)
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert window.last_result.status == "not_converged"
    assert window.run_status == "not_converged"
    assert window.status_label.text() == "Not converged"
    console = window.results_panel.workspace.console.toPlainText()
    assert "2D zero-flux static solve did not converge" in console
    assert "termination=" in console
    assert "Authoritative zero-flux residual" in console
    window.close()


def test_local_runner_remains_material_neutral():
    source = inspect.getsource(LocalRunner)
    assert "lcprop.pr" not in source
    runner = LocalRunner(operations=(PR_TRANSVERSE_STATIC_OPERATION,))
    assert runner.registered_operations == (PR_TRANSVERSE_STATIC_OPERATION,)
