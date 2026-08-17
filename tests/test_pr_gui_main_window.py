from __future__ import annotations

import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication
import numpy as np
import pytest

import lcprop.pr.gui.main_window as pr_main_window_module
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.operations import PR_STATIC_OPERATION, PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import (
    PR_EULER_INTEGRATOR,
    PR_MATERIAL_ID,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.pr.static_workflow import PRStaticRunRequest, PR_STATIC_WORKFLOW


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tiny_window(app, *, steps: int = 2) -> PRMainWindow:
    window = PRMainWindow()
    window.grid_panel.Nx.setValue(8)
    window.grid_panel.Ny.setValue(8)
    window.grid_panel.dz_um.setValue(10.0)
    window.grid_panel.z_length_um.setValue(20.0)
    window.evolution_panel.Nt.setValue(steps)
    return window


def _wait_for(app, predicate, *, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for PR GUI background run")
        time.sleep(0.002)
    app.processEvents()


def test_pr_window_is_standalone_and_registers_both_pr_operations(app):
    window = PRMainWindow()

    assert window.windowTitle() == "LCProp PR"
    assert window.runner.registered_operations == (
        PR_TIMEDEPENDENT_OPERATION,
        PR_STATIC_OPERATION,
    )
    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "PR Material",
        "Beam",
        "Grid",
        "Evolution",
        "Results",
    ]
    assert window.run_status == "idle"
    assert not window.stop_button.isVisible()
    window.close()


def test_static_mode_builds_request_and_gates_td_checkpoint_controls(app):
    window = _tiny_window(app)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)

    request = window.build_request()
    summary = window.describe_request(request)

    assert isinstance(request, PRStaticRunRequest)
    assert request.solver.material_solver is None
    assert "Workflow: pr_static" in summary
    assert "Maximum coupled passes per z slice: 20" in summary
    assert (
        "Static material solver: precision-aware automatic defaults" in summary
    )
    assert "Material steps:" not in summary
    assert "Normalized timestep:" not in summary
    assert "Material integrator:" not in summary
    assert "Conservative normalized timestep limit:" not in summary
    assert not window.continue_button.isEnabled()
    assert not window.save_checkpoint_button.isEnabled()
    window.close()


def test_pr_window_dispatches_exact_registered_operation_in_worker(app):
    window = _tiny_window(app)
    gui_thread = QThread.currentThread()
    worker_threads = []
    calls = []
    queued_event_seen = []
    original = window.runner.run_registered

    def observed(material_id, workflow_id, request, **kwargs):
        worker_threads.append(QThread.currentThread())
        calls.append((material_id, workflow_id, request))
        return original(material_id, workflow_id, request, **kwargs)

    window.runner.run_registered = observed
    QTimer.singleShot(0, lambda: queued_event_seen.append(True))
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert queued_event_seen
    assert worker_threads and worker_threads[0] is not gui_thread
    assert len(calls) == 1
    assert calls[0][0:2] == (PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW)
    assert window.last_runner_result.material_id == PR_MATERIAL_ID
    assert window.last_runner_result.kind == PR_TIMEDEPENDENT_WORKFLOW
    assert window.last_progress_thread is gui_thread
    assert window.run_status == "completed"
    assert window.last_result.status == "completed"
    assert window.last_checkpoint is window.last_result.checkpoint
    assert window.run_button.isEnabled()
    assert window.continue_button.isEnabled()
    assert window.save_checkpoint_button.isEnabled()
    assert not window.stop_button.isVisible()
    window.close()


def test_pr_window_dispatches_static_and_presents_registered_run_data(app):
    window = _tiny_window(app)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    gui_thread = QThread.currentThread()
    calls = []
    worker_threads = []
    original = window.runner.run_registered
    original_set_run_data = window.results_panel.set_run_data
    rendering_states = []

    def observed_set_run_data(run_data):
        rendering_states.append(
            (
                window.status_label.text(),
                window.results_panel.workspace.image_pane.td_time_label.text(),
                window.stop_button.isEnabled(),
                window._background_running,
            )
        )
        return original_set_run_data(run_data)

    window.results_panel.set_run_data = observed_set_run_data

    def observed(material_id, workflow_id, request, **kwargs):
        calls.append((material_id, workflow_id, request))
        worker_threads.append(QThread.currentThread())
        return original(material_id, workflow_id, request, **kwargs)

    window.runner.run_registered = observed
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert calls[0][0:2] == (PR_MATERIAL_ID, PR_STATIC_WORKFLOW)
    assert isinstance(calls[0][2], PRStaticRunRequest)
    assert worker_threads[0] is not gui_thread
    assert window.last_runner_result.kind == PR_STATIC_WORKFLOW
    assert window.last_result.status == "converged"
    assert window.run_status == "completed"
    assert window.status_label.text() == "Converged"
    assert window.last_progress.workflow == PR_STATIC_WORKFLOW
    assert (window.last_progress.diagnostics or {}).get("phase") == (
        "gui_products"
    )
    assert window.last_progress_thread is gui_thread
    assert not window.continue_button.isEnabled()
    assert not window.save_checkpoint_button.isEnabled()

    workspace = window.results_panel.workspace
    console_text = workspace.console.toPlainText()
    assert "PR static progress: slice 1/2" in console_text
    expected_phases = (
        "Validating static solution: independent replay 0/2",
        "Validating replay consistency...",
        "Preparing scientific results...",
        "Preparing GUI results...",
        "Rendering results...",
    )
    positions = [console_text.index(message) for message in expected_phases]
    assert positions == sorted(positions)
    assert rendering_states == [
        ("Rendering results...", "Rendering results...", True, True)
    ]
    assert "Static solve converged" in workspace.console.toPlainText()
    assert workspace.longitudinal_pane.field_selector.findData(
        "final_E_stack"
    ) >= 0
    assert (
        workspace.longitudinal_pane.field_selector.findData(
            "pr_driving_intensity_stack"
        )
        >= 0
    )
    assert (
        workspace.longitudinal_pane.field_selector.findData(
            "pr_static_residual_stack"
        )
        >= 0
    )
    assert workspace.curve_pane.curve_selector.findData(
        "static_final_residual_rms"
    ) >= 0
    assert workspace.curve_pane.curve_selector.findData(
        "static_final_residual_max"
    ) >= 0
    assert workspace.curve_pane.curve_selector.findData(
        "static_coupled_passes"
    ) >= 0
    diagnostics = workspace.diagnostics_view.toPlainText()
    for key, authoritative in (
        ("initial_E_stack", window.last_result.E_initial),
        ("final_E_stack", window.last_result.E_final),
        (
            "pr_driving_intensity_stack",
            window.last_result.source_intensity_stack,
        ),
        ("pr_static_residual_stack", window.last_result.residual_stack),
    ):
        presented = window.last_runner_result.run_data.fields[key].data
        assert np.shares_memory(presented, authoritative)
        assert not presented.flags.writeable
    assert "status: converged" in diagnostics
    assert "replay:" in diagnostics
    assert workspace.image_pane.td_time_label.text() == (
        "Final PR static z: 20 µm; slices: 2/2"
    )
    window.close()


def test_pr_window_does_not_present_static_nonconvergence_as_success(app):
    window = _tiny_window(app)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    window.evolution_panel.max_coupled_passes.setValue(1)
    window.material_panel.gain_length_product.setValue(3.0)

    window.run_clicked()
    _wait_for(app, lambda: not window._background_running)

    assert window.last_result.status == "not_converged"
    assert window.run_status == "not_converged"
    assert window.status_label.text() == "Not converged"
    assert "Static solve did not converge" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()


def test_pr_window_displays_request_progress_and_final_run_data(app):
    window = _tiny_window(app, steps=2)
    window.run_clicked()
    _wait_for(app, lambda: not window._background_running)

    workspace = window.results_panel.workspace
    request_text = workspace.request_summary.toPlainText()
    console_text = workspace.console.toPlainText()
    assert "Workflow: pr_timedependent" in request_text
    assert "phase gradients=(0, 0) rad/µm" in request_text
    assert "Preflight warnings:" in request_text
    assert "PR progress: step 1/2" in console_text
    assert "Run complete" in console_text
    assert workspace.image_pane.field_selector.findData("final_E") >= 0
    assert workspace.image_pane.td_time_label.text() == (
        "Final PR time: 0.002 normalized; steps: 2/2"
    )
    window.close()


def test_pr_window_stop_returns_cancelled_result_and_checkpoint(app):
    window = _tiny_window(app, steps=20)
    original = window.runner.run_registered

    def slow_runner(material_id, workflow_id, request, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            material_id,
            workflow_id,
            request,
            **{**kwargs, "progress_callback": slow_progress},
        )

    window.runner.run_registered = slow_runner
    window.run_clicked()
    console = window.results_panel.workspace.console
    _wait_for(app, lambda: "PR progress: step 1/20" in console.toPlainText())
    assert not window.continue_button.isEnabled()
    assert not window.save_checkpoint_button.isEnabled()
    assert not window.load_checkpoint_button.isEnabled()
    window.stop_button.click()
    assert window.run_status == "stopping"
    _wait_for(app, lambda: not window._background_running)

    assert window.run_status == "stopped"
    assert window.last_result.status == "cancelled"
    assert 1 <= window.last_result.completed_steps < 20
    assert window.last_checkpoint.status == "cancelled"
    assert "Run cancelled" in console.toPlainText()
    assert window.run_button.isEnabled()
    window.close()


def test_pr_window_static_stop_finishes_at_an_accepted_slice(app):
    window = _tiny_window(app)
    window.grid_panel.z_length_um.setValue(40.0)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    original = window.runner.run_registered

    def slow_runner(material_id, workflow_id, request, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            material_id,
            workflow_id,
            request,
            **{**kwargs, "progress_callback": slow_progress},
        )

    window.runner.run_registered = slow_runner
    window.run_clicked()
    console = window.results_panel.workspace.console
    _wait_for(
        app,
        lambda: "PR static progress: slice 1/4" in console.toPlainText(),
    )
    window.stop_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert window.run_status == "stopped"
    assert window.last_result.status == "cancelled"
    assert 1 <= window.last_result.completed_slices < 4
    assert len(window.last_result.slice_summaries) == (
        window.last_result.completed_slices
    )
    assert "Static run cancelled" in console.toPlainText()
    assert not window.continue_button.isEnabled()
    assert not window.save_checkpoint_button.isEnabled()
    window.close()


def test_pr_window_shutdown_cooperatively_joins_active_worker(app):
    window = _tiny_window(app, steps=20)
    original = window.runner.run_registered

    def slow_runner(material_id, workflow_id, request, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            material_id,
            workflow_id,
            request,
            **{**kwargs, "progress_callback": slow_progress},
        )

    window.runner.run_registered = slow_runner
    window.run_clicked()
    _wait_for(app, lambda: window.last_progress is not None)

    assert window.shutdown_background_run(timeout_ms=5000)
    app.processEvents()
    assert window._thread is None or not window._thread.isRunning()
    assert not window._background_running
    window.close()


def test_pr_window_joins_native_thread_before_normal_completion(app, monkeypatch):
    threads = []

    class ObservedThread(QThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wait_calls = 0
            threads.append(self)

        def wait(self, *args, **kwargs):
            self.wait_calls += 1
            return super().wait(*args, **kwargs)

    monkeypatch.setattr(pr_main_window_module, "QThread", ObservedThread)
    window = _tiny_window(app, steps=1)

    window.run_clicked()
    _wait_for(app, lambda: not window._background_running)

    assert len(threads) == 1
    assert threads[0].wait_calls >= 1
    assert window._thread is None
    assert window.last_result.status == "completed"
    window.close()


def test_pr_window_shared_disk_round_trip_hydrates_controls(app, tmp_path):
    source = _tiny_window(app, steps=1)
    source.run_clicked()
    _wait_for(app, lambda: not source._background_running)
    checkpoint = source.last_checkpoint
    directory = tmp_path / "pr-checkpoint"

    assert source.save_checkpoint_to(directory) == directory

    loaded_window = PRMainWindow()
    loaded = loaded_window.load_checkpoint_from(directory)

    assert (
        loaded_window.evolution_panel.workflow_id()
        == PR_TIMEDEPENDENT_WORKFLOW
    )
    assert loaded.request == checkpoint.request
    assert loaded_window.build_request() == checkpoint.request
    assert loaded_window.last_checkpoint is loaded
    assert loaded_window.continue_button.isEnabled()
    assert loaded_window.save_checkpoint_button.isEnabled()
    assert loaded_window.checkpoint_compatibility_reason is None
    assert loaded_window.results_panel.workspace.image_pane.td_time_label.text() == (
        "Loaded PR checkpoint: 0.001 normalized; step 1/1"
    )
    source.close()
    loaded_window.close()


def test_loading_td_checkpoint_returns_static_gui_to_td_mode(app, tmp_path):
    source = _tiny_window(app, steps=1)
    source.run_clicked()
    _wait_for(app, lambda: not source._background_running)
    directory = tmp_path / "static-to-td"
    source.save_checkpoint_to(directory)

    loaded_window = PRMainWindow()
    loaded_window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    assert not loaded_window.continue_button.isEnabled()
    loaded_window.load_checkpoint_from(directory)

    assert (
        loaded_window.evolution_panel.workflow_id()
        == PR_TIMEDEPENDENT_WORKFLOW
    )
    assert loaded_window.continue_button.isEnabled()
    assert loaded_window.save_checkpoint_button.isEnabled()
    source.close()
    loaded_window.close()


def test_disk_loaded_gui_continuation_matches_uninterrupted_run(app, tmp_path):
    first_window = _tiny_window(app, steps=1)
    first_window.run_clicked()
    _wait_for(app, lambda: not first_window._background_running)
    directory = tmp_path / "continuation"
    first_window.save_checkpoint_to(directory)

    resumed_window = PRMainWindow()
    checkpoint = resumed_window.load_checkpoint_from(directory)
    resumed_window.evolution_panel.Nt.setValue(2)
    assert resumed_window.continue_button.isEnabled()
    resumed_window.continue_button.click()
    _wait_for(app, lambda: not resumed_window._background_running)

    uninterrupted_request = replace(
        checkpoint.request,
        solver=replace(checkpoint.request.solver, Nt=3),
    )
    uninterrupted = run_pr_timedependent(uninterrupted_request)
    resumed = resumed_window.last_result
    assert resumed.completed_steps == 3
    assert resumed.requested_steps == 3
    assert resumed.time_normalized == pytest.approx(0.003)
    assert resumed.status == "completed"
    assert resumed.power_initial == uninterrupted.power_initial
    assert resumed.power_final == uninterrupted.power_final
    np.testing.assert_array_equal(resumed.A_final, uninterrupted.A_final)
    np.testing.assert_array_equal(resumed.E_final, uninterrupted.E_final)
    first_window.close()
    resumed_window.close()


def test_checkpoint_compatibility_tracks_edits_without_discarding_state(
    app,
    tmp_path,
):
    source = _tiny_window(app, steps=1)
    source.run_clicked()
    _wait_for(app, lambda: not source._background_running)
    directory = tmp_path / "compatibility"
    source.save_checkpoint_to(directory)

    window = PRMainWindow()
    checkpoint = window.load_checkpoint_from(directory)
    original_field = window.material_panel.applied_field.value()
    window.material_panel.applied_field.setValue(original_field + 0.25)

    assert window.last_checkpoint is checkpoint
    assert not window.continue_button.isEnabled()
    assert "incompatible material" in window.checkpoint_compatibility_reason
    assert "incompatible material" in window.continue_button.toolTip()

    window.material_panel.applied_field.setValue(original_field)
    assert window.last_checkpoint is checkpoint
    assert window.continue_button.isEnabled()
    assert window.checkpoint_compatibility_reason is None

    original_integrator = checkpoint.request.solver.integrator
    other_integrator = (
        PR_EULER_INTEGRATOR
        if original_integrator != PR_EULER_INTEGRATOR
        else PR_SEMI_IMPLICIT_INTEGRATOR
    )
    window.evolution_panel.integrator.setCurrentIndex(
        window.evolution_panel.integrator.findData(other_integrator)
    )
    assert not window.continue_button.isEnabled()
    assert "incompatible solver" in window.checkpoint_compatibility_reason
    window.evolution_panel.integrator.setCurrentIndex(
        window.evolution_panel.integrator.findData(original_integrator)
    )
    assert window.continue_button.isEnabled()

    window.evolution_panel.Nt.setValue(checkpoint.request.solver.Nt + 4)
    assert window.continue_button.isEnabled()
    assert window.last_checkpoint is checkpoint
    source.close()
    window.close()


def test_pr_window_rejects_checkpoint_from_another_material(
    app,
    monkeypatch,
    tmp_path,
):
    window = PRMainWindow()
    monkeypatch.setattr(
        "lcprop.pr.gui.main_window.load_run_checkpoint",
        lambda _path: object(),
    )

    with pytest.raises(TypeError, match="pr/pr_timedependent"):
        window.load_checkpoint_from(tmp_path)

    assert window.last_checkpoint is None
    assert not window.continue_button.isEnabled()
    window.close()
