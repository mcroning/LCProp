import os
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication
import numpy as np

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.persistence.timedependent import (
    load_timedependent_checkpoint,
    save_timedependent_checkpoint,
)


def _tiny_td_window(*, steps=2):
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    win.experiment_panel.experiment.setCurrentText("Time-dependent propagation")
    win.grid_panel.Nx.setValue(16)
    win.grid_panel.Ny.setValue(16)
    win.grid_panel.dz_um.setValue(5.0)
    win.grid_panel.z_length_um.setValue(10.0)
    win.solver_panel.Nt.setValue(steps)
    return app, win


def _wait_for(app, predicate, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background TD run")
        time.sleep(0.002)
    app.processEvents()


def test_td_continue_requires_a_real_td_checkpoint():
    app, win = _tiny_td_window(steps=1)
    assert win.last_timedependent_checkpoint is None
    assert not win.continue_button.isEnabled()

    # A static result/checkpoint is not a TD continuation source.
    win.last_static_checkpoint = object()
    win.update_run_button()
    assert not win.continue_button.isEnabled()
    win.close()


def test_gui_builds_timedependent_request():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_timedependent_request()

    assert req.solver.Nt == 2
    assert req.solver.dt == 750e-6
    assert req.grid.Nx == 64


def test_gui_runs_timedependent_through_runner():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    win.experiment_panel.experiment.setCurrentText("Time-dependent propagation")
    req = win.build_timedependent_request()

    result = win.runner.run_timedependent(req)

    assert result.kind == "timedependent"
    assert result.result.theta_final.ndim == 3


def test_gui_timedependent_run_uses_worker_thread_and_processes_qt_event():
    app, win = _tiny_td_window()
    gui_thread = QThread.currentThread()
    worker_threads = []
    queued_event_seen = []
    responsive_while_running = []
    original = win._run_registered

    def observed_runner(operation, request, **kwargs):
        worker_threads.append(QThread.currentThread())
        time.sleep(0.05)
        return original(operation, request, **kwargs)

    win._run_registered = observed_runner
    QTimer.singleShot(0, lambda: queued_event_seen.append(True))
    win.run_static_clicked()
    assert not win.continue_button.isEnabled()

    deadline = time.monotonic() + 5.0
    while win._background_running:
        app.processEvents()
        if queued_event_seen and win._background_running:
            responsive_while_running.append(True)
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background TD run")
        time.sleep(0.002)

    assert worker_threads
    assert worker_threads[0] is not gui_thread
    assert responsive_while_running
    assert win.last_timedependent_checkpoint.status == "completed"
    assert win.run_button.isEnabled()
    assert not win.stop_button.isVisible()


def test_gui_stop_returns_cancelled_checkpoint_not_failure():
    app, win = _tiny_td_window(steps=20)
    original = win._run_registered

    def slow_runner(operation, request, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            operation,
            request,
            **{**kwargs, "progress_callback": slow_progress},
        )

    win._run_registered = slow_runner
    win.run_static_clicked()
    console = win.results_panel.workspace.console
    _wait_for(app, lambda: "segment step 1/20" in console.toPlainText())
    win.stop_timedependent_clicked()
    _wait_for(app, lambda: not win._background_running)

    assert win.last_timedependent_checkpoint.status == "cancelled"
    assert 1 <= win.last_timedependent_checkpoint.completed_steps < 20
    assert "Run cancelled" in console.toPlainText()
    assert "ERROR" not in console.toPlainText()
    assert win.run_button.isEnabled()
    assert win.continue_button.isEnabled()
    checkpoint_time = win.last_timedependent_checkpoint.current_time
    assert win.last_timedependent_progress.cumulative_time == checkpoint_time
    assert win.results_panel.workspace.image_pane.td_time_label.text() == (
        f"TD time at stop: {checkpoint_time:.3f}"
    )
    assert re.fullmatch(
        r"TD time at stop: -?\d+\.\d{3}",
        win.results_panel.workspace.image_pane.td_time_label.text(),
    )
    summary = (
        win.results_panel.workspace.image_pane._run_data
        .diagnostics["summary"].values
    )
    assert summary["cumulative_time"] == checkpoint_time
    assert f"cumulative_time: {checkpoint_time:.3f}" in console.toPlainText()
    assert (
        win.results_panel.workspace.image_pane.field_selector.currentData()
        == "final_intensity"
    )
    assert (
        win.results_panel.workspace.longitudinal_pane.field_selector.currentData()
        == "final_intensity_stack"
    )
    assert (
        win.results_panel.workspace.image_pane.field_selector.currentText()
        == "Intensity at Stop"
    )
    assert (
        win.results_panel.workspace.longitudinal_pane.field_selector.currentText()
        == "Intensity at Stop"
    )
    stopped_delta = (
        win.results_panel.workspace.image_pane._run_data
        .fields["final_delta_theta_stack"].data
    )
    expected_checkpoint_delta = (
        np.asarray(win.last_timedependent_checkpoint.theta)
        - np.asarray(win.last_timedependent_result.theta_bias)[None, :, :]
    )
    np.testing.assert_array_equal(stopped_delta, expected_checkpoint_delta)
    assert (
        win.results_panel.workspace.image_pane.field_selector.findData(
            "initial_intensity"
        )
        >= 0
    )
    assert (
        win.results_panel.workspace.image_pane.field_selector.findData(
            "initial_delta_theta"
        )
        >= 0
    )


def test_gui_worker_exception_is_returned_to_gui():
    app, win = _tiny_td_window()

    def failing_runner(operation, request, **kwargs):
        raise RuntimeError("worker test failure")

    win._run_registered = failing_runner
    win.run_static_clicked()
    _wait_for(app, lambda: not win._background_running)

    console_text = win.results_panel.workspace.console.toPlainText()
    assert "ERROR" in console_text
    assert "RuntimeError: worker test failure" in console_text
    assert win.run_button.isEnabled()


def test_gui_continuation_indicator_starts_from_checkpoint_and_updates_on_gui_thread():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    initial = win.runner.run_timedependent(request).result
    prior_time = initial.checkpoint.current_time
    gui_thread = QThread.currentThread()
    original_continue = win.runner.run_operation

    def slow_continue(operation, request, checkpoint, additional_steps, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original_continue(
            operation,
            request,
            checkpoint,
            additional_steps,
            **{**kwargs, "progress_callback": slow_progress},
        )

    win.runner.run_operation = slow_continue
    win.start_timedependent_continuation(
        request,
        initial.checkpoint,
        additional_steps=10,
    )

    label = win.results_panel.workspace.image_pane.td_time_label
    assert label.text() == f"TD time: {prior_time:.3f}"
    console = win.results_panel.workspace.console
    _wait_for(app, lambda: "segment step 1/10" in console.toPlainText())
    win.stop_timedependent_clicked()
    _wait_for(app, lambda: not win._background_running)

    checkpoint = win.last_timedependent_checkpoint
    assert checkpoint.current_time > prior_time
    assert checkpoint.current_time == win.last_timedependent_progress.cumulative_time
    assert label.text() == f"TD time at stop: {checkpoint.current_time:.3f}"
    assert win._last_td_progress_thread is gui_thread


def test_gui_completed_continuation_uses_cumulative_final_time_and_final_fields():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    initial = win.runner.run_timedependent(request).result

    win.start_timedependent_continuation(
        request,
        initial.checkpoint,
        additional_steps=2,
    )
    _wait_for(app, lambda: not win._background_running)

    result_time = win.last_timedependent_checkpoint.current_time
    assert result_time == 3 * request.solver.dt
    assert win.results_panel.workspace.image_pane.td_time_label.text() == (
        f"Final TD time: {result_time:.3f}"
    )
    image_pane = win.results_panel.workspace.image_pane
    longitudinal_pane = win.results_panel.workspace.longitudinal_pane
    assert image_pane.field_selector.currentData() == "final_intensity"
    assert longitudinal_pane.field_selector.currentData() == "final_intensity_stack"
    assert image_pane.field_selector.findData("initial_intensity") >= 0
    assert image_pane.field_selector.findData("initial_delta_theta") >= 0


def test_gui_continue_button_uses_retained_checkpoint():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    first = win.runner.run_timedependent(request).result
    win.last_timedependent_checkpoint = first.checkpoint
    win.update_run_button()

    assert win.continue_button.isEnabled()
    win.continue_button.click()
    _wait_for(app, lambda: not win._background_running)

    checkpoint = win.last_timedependent_checkpoint
    assert checkpoint.completed_steps == 2
    assert checkpoint.current_time == 2 * request.solver.dt
    assert win.results_panel.workspace.image_pane.td_time_label.text() == (
        f"Final TD time: {checkpoint.current_time:.3f}"
    )
    assert win.continue_button.isEnabled()


def test_gui_loaded_compatible_td_checkpoint_enables_continue(tmp_path):
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    result = win.runner.run_timedependent(request).result
    save_timedependent_checkpoint(result.checkpoint, tmp_path)

    win.last_timedependent_checkpoint = load_timedependent_checkpoint(tmp_path)
    win.update_run_button()

    assert win.continue_button.isEnabled()
    win.close()


def test_gui_fresh_run_is_explicit_and_does_not_consume_checkpoint():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    first = win.runner.run_timedependent(request).result
    win.last_timedependent_checkpoint = first.checkpoint
    win.update_run_button()

    win.run_button.click()
    _wait_for(app, lambda: not win._background_running)

    checkpoint = win.last_timedependent_checkpoint
    assert checkpoint.completed_steps == 1
    assert checkpoint.current_time == request.solver.dt
    assert "Starting a fresh TD run; previous continuation checkpoint cleared." in (
        win.results_panel.workspace.console.toPlainText()
    )


def test_gui_incompatible_change_invalidates_continue_visibly():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    first = win.runner.run_timedependent(request).result
    win.last_timedependent_checkpoint = first.checkpoint
    win.update_run_button()
    win.grid_panel.x_aperture_um.setValue(
        win.grid_panel.x_aperture_um.value() + 1.0
    )

    assert win.last_timedependent_checkpoint is None
    assert not win.continue_button.isEnabled()

    win.continue_button.click()
    app.processEvents()

    assert not win._background_running
    assert win.last_timedependent_checkpoint is None
    assert not win.continue_button.isEnabled()
    assert "Continuation invalidated: cannot continue TD checkpoint with incompatible grid" in (
        win.results_panel.workspace.console.toPlainText()
    )


def test_td_time_display_has_exactly_three_decimal_places_without_rounding_state():
    app, win = _tiny_td_window(steps=1)
    request = win.build_timedependent_request()
    win.run_static_clicked()
    _wait_for(app, lambda: not win._background_running)

    checkpoint = win.last_timedependent_checkpoint
    assert checkpoint.current_time == request.solver.dt
    assert checkpoint.current_time == 0.00075
    assert win.results_panel.workspace.image_pane.td_time_label.text() == (
        "Final TD time: 0.001"
    )
