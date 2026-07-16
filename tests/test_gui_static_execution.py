from __future__ import annotations

from dataclasses import replace
import os
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication

from lcprop.gui.main_window import LCPropMainWindow


def _window():
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.experiment_panel.experiment.setCurrentText("Static propagation")
    request = window.build_request()
    request = replace(
        request,
        grid=replace(
            request.grid,
            Nx=8,
            Ny=10,
            z_length_um=20.0,
            dz_um=5.0,
        ),
        solver=replace(
            request.solver,
            static_max_coupled_passes=1,
            static_max_relax_iterations=1,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        ),
    )
    window.build_request = lambda **_kwargs: request
    return app, window, request


def _wait_for(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background static run")
        time.sleep(0.002)
    app.processEvents()


def test_gui_static_worker_stop_continue_and_live_latest_fields():
    app, window, request = _window()
    gui_thread = QThread.currentThread()
    worker_threads = []
    queued_event = []
    original = window.runner.run_static

    def slow_runner(request, **kwargs):
        worker_threads.append(QThread.currentThread())
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            request, **{**kwargs, "progress_callback": slow_progress}
        )

    window.runner.run_static = slow_runner
    QTimer.singleShot(0, lambda: queued_event.append(True))
    window.run_button.click()
    console = window.results_panel.workspace.console
    _wait_for(app, lambda: "slices 1/4" in console.toPlainText())

    assert queued_event
    assert worker_threads[0] is not gui_thread
    assert window.run_status == "running"
    assert (
        window.results_panel.workspace.image_pane.field_selector.currentData()
        == "final_intensity"
    )
    assert (
        window.results_panel.workspace.image_pane.field_selector.currentText()
        == "Intensity at current z"
    )

    window.stop_button.click()
    assert window.run_status == "stopping"
    _wait_for(app, lambda: not window._background_running)
    checkpoint = window.last_static_checkpoint
    assert window.run_status == "stopped"
    assert 1 <= checkpoint.completed_slices < 4
    assert checkpoint.theta_stack.shape[0] == checkpoint.completed_slices
    assert re.fullmatch(
        r"z at stop: -?\d+\.\d{3} um; slices: \d+/4",
        window.results_panel.workspace.image_pane.td_time_label.text(),
    )
    assert window.continue_button.isEnabled()

    window.continue_button.click()
    _wait_for(app, lambda: not window._background_running)
    assert window.run_status == "completed"
    assert window.last_static_result.completed_slices == 4
    assert window.last_static_checkpoint is None
    assert window.results_panel.workspace.image_pane.td_time_label.text() == (
        "Final z: 20.000 um; slices: 4/4"
    )
    assert (
        window.results_panel.workspace.image_pane.field_selector.findData(
            "input_intensity"
        )
        >= 0
    )
