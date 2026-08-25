import os
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.transverse.static_workflow import PR_TRANSVERSE_STATIC_WORKFLOW
from lcprop.runners.slurm import RemoteRunCancelled
from lcprop.transport.status import RemoteRunState, RemoteRunStatus
from lcprop.gui.remote_execution import remote_status_text


class DummySlurmRunner:
    name = "Slurm"
    supports_parallel_sweeps = False
    registered_operations = ()


def _app():
    return QApplication.instance() or QApplication([])


def test_lc_execution_target_defaults_local_and_only_static_is_remote_enabled():
    _app()
    window = LCPropMainWindow(slurm_runner=DummySlurmRunner())
    assert window.execution_target_selector.currentData() == "local"
    assert window.runner is window.local_runner
    window.execution_target_selector.setCurrentIndex(1)
    assert window.runner is window.slurm_runner
    assert window.runner_label.text() == "Runner: Slurm"
    window.experiment_panel.experiment.setCurrentText("Time-dependent propagation")
    assert not window.run_button.isEnabled()
    assert "canonical LC static" in window.run_button.toolTip()
    window.experiment_panel.experiment.setCurrentText("Static propagation")
    assert window.run_button.isEnabled()
    window.close()


def test_pr_execution_target_is_independent_of_scientific_backend():
    _app()
    window = PRMainWindow(slurm_runner=DummySlurmRunner())
    window.evolution_panel.backend.setCurrentText("cupy")
    window.execution_target_selector.setCurrentIndex(1)
    assert window.runner is window.slurm_runner
    assert window.evolution_panel.backend.currentText() == "cupy"
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    assert window.build_request().backend.backend == "cupy"
    window.execution_target_selector.setCurrentIndex(0)
    assert window.evolution_panel.backend.currentText() == "cupy"
    window.close()


def test_remote_status_display_includes_job_and_requested_resolved_backend():
    status = RemoteRunStatus(
        run_id="run", execution_target="slurm", state=RemoteRunState.COMPLETED,
        remote_job_id="123", scientific_backend_requested="auto",
        scientific_backend_resolved="cupy", device_summary={"device": "H200"},
    )
    text = remote_status_text(status)
    assert "Job ID: 123" in text
    assert "Backend requested: auto" in text
    assert "Backend resolved: cupy" in text
    assert "Device: H200" in text


def test_remote_cancellation_is_presented_as_stopped_not_failed():
    _app()
    status = RemoteRunStatus(
        run_id="cancelled-run",
        execution_target="slurm",
        state=RemoteRunState.CANCELLED,
        remote_job_id="456",
    )
    lc_window = LCPropMainWindow(slurm_runner=DummySlurmRunner())
    lc_window.last_remote_status = status
    lc_window._on_timedependent_failed("synthetic cancellation traceback")
    assert lc_window.run_status == "stopped"
    assert "Remote job cancelled" in (
        lc_window.results_panel.workspace.console.toPlainText()
    )
    lc_window.close()

    pr_window = PRMainWindow(slurm_runner=DummySlurmRunner())
    pr_window.last_remote_status = status
    pr_window._on_failed("synthetic cancellation traceback")
    assert pr_window.run_status == "stopped"
    assert pr_window.status_label.text() == "Stopped"
    assert "Remote job cancelled" in (
        pr_window.results_panel.workspace.console.toPlainText()
    )
    pr_window.close()


def test_pr_window_close_after_remote_cancel_joins_worker_thread():
    app = _app()
    window = PRMainWindow(slurm_runner=DummySlurmRunner())
    window.show()
    request = window.build_request()

    def cancelled_remote_run(
        _request, *, cancellation_token, progress_callback
    ):
        base = {
            "run_id": "lcprop-" + "a" * 32,
            "execution_target": "slurm",
            "remote_job_id": "456",
        }
        progress_callback(
            RemoteRunStatus(state=RemoteRunState.PENDING, **base)
        )
        while not cancellation_token.is_cancelled():
            sleep(0.001)
        progress_callback(
            RemoteRunStatus(state=RemoteRunState.CANCEL_REQUESTED, **base)
        )
        progress_callback(
            RemoteRunStatus(state=RemoteRunState.CANCELLED, **base)
        )
        raise RemoteRunCancelled("456")

    window._start_background(
        request,
        summary="remote cancellation lifecycle",
        runner_callable=cancelled_remote_run,
        run_label="Running",
    )

    deadline = monotonic() + 5.0
    while (
        window.last_remote_status is None
        or window.last_remote_status.state != RemoteRunState.PENDING
    ):
        assert monotonic() < deadline
        app.processEvents()
        sleep(0.001)

    window.stop_clicked()
    window.close()
    while window._thread is not None:
        assert monotonic() < deadline
        app.processEvents()
        sleep(0.001)

    app.processEvents()
    assert window.run_status == "stopped"
    assert not window._background_running
    assert window.last_remote_status.state == RemoteRunState.CANCELLED
    assert not window.isVisible()
