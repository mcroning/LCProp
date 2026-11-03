from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_LIFECYCLE_SCRIPT = r"""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.static_workflow import PR_STATIC_WORKFLOW


mode = sys.argv[1]
app = QApplication([])
window = PRMainWindow()
window.grid_panel.Nx.setValue(8)
window.grid_panel.Ny.setValue(8)
window.grid_panel.dz_um.setValue(10.0)
window.grid_panel.z_length_um.setValue(40.0)
window.evolution_panel.Nt.setValue(20)
if mode == "static_close":
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)

original = window.runner.run_registered


def delayed_runner(material_id, workflow_id, request, **kwargs):
    gui_progress = kwargs["progress_callback"]

    def delayed_progress(progress):
        gui_progress(progress)
        time.sleep(0.03)

    return original(
        material_id,
        workflow_id,
        request,
        **{**kwargs, "progress_callback": delayed_progress},
    )


window.runner.run_registered = delayed_runner
app.aboutToQuit.connect(window.shutdown_background_run)
window.show()


def request_close_when_ready():
    if mode == "completed_close":
        if window._background_running:
            QTimer.singleShot(2, request_close_when_ready)
            return
    elif window.last_progress is None:
        QTimer.singleShot(2, request_close_when_ready)
        return
    if mode == "explicit_cancel_close":
        window.stop_clicked()
    if mode == "app_quit":
        app.quit()
    else:
        window.close()


QTimer.singleShot(0, window.run_clicked)
QTimer.singleShot(1, request_close_when_ready)
QTimer.singleShot(15000, lambda: os._exit(92))
exit_code = app.exec()

thread = window._thread
result = window.last_result
payload = {
    "mode": mode,
    "exit_code": exit_code,
    "thread_is_none": thread is None,
    "thread_running": False if thread is None else thread.isRunning(),
    "background_running": window._background_running,
    "status": None if result is None else result.status,
}
print(json.dumps(payload, sort_keys=True))

assert exit_code == 0
assert thread is None
assert not window._background_running
if mode == "completed_close":
    assert result is not None and result.status == "completed"
else:
    assert result is not None and result.status == "cancelled"
"""


@pytest.mark.parametrize(
    "mode",
    (
        "td_close",
        "static_close",
        "explicit_cancel_close",
        "app_quit",
        "completed_close",
    ),
)
def test_pr_window_process_joins_worker_before_close(mode, tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment.setdefault("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    completed = subprocess.run(
        [sys.executable, "-c", _LIFECYCLE_SCRIPT, mode],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    for forbidden in (
        "QThread: Destroyed while thread is still running",
        "Fatal Python error",
        "Segmentation fault",
        "SIGABRT",
    ):
        assert forbidden not in completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["mode"] == mode
    assert payload["thread_is_none"]
    assert not payload["thread_running"]
    assert not payload["background_running"]
