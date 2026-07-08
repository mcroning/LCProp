import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.gui.main_window import LCPropMainWindow


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
