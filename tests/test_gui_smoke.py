import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.runners.local import LocalRunner


def test_gui_builds_static_request():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_request()

    assert req.grid.Nx == 64
    assert req.grid.z_length_um == 3000.0
    assert req.beams.channels[0].waist_x_um == 3.0
    assert req.beams.channels[0].power_mW == 1.0


def test_local_runner_static_smoke():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_request()

    result = LocalRunner().run_static(req)

    assert result.kind == "static"
    assert result.result.A_final.shape == (1, 64, 64)
    assert result.result.theta_final.shape == (64, 64)
