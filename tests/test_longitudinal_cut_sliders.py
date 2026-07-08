import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_timedependent
from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from tests.test_all_workflows import make_base_static_request


def _td_run_data():
    base = make_base_static_request()
    result = run_timedependent(
        TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=1),
            output=base.output,
        )
    )
    return to_run_data(result)


def test_longitudinal_sliders_initialize_to_center():
    app = QApplication.instance() or QApplication([])
    pane = LongitudinalPane()
    pane.set_run_data(_td_run_data())

    assert pane.x_cut_slider.value() == (pane.x_cut_slider.maximum() + 1) // 2
    assert pane.y_cut_slider.value() == (pane.y_cut_slider.maximum() + 1) // 2
    assert "µm" in pane.x_cut_label.text()
    assert "µm" in pane.y_cut_label.text()


def test_longitudinal_sliders_update_views():
    app = QApplication.instance() or QApplication([])
    pane = LongitudinalPane()
    pane.set_run_data(_td_run_data())

    pane.x_cut_slider.setValue(3)
    pane.y_cut_slider.setValue(4)

    assert "x-z cut at y" in pane.y_cut_label.text()
    assert "y-z cut at x" in pane.x_cut_label.text()
    assert pane.xz_view.image is not None
    assert pane.yz_view.image is not None
