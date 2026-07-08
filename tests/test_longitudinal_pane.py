import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_timedependent
from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from tests.test_all_workflows import make_base_static_request


def test_timedependent_theta_stack_has_longitudinal_default_view():
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

    run_data = to_run_data(result)
    field = run_data.fields["theta_stack"]

    assert field.default_view == "longitudinal"
    assert field.axes == ("z", "x", "y")


def test_longitudinal_pane_displays_3d_field():
    app = QApplication.instance() or QApplication([])

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

    run_data = to_run_data(result)
    pane = LongitudinalPane()
    pane.set_run_data(run_data)

    assert pane.field_selector.count() == 2
    assert pane.field_selector.currentText() == "Delta theta stack"
    assert pane.xz_view.image is not None
    assert pane.yz_view.image is not None
