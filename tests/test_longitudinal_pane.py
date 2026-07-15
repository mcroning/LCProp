import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_static, run_timedependent
from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from tests.test_all_workflows import make_base_static_request


def test_timedependent_delta_theta_stack_has_longitudinal_default_view():
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
    field = run_data.fields["final_delta_theta_stack"]

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

    assert [
        pane.field_selector.itemText(i)
        for i in range(pane.field_selector.count())
    ] == [
        "Initial Intensity",
        "Final Intensity",
        "Initial Δθ",
        "Final Δθ",
    ]
    assert pane.field_selector.currentText() == "Initial Intensity"
    assert pane.xz_view.image is not None
    assert pane.yz_view.image is not None


def test_static_longitudinal_pane_lists_intensity_and_delta_theta():
    app = QApplication.instance() or QApplication([])
    run_data = to_run_data(run_static(make_base_static_request()))
    pane = LongitudinalPane()
    pane.set_run_data(run_data)

    assert [
        pane.field_selector.itemText(i)
        for i in range(pane.field_selector.count())
    ] == ["Intensity", "Δθ"]
    assert pane.xz_view.ax.get_title() == "Intensity(x,z)"
    assert pane.yz_view.ax.get_title() == "Intensity(y,z)"
