import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import from_timedependent_live_state, to_run_data
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
        "Initial Delta Theta",
        "Final Delta Theta",
    ]
    assert pane.field_selector.currentText() == "Final Intensity"
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


def test_live_td_selector_defaults_current_intensity_and_preserves_current_delta():
    app = QApplication.instance() or QApplication([])
    base = make_base_static_request()
    progress = []
    run_timedependent(
        TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=2),
            output=base.output,
        ),
        progress_callback=progress.append,
    )
    pane = LongitudinalPane()
    pane.set_run_data(
        from_timedependent_live_state(progress[0].latest_field_state)
    )

    labels = [
        pane.field_selector.itemText(i)
        for i in range(pane.field_selector.count())
    ]
    assert "Intensity at current t" in labels
    assert "Delta Theta at current t" in labels
    assert pane.field_selector.currentData() == "final_intensity_stack"
    assert pane.xz_view.ax.get_xlabel() == "z (um)"
    assert pane.xz_view.ax.get_ylabel() == "x (um)"
    assert pane.yz_view.ax.get_xlabel() == "z (um)"
    assert pane.yz_view.ax.get_ylabel() == "y (um)"

    pane.field_selector.setCurrentIndex(
        pane.field_selector.findData("final_delta_theta_stack")
    )
    pane.set_run_data(
        from_timedependent_live_state(progress[1].latest_field_state)
    )
    assert pane.field_selector.currentData() == "final_delta_theta_stack"
    longest = "Initial Static Delta Theta"
    assert pane.field_selector.minimumWidth() >= (
        pane.field_selector.fontMetrics().horizontalAdvance(longest) + 30
    )
