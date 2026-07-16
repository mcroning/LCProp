import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_soliton_existence, run_timedependent
from lcprop.workflows.soliton_existence import SolitonExistenceRequest
from lcprop.gui.views.curve_pane import CurvePane
from tests.test_all_workflows import make_base_static_request


def test_curve_pane_lists_existence_curves():
    app = QApplication.instance() or QApplication([])

    result = run_soliton_existence(
        SolitonExistenceRequest(
            base=make_base_static_request(),
            powers_mW=(0.03, 0.05),
            soliton_max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    run_data = to_run_data(result)

    pane = CurvePane()
    pane.set_run_data(run_data)

    assert pane.curve_selector.count() >= 1
    assert pane.curve_selector.itemText(0) == "Beta"
    assert pane.curve_view.line is not None


def test_curve_pane_can_switch_to_theta_max():
    app = QApplication.instance() or QApplication([])

    result = run_soliton_existence(
        SolitonExistenceRequest(
            base=make_base_static_request(),
            powers_mW=(0.03, 0.05),
            soliton_max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    run_data = to_run_data(result)

    pane = CurvePane()
    pane.set_run_data(run_data)

    for i in range(pane.curve_selector.count()):
        if pane.curve_selector.itemText(i) == "Theta max":
            pane.curve_selector.setCurrentIndex(i)
            break

    assert pane.curve_selector.currentText() == "Theta max"
    assert pane.curve_view.line is not None


def test_curve_pane_combines_soliton_existence_xs_and_ys():
    app = QApplication.instance() or QApplication([])
    result = run_soliton_existence(
        SolitonExistenceRequest(
            base=make_base_static_request(),
            powers_mW=(0.03, 0.05),
            soliton_max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    pane = CurvePane()
    pane.set_run_data(to_run_data(result))

    labels = [
        pane.curve_selector.itemText(index)
        for index in range(pane.curve_selector.count())
    ]
    assert "xs and ys" in labels

    pane.curve_selector.setCurrentIndex(labels.index("xs and ys"))
    assert pane.curve_view.ax.get_ylabel() == "RMS width (µm)"
    assert len(pane.curve_view.lines) == 2
    assert [line.get_label() for line in pane.curve_view.lines] == ["xs", "ys"]


def test_curve_pane_lists_timedependent_beam_widths():
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

    pane = CurvePane()
    pane.set_run_data(to_run_data(result))

    assert [
        pane.curve_selector.itemText(index)
        for index in range(pane.curve_selector.count())
    ] == ["Beam x RMS width", "Beam y RMS width"]
    assert pane.curve_view.ax.get_ylabel() == "x RMS width (µm)"
