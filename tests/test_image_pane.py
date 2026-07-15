import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_static, run_timedependent
from lcprop.gui.views.image_pane import ImagePane
from tests.test_all_workflows import make_base_static_request


def test_image_pane_lists_2d_fields():
    app = QApplication.instance() or QApplication([])

    run_data = to_run_data(run_static(make_base_static_request()))
    pane = ImagePane()
    pane.set_run_data(run_data)

    assert [
        pane.field_selector.itemText(i)
        for i in range(pane.field_selector.count())
    ] == [
        "Input Plane Intensity",
        "Output Plane Intensity",
        "Input Plane Δθ",
        "Output Plane Δθ",
    ]
    assert pane.image_view.image is not None


def test_image_pane_can_switch_to_output_delta_theta():
    app = QApplication.instance() or QApplication([])

    run_data = to_run_data(run_static(make_base_static_request()))
    pane = ImagePane()
    pane.set_run_data(run_data)

    for i in range(pane.field_selector.count()):
        if pane.field_selector.itemText(i) == "Output Plane Δθ":
            pane.field_selector.setCurrentIndex(i)
            break

    assert pane.field_selector.currentText() == "Output Plane Δθ"
    assert pane.image_view.image is not None


def test_timedependent_image_pane_lists_only_initial_and_final_fields():
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
    pane = ImagePane()
    pane.set_run_data(to_run_data(result))

    assert [
        pane.field_selector.itemText(i)
        for i in range(pane.field_selector.count())
    ] == [
        "Initial Intensity",
        "Final Intensity",
        "Initial Δθ",
        "Final Δθ",
    ]

    final_delta_index = pane.field_selector.findData("final_delta_theta")
    pane.field_selector.setCurrentIndex(final_delta_index)
    pane.set_z_index(0)
    expected = np.asarray(result.theta_final)[0] - np.asarray(result.theta_bias)
    assert np.array_equal(
        np.asarray(pane.image_view.image.get_array()),
        expected.T,
    )
