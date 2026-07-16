import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import FieldCollection, FieldData, to_run_data
from lcprop.workflows import run_static, run_timedependent
from lcprop.gui.views.image_pane import ImagePane, display_limits
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
    assert pane.field_selector.currentText() == "Final Intensity"

    final_delta_index = pane.field_selector.findData("final_delta_theta")
    pane.field_selector.setCurrentIndex(final_delta_index)
    pane.set_z_index(0)
    expected = np.asarray(result.theta_final)[0] - np.asarray(result.theta_bias)
    assert np.array_equal(
        np.asarray(pane.image_view.image.get_array()),
        expected.T,
    )


def _replace_field_data(run_data, key, data):
    return replace(
        run_data,
        fields=FieldCollection(
            (
                field_key,
                replace(field, data=data) if field_key == key else field,
            )
            for field_key, field in run_data.fields.items()
        ),
    )


def test_image_pane_geometry_is_stable_while_clim_updates():
    app = QApplication.instance() or QApplication([])
    run_data = to_run_data(run_static(make_base_static_request()))
    pane = ImagePane()
    pane.resize(500, 500)
    pane.show()
    pane.set_run_data(run_data)
    app.processEvents()

    initial_size = pane.image_view.size()
    initial_axes = pane.image_view.ax.get_position().bounds
    initial_colorbar = pane.image_view.colorbar.ax.get_position().bounds
    initial_ax_object = pane.image_view.ax
    initial_colorbar_object = pane.image_view.colorbar
    intensity_limits = pane.image_view.image.get_clim()
    for factor in (2.0, 3.0, 0.5):
        updated = _replace_field_data(
            run_data,
            "final_intensity",
            np.asarray(run_data.fields["final_intensity"].data) * factor,
        )
        pane.set_run_data(updated)
        app.processEvents()
        assert pane.image_view.size() == initial_size
        assert pane.image_view.ax.get_position().bounds == initial_axes
        assert pane.image_view.colorbar.ax.get_position().bounds == initial_colorbar
        assert pane.image_view.ax is initial_ax_object
        assert pane.image_view.colorbar is initial_colorbar_object

    assert pane.image_view.image.get_clim() != intensity_limits
    pane.close()


def test_robust_intensity_limits_clip_single_hot_pixel():
    data = np.ones((100, 100))
    data[50, 50] = 1.0e9
    field = FieldData(
        "intensity",
        "Intensity",
        data,
        ("x", "y"),
        "intensity",
    )

    vmin, vmax = display_limits(field)

    assert vmin == 0.0
    assert vmax == 1.0
    assert vmax < data.max()


def test_all_zero_intensity_has_finite_nonzero_display_range():
    field = FieldData(
        "intensity",
        "Intensity",
        np.zeros((8, 9)),
        ("x", "y"),
        "intensity",
    )

    vmin, vmax = display_limits(field)

    assert vmin == 0.0
    assert np.isfinite(vmax)
    assert vmax > 0.0


def test_field_dropdown_is_wide_enough_for_normal_labels():
    app = QApplication.instance() or QApplication([])
    pane = ImagePane()
    pane.set_run_data(to_run_data(run_static(make_base_static_request())))
    longest = "Output Plane Intensity at Stop"
    assert pane.field_selector.minimumWidth() >= (
        pane.field_selector.fontMetrics().horizontalAdvance(longest) + 30
    )
