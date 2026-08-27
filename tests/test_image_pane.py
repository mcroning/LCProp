import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton
from PySide6.QtWidgets import QApplication

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import (
    FieldCollection,
    FieldData,
    Geometry,
    RunData,
    make_field,
    to_run_data,
)
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


def _interactive_run_data():
    x = np.linspace(-10.0, 10.0, 21)
    y = np.linspace(-5.0, 5.0, 11)
    first = np.arange(x.size * y.size, dtype=float).reshape(x.size, y.size)
    second = first + 1000.0
    return RunData(
        workflow="image_interaction_test",
        geometry=Geometry(x=x, y=y, units="um"),
        fields=FieldCollection(
            [
                (
                    "recommended",
                    make_field(
                        "recommended",
                        "Recommended",
                        first,
                        ("x", "y"),
                        "intensity",
                        {"x": "um", "y": "um"},
                        default_display_extent=(-4.0, 4.0, -2.0, 2.0),
                        initially_selected=True,
                    ),
                ),
                (
                    "full",
                    make_field(
                        "full",
                        "Full",
                        second,
                        ("x", "y"),
                        "intensity",
                        {"x": "um", "y": "um"},
                    ),
                ),
            ]
        ),
    )


def test_transverse_zoom_pan_fit_and_full_aperture_preserve_data_and_guides():
    app = QApplication.instance() or QApplication([])
    run_data = _interactive_run_data()
    original = np.asarray(run_data.fields["recommended"].data).copy()
    pane = ImagePane()
    pane.resize(600, 500)
    pane.show()
    pane.set_run_data(run_data)
    app.processEvents()
    view = pane.image_view
    view.set_crosshair(7, 3)
    crosshair_x = tuple(view._vline.get_xdata())
    crosshair_y = tuple(view._hline.get_ydata())

    view._on_scroll(
        SimpleNamespace(
            inaxes=view.ax,
            xdata=1.0,
            ydata=0.5,
            button="up",
            step=1.0,
        )
    )
    zoomed_x = view.ax.get_xlim()
    zoomed_y = view.ax.get_ylim()
    assert zoomed_x[1] - zoomed_x[0] < 8.0
    assert zoomed_y[1] - zoomed_y[0] < 4.0
    view._on_mouse_press(
        SimpleNamespace(
            inaxes=view.ax,
            button=MouseButton.LEFT,
            x=300.0,
            y=250.0,
            xdata=0.0,
            ydata=0.0,
        )
    )
    assert view._crosshair_index == (10, 5)
    crosshair_x = tuple(view._vline.get_xdata())
    crosshair_y = tuple(view._hline.get_ydata())

    view._on_mouse_press(
        SimpleNamespace(
            inaxes=view.ax,
            button=MouseButton.RIGHT,
            x=300.0,
            y=250.0,
            xdata=1.0,
            ydata=0.5,
        )
    )
    view._on_mouse_motion(SimpleNamespace(x=320.0, y=265.0))
    assert view.ax.get_xlim() != zoomed_x
    assert view.ax.get_ylim() != zoomed_y
    view._on_mouse_release(SimpleNamespace(button=MouseButton.RIGHT))

    pane.fit_button.click()
    assert view.ax.get_xlim() == (-4.0, 4.0)
    assert view.ax.get_ylim() == (-2.0, 2.0)
    pane.full_aperture_button.click()
    assert view.ax.get_xlim() == (-10.0, 10.0)
    assert view.ax.get_ylim() == (-5.0, 5.0)
    assert tuple(view._vline.get_xdata()) == crosshair_x
    assert tuple(view._hline.get_ydata()) == crosshair_y
    assert np.array_equal(run_data.fields["recommended"].data, original)
    pane.close()


def test_transverse_product_switch_resets_each_field_to_its_canonical_view():
    app = QApplication.instance() or QApplication([])
    run_data = _interactive_run_data()
    pane = ImagePane()
    pane.set_run_data(run_data)
    pane.image_view.ax.set_xlim(-1.0, 1.0)
    pane.image_view.ax.set_ylim(-1.0, 1.0)

    pane.field_selector.setCurrentIndex(pane.field_selector.findData("full"))
    assert pane.image_view.ax.get_xlim() == (-10.0, 10.0)
    assert pane.image_view.ax.get_ylim() == (-5.0, 5.0)
    pane.field_selector.setCurrentIndex(
        pane.field_selector.findData("recommended")
    )
    assert pane.image_view.ax.get_xlim() == (-4.0, 4.0)
    assert pane.image_view.ax.get_ylim() == (-2.0, 2.0)
    pane.close()
