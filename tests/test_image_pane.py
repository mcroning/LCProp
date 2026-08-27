import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton
import pytest
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
from lcprop.gui.workspace import Workspace
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
    z = np.linspace(0.0, 2.0, 3)
    first = np.arange(x.size * y.size, dtype=float).reshape(x.size, y.size)
    second = first + 1000.0
    volume = np.stack((first, first + 1.0, first + 2.0), axis=0)
    return RunData(
        workflow="image_interaction_test",
        geometry=Geometry(x=x, y=y, z=z, units="um"),
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
                        coordinates={
                            "x": np.linspace(100.0, 104.0, x.size),
                            "y": np.linspace(-2.0, 2.0, y.size),
                        },
                    ),
                ),
                (
                    "volume",
                    make_field(
                        "volume",
                        "Volume",
                        volume,
                        ("z", "x", "y"),
                        "intensity",
                        {"z": "um", "x": "um", "y": "um"},
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
    assert pane.image_view.ax.get_xlim() == (100.0, 104.0)
    assert pane.image_view.ax.get_ylim() == (-2.0, 2.0)
    pane.field_selector.setCurrentIndex(
        pane.field_selector.findData("recommended")
    )
    assert pane.image_view.ax.get_xlim() == (-4.0, 4.0)
    assert pane.image_view.ax.get_ylim() == (-2.0, 2.0)
    pane.close()


def _scroll(view, *, xdata: float, ydata: float, direction: str) -> None:
    view._on_scroll(
        SimpleNamespace(
            inaxes=view.ax,
            xdata=xdata,
            ydata=ydata,
            button=direction,
            step=1.0 if direction == "up" else -1.0,
        )
    )


def _pan(view, *, dx_pixels: float, dy_pixels: float) -> None:
    view._on_mouse_press(
        SimpleNamespace(
            inaxes=view.ax,
            button=MouseButton.RIGHT,
            x=300.0,
            y=250.0,
            xdata=0.0,
            ydata=0.0,
        )
    )
    view._on_mouse_motion(
        SimpleNamespace(x=300.0 + dx_pixels, y=250.0 + dy_pixels)
    )
    view._on_mouse_release(SimpleNamespace(button=MouseButton.RIGHT))


def test_repeated_off_center_zoom_out_clamps_to_rectangular_full_aperture():
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
    guide_x = tuple(view._vline.get_xdata())
    guide_y = tuple(view._hline.get_ydata())
    color_limits = view.image.get_clim()

    for _ in range(20):
        _scroll(view, xdata=3.9, ydata=1.9, direction="down")
    assert view.ax.get_xlim() == (-10.0, 10.0)
    assert view.ax.get_ylim() == (-5.0, 5.0)

    limits_at_boundary = (view.ax.get_xlim(), view.ax.get_ylim())
    _scroll(view, xdata=9.9, ydata=4.9, direction="down")
    assert (view.ax.get_xlim(), view.ax.get_ylim()) == limits_at_boundary
    assert tuple(view._vline.get_xdata()) == guide_x
    assert tuple(view._hline.get_ydata()) == guide_y
    assert view.image.get_clim() == color_limits
    assert np.array_equal(run_data.fields["recommended"].data, original)
    pane.close()


@pytest.mark.parametrize(
    ("dx_pixels", "dy_pixels", "expected_x", "expected_y"),
    (
        (10000.0, 0.0, (-10.0, -6.0), (-1.0, 1.0)),
        (-10000.0, 0.0, (6.0, 10.0), (-1.0, 1.0)),
        (0.0, 10000.0, (-2.0, 2.0), (-5.0, -3.0)),
        (0.0, -10000.0, (-2.0, 2.0), (3.0, 5.0)),
    ),
)
def test_pan_clamps_at_each_physical_boundary_without_resizing(
    dx_pixels,
    dy_pixels,
    expected_x,
    expected_y,
):
    app = QApplication.instance() or QApplication([])
    pane = ImagePane()
    pane.resize(600, 500)
    pane.show()
    pane.set_run_data(_interactive_run_data())
    app.processEvents()
    view = pane.image_view
    view.ax.set_xlim(-2.0, 2.0)
    view.ax.set_ylim(-1.0, 1.0)

    _pan(view, dx_pixels=dx_pixels, dy_pixels=dy_pixels)

    assert view.ax.get_xlim() == expected_x
    assert view.ax.get_ylim() == expected_y
    assert view.ax.get_xlim()[1] - view.ax.get_xlim()[0] == pytest.approx(4.0)
    assert view.ax.get_ylim()[1] - view.ax.get_ylim()[0] == pytest.approx(2.0)
    pane.close()


def test_full_aperture_disables_further_zoom_out_and_pan_beyond_bounds():
    app = QApplication.instance() or QApplication([])
    pane = ImagePane()
    pane.resize(600, 500)
    pane.show()
    pane.set_run_data(_interactive_run_data())
    app.processEvents()
    view = pane.image_view
    pane.full_aperture_button.click()
    full_limits = (view.ax.get_xlim(), view.ax.get_ylim())

    _scroll(view, xdata=9.0, ydata=4.0, direction="down")
    _pan(view, dx_pixels=10000.0, dy_pixels=-10000.0)

    assert (view.ax.get_xlim(), view.ax.get_ylim()) == full_limits
    pane.fit_button.click()
    assert view.ax.get_xlim() == (-4.0, 4.0)
    assert view.ax.get_ylim() == (-2.0, 2.0)
    pane.close()


def test_navigation_and_product_switching_do_not_mutate_physical_coordinates():
    app = QApplication.instance() or QApplication([])
    run_data = _interactive_run_data()
    geometry_x = np.asarray(run_data.geometry.x).copy()
    geometry_y = np.asarray(run_data.geometry.y).copy()
    product_x = np.asarray(run_data.fields["full"].coordinates["x"]).copy()
    product_y = np.asarray(run_data.fields["full"].coordinates["y"]).copy()
    pane = ImagePane()
    pane.resize(600, 500)
    pane.show()
    pane.set_run_data(run_data)
    app.processEvents()

    _scroll(pane.image_view, xdata=1.0, ydata=0.5, direction="up")
    _pan(pane.image_view, dx_pixels=50.0, dy_pixels=-30.0)
    pane.fit_button.click()
    pane.full_aperture_button.click()
    pane.field_selector.setCurrentIndex(pane.field_selector.findData("full"))
    pane.field_selector.setCurrentIndex(
        pane.field_selector.findData("recommended")
    )

    assert np.array_equal(run_data.geometry.x, geometry_x)
    assert np.array_equal(run_data.geometry.y, geometry_y)
    assert np.array_equal(run_data.fields["full"].coordinates["x"], product_x)
    assert np.array_equal(run_data.fields["full"].coordinates["y"], product_y)
    pane.close()


def test_left_click_after_navigation_reaches_longitudinal_cut_synchronization():
    app = QApplication.instance() or QApplication([])
    workspace = Workspace()
    workspace.resize(1000, 700)
    workspace.show()
    workspace.set_run_data(_interactive_run_data())
    app.processEvents()
    pane = workspace.image_pane
    selected = []
    pane.positionSelected.connect(lambda ix, iy: selected.append((ix, iy)))

    _scroll(pane.image_view, xdata=1.0, ydata=0.5, direction="up")
    _pan(pane.image_view, dx_pixels=30.0, dy_pixels=-20.0)
    pane.image_view._on_mouse_press(
        SimpleNamespace(
            inaxes=pane.image_view.ax,
            button=MouseButton.LEFT,
            x=300.0,
            y=250.0,
            xdata=0.0,
            ydata=0.0,
        )
    )

    assert selected == [(10, 5)]
    assert workspace.longitudinal_pane.x_cut_slider.value() == 10
    assert workspace.longitudinal_pane.y_cut_slider.value() == 5
    assert workspace.longitudinal_pane._ix == 10
    assert workspace.longitudinal_pane._iy == 5
    workspace.close()
