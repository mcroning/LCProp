from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QScrollArea

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.products.data_model import (
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)


def _td_fields(*, from_static: bool, state: str, scale: float = 1.0) -> RunData:
    x = np.linspace(-37.5, 37.5, 12)
    y = np.linspace(-50.0, 50.0, 14)
    z = np.linspace(0.0, 20.0, 5)
    volume = scale * np.arange(z.size * x.size * y.size, dtype=float).reshape(
        z.size, x.size, y.size
    )
    delta = volume / max(float(volume.max()), 1.0)
    if state == "running":
        intensity_label = "Intensity at current t"
        delta_label = "Delta Theta at current t"
    elif state == "stopped":
        intensity_label = "Intensity at Stop"
        delta_label = "Delta Theta at Stop"
    else:
        intensity_label = "Final Intensity"
        delta_label = "Final Delta Theta"

    fields = [
        (
            "final_intensity",
            make_field(
                "final_intensity",
                intensity_label,
                volume[-1],
                ("x", "y"),
                "intensity",
                {"x": "um", "y": "um"},
                value_unit="1/µm²",
            ),
        ),
        (
            "final_delta_theta",
            make_field(
                "final_delta_theta",
                delta_label,
                delta[-1],
                ("x", "y"),
                "theta_delta",
                {"x": "um", "y": "um"},
                value_unit="rad",
            ),
        ),
    ]
    if from_static:
        fields.extend(
            [
                (
                    "initial_static_intensity_stack",
                    make_field(
                        "initial_static_intensity_stack",
                        "Initial Static Intensity",
                        volume.copy(),
                        ("z", "x", "y"),
                        "intensity",
                        {"z": "um", "x": "um", "y": "um"},
                        "longitudinal",
                        value_unit="1/µm²",
                    ),
                ),
                (
                    "initial_static_delta_theta_stack",
                    make_field(
                        "initial_static_delta_theta_stack",
                        "Initial Static Delta Theta",
                        delta.copy(),
                        ("z", "x", "y"),
                        "theta_delta",
                        {"z": "um", "x": "um", "y": "um"},
                        "longitudinal",
                        value_unit="rad",
                    ),
                ),
            ]
        )
    fields.extend(
        [
            (
                "final_intensity_stack",
                make_field(
                    "final_intensity_stack",
                    intensity_label,
                    volume,
                    ("z", "x", "y"),
                    "intensity",
                    {"z": "um", "x": "um", "y": "um"},
                    "longitudinal",
                    value_unit="1/µm²",
                ),
            ),
            (
                "final_delta_theta_stack",
                make_field(
                    "final_delta_theta_stack",
                    delta_label,
                    delta,
                    ("z", "x", "y"),
                    "theta_delta",
                    {"z": "um", "x": "um", "y": "um"},
                    "longitudinal",
                    value_unit="rad",
                ),
            ),
        ]
    )
    return RunData(
        workflow="timedependent",
        geometry=Geometry(x=x, y=y, z=z),
        fields=FieldCollection(fields),
    )


def _rect_in(widget, ancestor) -> QRect:
    return QRect(widget.mapTo(ancestor, QPoint(0, 0)), widget.size())


def _assert_plot_artists_inside_canvas(view) -> None:
    view.draw()
    renderer = view.figure.canvas.get_renderer()
    for bounds in (
        view.ax.get_tightbbox(renderer),
        view.colorbar.ax.get_tightbbox(renderer),
    ):
        assert bounds.x0 >= 0.0
        assert bounds.y0 >= 0.0
        assert bounds.x1 <= view.width()
        assert bounds.y1 <= view.height()


@pytest.mark.parametrize("from_static", [False, True])
def test_longitudinal_fields_fit_default_window_for_all_td_states(from_static):
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.resize(1200, 760)
    window.tabs.setCurrentWidget(window.results_panel)
    window.show()

    expected_canvas_sizes = None
    for index, state in enumerate(("running", "stopped", "completed"), start=1):
        window.results_panel.set_run_data(
            _td_fields(from_static=from_static, state=state, scale=float(index))
        )
        app.processEvents()
        app.processEvents()

        pane = window.results_panel.workspace.longitudinal_pane
        pane_rect = pane.contentsRect()
        controls_rect = _rect_in(pane.controls_widget, pane)
        xz_row_rect = _rect_in(pane.xz_row, pane)
        yz_row_rect = _rect_in(pane.yz_row, pane)
        xz_rect = _rect_in(pane.xz_view, pane)
        yz_rect = _rect_in(pane.yz_view, pane)

        for widget in (
            pane.controls_widget,
            pane.y_cut_label,
            pane.y_cut_slider,
            pane.xz_view,
            pane.x_cut_label,
            pane.x_cut_slider,
            pane.yz_view,
        ):
            assert pane_rect.contains(_rect_in(widget, pane))
        assert controls_rect.bottom() < xz_row_rect.top()
        assert xz_row_rect.bottom() < yz_row_rect.top()
        assert pane_rect.contains(xz_rect)
        assert pane_rect.contains(yz_rect)
        assert xz_rect.bottom() < yz_rect.top()
        assert yz_rect.bottom() <= pane_rect.bottom()
        assert abs(pane.xz_row.height() - pane.yz_row.height()) <= 1
        assert abs(pane.xz_view.height() - pane.yz_view.height()) <= 1
        assert pane.findChildren(QScrollArea) == []
        _assert_plot_artists_inside_canvas(pane.xz_view)
        _assert_plot_artists_inside_canvas(pane.yz_view)
        transverse = window.results_panel.workspace.image_pane.image_view
        _assert_plot_artists_inside_canvas(transverse)
        transverse.draw()
        transverse_renderer = transverse.figure.canvas.get_renderer()
        transverse_colorbar = transverse.colorbar.ax.get_tightbbox(
            transverse_renderer
        )
        assert transverse.width() - transverse_colorbar.x1 >= 8.0

        sizes = (pane.xz_view.size(), pane.yz_view.size())
        if expected_canvas_sizes is None:
            expected_canvas_sizes = sizes
        else:
            assert sizes == expected_canvas_sizes

        delta_index = pane.field_selector.findData(
            "final_delta_theta_stack"
        )
        pane.field_selector.setCurrentIndex(delta_index)
        app.processEvents()
        assert (pane.xz_view.size(), pane.yz_view.size()) == expected_canvas_sizes
        _assert_plot_artists_inside_canvas(pane.xz_view)
        _assert_plot_artists_inside_canvas(pane.yz_view)

    window.close()


def test_ordinary_and_td_from_static_use_identical_longitudinal_geometry():
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.resize(1200, 760)
    window.tabs.setCurrentWidget(window.results_panel)
    window.show()

    geometries = []
    for from_static in (False, True):
        window.results_panel.set_run_data(
            _td_fields(from_static=from_static, state="running")
        )
        app.processEvents()
        pane = window.results_panel.workspace.longitudinal_pane
        geometries.append(
            (
                _rect_in(pane.xz_view, pane),
                _rect_in(pane.yz_view, pane),
            )
        )

    assert geometries[0] == geometries[1]
    window.close()
