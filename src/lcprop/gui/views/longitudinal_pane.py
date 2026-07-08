from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from lcprop.products.data_model import FieldData
from lcprop.gui.views.image_view import ImageView


class LongitudinalPane(QWidget):
    """Viewer for longitudinal x-z and y-z cuts from 3-D fields.

    Assumes fields with axes ("z", "x", "y").
    """

    def __init__(self):
        super().__init__()
        self._run_data = None

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("3-D field"))
        self.field_selector = QComboBox()
        self.field_selector.currentIndexChanged.connect(self._field_changed)
        controls.addWidget(self.field_selector)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.y_cut_label = QLabel("x-z cut at center y")
        self.y_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_cut_slider.valueChanged.connect(self._cut_changed)
        layout.addWidget(self.y_cut_label)
        layout.addWidget(self.y_cut_slider)

        self.xz_view = ImageView()
        layout.addWidget(self.xz_view)

        self.x_cut_label = QLabel("y-z cut at center x")
        self.x_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_cut_slider.valueChanged.connect(self._cut_changed)
        layout.addWidget(self.x_cut_label)
        layout.addWidget(self.x_cut_slider)

        self.yz_view = ImageView()
        layout.addWidget(self.yz_view)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        for key, field in run_data.fields.items():
            data = np.asarray(field.data)
            if data.ndim == 3 and field.axes == ("z", "x", "y"):
                self.field_selector.addItem(field.display_name, key)

        self.field_selector.blockSignals(False)

        if self.field_selector.count() > 0:
            self.field_selector.setCurrentIndex(0)
            self._field_changed(0)

    def _field_changed(self, index: int) -> None:
        if self._run_data is None or index < 0:
            return

        key = self.field_selector.itemData(index)
        if key is None:
            return

        field = self._run_data.fields[key]
        data = np.asarray(field.data)
        if data.ndim != 3:
            return

        _, nx, ny = data.shape

        self.x_cut_slider.blockSignals(True)
        self.y_cut_slider.blockSignals(True)
        self.x_cut_slider.setRange(0, nx - 1)
        self.y_cut_slider.setRange(0, ny - 1)
        self.x_cut_slider.setValue(nx // 2)
        self.y_cut_slider.setValue(ny // 2)
        self.x_cut_slider.blockSignals(False)
        self.y_cut_slider.blockSignals(False)

        self._update_views()

    def _cut_changed(self, _value: int) -> None:
        self._update_views()

    def _update_views(self) -> None:
        if self._run_data is None:
            return

        index = self.field_selector.currentIndex()
        if index < 0:
            return

        key = self.field_selector.itemData(index)
        if key is None:
            return

        field = self._run_data.fields[key]
        data = np.asarray(field.data)
        if data.ndim != 3:
            return

        _, nx, ny = data.shape
        ix = min(max(self.x_cut_slider.value(), 0), nx - 1)
        iy = min(max(self.y_cut_slider.value(), 0), ny - 1)

        x_value = self._coord_value("x", ix)
        y_value = self._coord_value("y", iy)

        self.y_cut_label.setText(f"x-z cut at y = {y_value:.6g} µm")
        self.x_cut_label.setText(f"y-z cut at x = {x_value:.6g} µm")

        xz = data[:, :, iy]   # (z, x)
        yz = data[:, ix, :]   # (z, y)

        xz_field = FieldData(
            key=f"{field.key}_xz",
            display_name=f"{field.display_name}(x,z)",
            data=xz,
            axes=("z", "x"),
            kind=field.kind,
            units=field.units,
            default_view="image",
            quantity=field.quantity,
            value_unit=field.value_unit,
            colormap=field.colormap,
        )

        yz_field = FieldData(
            key=f"{field.key}_yz",
            display_name=f"{field.display_name}(y,z)",
            data=yz,
            axes=("z", "y"),
            kind=field.kind,
            units=field.units,
            default_view="image",
            quantity=field.quantity,
            value_unit=field.value_unit,
            colormap=field.colormap,
        )

        self.xz_view.set_field(xz_field, extent=self._run_data.geometry.extent_zx())
        self.yz_view.set_field(yz_field, extent=self._run_data.geometry.extent_zy())

    def _coord_value(self, axis: str, index: int) -> float:
        if self._run_data is None:
            return float(index)
        return self._run_data.geometry.value(axis, index)
