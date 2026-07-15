from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from lcprop.products.data_model import FieldData
from lcprop.gui.views.image_view import ImageView


class LongitudinalPane(QWidget):
    cutChanged = Signal(int, int)
    zPlaneChanged = Signal(int)
    guidesVisibilityChanged = Signal(bool)
    """Viewer for longitudinal x-z and y-z cuts from 3-D fields.

    Assumes fields with axes ("z", "x", "y").
    """

    def __init__(self):
        super().__init__()
        self._run_data = None
        self._current_vmin = None
        self._current_vmax = None
        self._updating_cuts = False
        self._ix = 0
        self._iy = 0
        self._iz = 0
        self._show_guides = True

        layout = QVBoxLayout(self)

        self.no_data_label = QLabel("No longitudinal fields are available for this experiment.")
        self.no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_data_label.hide()

        self.controls_widget = QWidget()
        controls = QHBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("3-D field"))
        self.field_selector = QComboBox()
        self.field_selector.currentIndexChanged.connect(self._field_changed)
        controls.addWidget(self.field_selector)
        self.show_guides = QCheckBox("Show selection guides")
        self.show_guides.setChecked(True)
        self.show_guides.toggled.connect(self._show_guides_changed)
        controls.addWidget(self.show_guides)
        controls.addStretch(1)
        layout.addWidget(self.controls_widget)

        self.y_cut_label = QLabel("x-z cut at center y")
        self.y_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_cut_slider.valueChanged.connect(self._cut_changed)
        layout.addWidget(self.y_cut_label)
        layout.addWidget(self.y_cut_slider)

        self.xz_view = ImageView()
        layout.addWidget(self.xz_view)
        self.xz_view.positionSelected.connect(self._xz_position_selected)

        self.x_cut_label = QLabel("y-z cut at center x")
        self.x_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_cut_slider.valueChanged.connect(self._cut_changed)
        layout.addWidget(self.x_cut_label)
        layout.addWidget(self.x_cut_slider)

        self.yz_view = ImageView()
        layout.addWidget(self.yz_view)
        self.yz_view.positionSelected.connect(self._yz_position_selected)

        layout.addWidget(self.no_data_label)

    def _xz_position_selected(self, iz: int, ix: int) -> None:
        """Clicking an x-z view changes the selected x index for the y-z cut."""
        self._iz = int(iz)
        self.set_cut_indices(ix, self._iy)
        self.zPlaneChanged.emit(self._iz)

    def _yz_position_selected(self, iz: int, iy: int) -> None:
        """Clicking a y-z view changes the selected y index for the x-z cut."""
        self._iz = int(iz)
        self.set_cut_indices(self._ix, iy)
        self.zPlaneChanged.emit(self._iz)

    def _show_guides_changed(self, checked: bool) -> None:
        self._show_guides = bool(checked)
        self._apply_guides()
        self.guidesVisibilityChanged.emit(self._show_guides)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        for key, field in run_data.fields.items():
            data = np.asarray(field.data)
            if data.ndim == 3 and field.axes == ("z", "x", "y"):
                self.field_selector.addItem(field.display_name, key)

        has_fields = self.field_selector.count() > 0

        self.controls_widget.setVisible(has_fields)
        for widget in (
            self.y_cut_label,
            self.y_cut_slider,
            self.xz_view,
            self.x_cut_label,
            self.x_cut_slider,
            self.yz_view,
        ):
            widget.setVisible(has_fields)

        self.no_data_label.setVisible(not has_fields)

        if not has_fields:
# in longitudinal_pane.py
            self.xz_view.setVisible(False)
            self.yz_view.setVisible(False)
            self._current_vmin = None
            self._current_vmax = None
            self.field_selector.blockSignals(False)
            return

        self.field_selector.blockSignals(False)
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

        self._iz = data.shape[0] // 2

        self._current_vmin = float(np.nanmin(data))
        self._current_vmax = float(np.nanmax(data))

        if not self.show_guides.isChecked():
            self.show_guides.setChecked(True)
        else:
            self._show_guides = True
            self.guidesVisibilityChanged.emit(True)

        self.x_cut_slider.blockSignals(True)
        self.y_cut_slider.blockSignals(True)
        self.x_cut_slider.setRange(0, nx - 1)
        self.y_cut_slider.setRange(0, ny - 1)
        self.x_cut_slider.blockSignals(False)
        self.y_cut_slider.blockSignals(False)

        self.set_cut_indices(nx // 2, ny // 2, emit=False)

    def _cut_changed(self, _value: int) -> None:
        if self._updating_cuts:
            return
        self._ix = self.x_cut_slider.value()
        self._iy = self.y_cut_slider.value()
        self._update_views()
        self.cutChanged.emit(self._ix, self._iy)

    def set_cut_indices(self, ix: int, iy: int, *, emit: bool = True) -> None:
        """Set longitudinal cut indices in LCProp field order: x index, y index."""
        ix = min(max(int(ix), self.x_cut_slider.minimum()), self.x_cut_slider.maximum())
        iy = min(max(int(iy), self.y_cut_slider.minimum()), self.y_cut_slider.maximum())

        self._ix = ix
        self._iy = iy

        self._updating_cuts = True
        try:
            self.x_cut_slider.setValue(ix)
            self.y_cut_slider.setValue(iy)
        finally:
            self._updating_cuts = False

        self._update_views()
        if emit:
            self.cutChanged.emit(ix, iy)

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
        ix = min(max(int(self._ix), 0), nx - 1)
        iy = min(max(int(self._iy), 0), ny - 1)

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

        self.xz_view.set_field(
            xz_field,
            extent=self._run_data.geometry.extent_zx(),
            vmin=self._current_vmin,
            vmax=self._current_vmax,
        )
        self.yz_view.set_field(
            yz_field,
            extent=self._run_data.geometry.extent_zy(),
            vmin=self._current_vmin,
            vmax=self._current_vmax,
        )
        self._apply_guides()

    def _apply_guides(self) -> None:
        if self._show_guides:
            self.xz_view.set_crosshair(self._iz, self._ix)
            self.yz_view.set_crosshair(self._iz, self._iy)
        else:
            self.xz_view.clear_crosshair()
            self.yz_view.clear_crosshair()

    def _coord_value(self, axis: str, index: int) -> float:
        if self._run_data is None:
            return float(index)
        return self._run_data.geometry.value(axis, index)
