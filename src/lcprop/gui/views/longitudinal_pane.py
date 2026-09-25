from __future__ import annotations

import numpy as np
from lcprop.gui.number_format import format_number

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lcprop.products.data_model import FieldData
from lcprop.gui.views.image_view import ImageView
from lcprop.gui.views.display_scale import DisplayScales, DisplayScaleControls, scale_key, volume_limits


class LongitudinalPane(QWidget):
    cutChanged = Signal(int, int)
    zPlaneChanged = Signal(int)
    volumeSelectionChanged = Signal(str)
    guidesVisibilityChanged = Signal(bool)
    guidesChanged = Signal()
    """Viewer for longitudinal x-z and y-z cuts from 3-D fields.

    Assumes fields with axes ("z", "x", "y").
    """

    _RETAINED_FAST_CUTS = "__retained_fast_optical_intensity_cuts__"

    def __init__(self, scales=None):
        super().__init__()
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.scales = scales if scales is not None else DisplayScales()
        self._run_data = None
        self._current_vmin = None
        self._current_vmax = None
        self._updating_cuts = False
        self._ix = 0
        self._iy = 0
        self._iz = 0
        self._show_guides = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        self.no_data_label = QLabel("No longitudinal fields are retained in this result.")
        self.no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_data_label.hide()

        self.controls_widget = QWidget()
        self.controls_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        controls = QVBoxLayout(self.controls_widget)
        selector_row = QHBoxLayout()
        controls.addLayout(selector_row)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(2)
        self.field_selector_label = QLabel("3-D field")
        selector_row.addWidget(self.field_selector_label)
        self.field_selector = QComboBox()
        self.field_selector.setMinimumWidth(180)
        self.field_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.field_selector.setMinimumContentsLength(20)
        self.field_selector.currentIndexChanged.connect(self._field_changed)
        selector_row.addWidget(self.field_selector, 1)
        self.position_label = QLabel("x=—, y=—, z=—")
        self.position_label.setWordWrap(True)
        position_row = QHBoxLayout()
        position_row.addWidget(self.position_label, 1)
        controls.addLayout(position_row)
        self.show_guides = QCheckBox("Show selection guides")
        self.show_guides.setChecked(True)
        self.show_guides.toggled.connect(self._show_guides_changed)
        position_row.addWidget(self.show_guides)
        controls.addStretch(1)
        layout.addWidget(self.controls_widget)

        self.scale_controls = DisplayScaleControls(self.scales)
        layout.addWidget(self.scale_controls)
        self.scales.changed.connect(self._update_views)

        z_controls = QHBoxLayout()
        self.z_plane_label = QLabel("z plane")
        self.z_plane_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_plane_slider.valueChanged.connect(self._z_changed)
        z_controls.addWidget(self.z_plane_label)
        z_controls.addWidget(self.z_plane_slider, 1)
        layout.addLayout(z_controls)

        self.xz_row = QWidget()
        self.xz_row.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        xz_layout = QVBoxLayout(self.xz_row)
        xz_layout.setContentsMargins(0, 0, 0, 0)
        xz_layout.setSpacing(1)

        self.y_cut_label = QLabel("x-z cut at center y")
        self.y_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_cut_slider.valueChanged.connect(self._cut_changed)
        xz_layout.addWidget(self.y_cut_label)
        xz_layout.addWidget(self.y_cut_slider)

        self.xz_view = ImageView(compact_vertical=True)
        self.xz_view.setMinimumSize(300, 180)
        xz_layout.addWidget(self.xz_view, 1)
        self.xz_view.positionSelected.connect(self._xz_position_selected)
        layout.addWidget(self.xz_row, 1)

        self.yz_row = QWidget()
        self.yz_row.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        yz_layout = QVBoxLayout(self.yz_row)
        yz_layout.setContentsMargins(0, 0, 0, 0)
        yz_layout.setSpacing(1)

        self.x_cut_label = QLabel("y-z cut at center x")
        self.x_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_cut_slider.valueChanged.connect(self._cut_changed)
        yz_layout.addWidget(self.x_cut_label)
        yz_layout.addWidget(self.x_cut_slider)

        self.yz_view = ImageView(compact_vertical=True)
        self.yz_view.setMinimumSize(300, 180)
        yz_layout.addWidget(self.yz_view, 1)
        self.yz_view.positionSelected.connect(self._yz_position_selected)
        layout.addWidget(self.yz_row, 1)

        layout.addWidget(self.no_data_label)

    def _xz_position_selected(self, iz: int, ix: int) -> None:
        """Clicking an x-z view changes the selected x index for the y-z cut."""
        self._iz = int(iz)
        self._set_z_slider(self._iz)
        if self._is_fixed_cut_selection():
            self._update_fixed_cut_views()
            self.zPlaneChanged.emit(self._iz)
            return
        self.set_cut_indices(ix, self._iy)
        self.zPlaneChanged.emit(self._iz)

    def _yz_position_selected(self, iz: int, iy: int) -> None:
        """Clicking a y-z view changes the selected y index for the x-z cut."""
        self._iz = int(iz)
        self._set_z_slider(self._iz)
        if self._is_fixed_cut_selection():
            self._update_fixed_cut_views()
            self.zPlaneChanged.emit(self._iz)
            return
        self.set_cut_indices(self._ix, iy)
        self.zPlaneChanged.emit(self._iz)

    def _show_guides_changed(self, checked: bool) -> None:
        self._show_guides = bool(checked)
        self._apply_guides()
        self.guidesVisibilityChanged.emit(self._show_guides)

    def set_progress_state(self, in_progress):
        if self._run_data is None or self.field_selector.count():
            return
        message = getattr(self._run_data, "longitudinal_message", None)
        if message is None and self._run_data.workflow in {"soliton", "soliton_existence"}:
            message = "Stationary soliton results contain transverse fields only."
        self.no_data_label.setText(message or (
            "Longitudinal fields are not available yet."
            if in_progress and self._run_data.workflow in {"static", "timedependent"}
            else "No longitudinal fields are retained in this result."
        ))

    def set_run_data(self, run_data) -> None:
        previous_key = self.field_selector.currentData() or getattr(self, "_preferred_key", None)
        self._preferred_key = previous_key
        previous_data = self._run_data
        previous_position = (getattr(self, "_ix", 0), getattr(self, "_iy", 0),
                             getattr(self, "_iz", 0))
        self._fixed_cut_fields = {}
        self._run_data = run_data
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        if getattr(run_data, "longitudinal_enabled", True):
            for key, field in run_data.fields.items():
                data = np.asarray(field.data)
                if data.ndim == 3 and field.axes == ("z", "x", "y"):
                    self.field_selector.addItem(field.display_name, key)
                    self.field_selector.setItemData(
                        self.field_selector.count() - 1,
                        "Snapshot axes: z, x, y; selected TD time is a parameter.",
                        Qt.ItemDataRole.ToolTipRole,
                    )
            xz = run_data.fields.get("retained_fast_optical_intensity_xz")
            yz = run_data.fields.get("retained_fast_optical_intensity_yz")
            if (
                xz is not None
                and yz is not None
                and xz.axes == ("z", "x")
                and yz.axes == ("z", "y")
                and xz.source_volume_key == yz.source_volume_key
                == "retained_fast_optical_intensity"
            ):
                self._fixed_cut_fields[self._RETAINED_FAST_CUTS] = (
                    "retained_fast_optical_intensity_xz", "retained_fast_optical_intensity_yz")
                self.field_selector.addItem(
                    "Retained Fast Optical Intensity",
                    self._RETAINED_FAST_CUTS,
                )
                self.field_selector.setItemData(
                    self.field_selector.count() - 1,
                    "Fast result: exact full-resolution fixed nearest-zero "
                    "x-z and y-z cuts, separate from any downsampled MPR preview.",
                    Qt.ItemDataRole.ToolTipRole,
                )

            # Generic fixed paired cuts are explicitly declared by the product.
            for key, xz in run_data.fields.items():
                partner = xz.coordinates.get("paired_cut_key")
                yz = run_data.fields.get(partner) if partner else None
                if (xz.axes == ("z", "x") and yz is not None
                        and yz.axes == ("z", "y")
                        and yz.coordinates.get("paired_cut_key") == key
                        and xz.source_volume_key == yz.source_volume_key
                        and xz.quantity == yz.quantity and xz.value_unit == yz.value_unit):
                    if (np.ndim(xz.data) != 2 or np.ndim(yz.data) != 2
                            or np.shape(xz.data)[0] != np.shape(yz.data)[0]
                            or not np.array_equal(
                                xz.coordinates.get("z", run_data.geometry.z),
                                yz.coordinates.get("z", run_data.geometry.z))):
                        raise ValueError("Paired longitudinal cuts require matching z coordinates")
                    selection = "__paired_cuts__:" + key
                    self._fixed_cut_fields[selection] = (key, partner)
                    self.field_selector.addItem(xz.display_name, selection)

        has_fields = self.field_selector.count() > 0

        self.controls_widget.setVisible(has_fields)
        self.scale_controls.setEnabled(has_fields)
        for widget in (
            self.xz_row,
            self.yz_row,
            self.z_plane_label,
            self.z_plane_slider,
        ):
            widget.setVisible(has_fields)

        message = getattr(run_data, "longitudinal_message", None)
        self.no_data_label.setText(
            message
            or "No longitudinal fields are retained in this result."
        )
        self.no_data_label.setVisible(not has_fields)

        if not has_fields:
            self.xz_view.clear_field()
            self.yz_view.clear_field()
            self._current_vmin = None
            self._current_vmax = None
            self.field_selector.blockSignals(False)
            return

        self.field_selector.blockSignals(False)
        default_index = 0
        preferred_volume_found = False
        for preferred_key in (
            "fast_optical_intensity_preview",
            "optical_intensity_stack",
        ):
            preferred_index = self.field_selector.findData(preferred_key)
            if preferred_index >= 0:
                default_index = preferred_index
                preferred_volume_found = True
                break
        if run_data.workflow == "timedependent" and not preferred_volume_found:
            final_index = -1
            if previous_key in {
                "final_intensity_stack",
                "final_delta_theta_stack",
            }:
                final_index = self.field_selector.findData(previous_key)
            if final_index < 0:
                final_index = self.field_selector.findData("final_intensity_stack")
            if final_index < 0:
                final_index = self.field_selector.findData(
                    "final_delta_theta_stack"
                )
            if final_index >= 0:
                default_index = final_index
        previous_index = self.field_selector.findData(previous_key)
        if previous_index >= 0:
            default_index = previous_index
        self.field_selector.setCurrentIndex(default_index)
        self._field_changed(default_index)
        if (previous_data is not None and previous_key == self.field_selector.currentData()
                and self._same_coordinates(previous_data, run_data, previous_key)):
            ix, iy, iz = previous_position
            self.z_plane_slider.setValue(min(iz, self.z_plane_slider.maximum()))
            if not self._is_fixed_cut_selection():
                self.set_cut_indices(ix, iy)

    def _same_coordinates(self, previous, current, selection):
        keys = self._fixed_cut_fields.get(selection, (selection,))
        for key in keys:
            before, after = previous.fields.get(key), current.fields.get(key)
            if before is None or after is None or before.axes != after.axes:
                return False
            if np.shape(before.data) != np.shape(after.data):
                return False
            for axis in before.axes:
                old = before.coordinates.get(axis, previous.geometry.coord(axis))
                new = after.coordinates.get(axis, current.geometry.coord(axis))
                if not np.array_equal(old, new):
                    return False
        return True

    def _field_changed(self, index: int) -> None:
        if self._run_data is None or index < 0:
            return

        key = self.field_selector.itemData(index)
        if key is None:
            return

        if key in self._fixed_cut_fields:
            self._configure_fixed_cuts()
            return
        field = self._run_data.fields[key]
        data = np.asarray(field.data)
        if data.ndim != 3:
            return

        _, nx, ny = data.shape

        self.field_selector_label.setText("3-D field")
        self.x_cut_slider.show()
        self.y_cut_slider.show()
        self.show_guides.show()

        self._iz = data.shape[0] // 2

        self._current_vmin, self._current_vmax = volume_limits(data)

        self._show_guides = self.show_guides.isChecked()
        self.guidesVisibilityChanged.emit(self._show_guides)

        self.x_cut_slider.blockSignals(True)
        self.y_cut_slider.blockSignals(True)
        self.z_plane_slider.blockSignals(True)
        self.x_cut_slider.setRange(0, nx - 1)
        self.y_cut_slider.setRange(0, ny - 1)
        self.z_plane_slider.setRange(0, data.shape[0] - 1)
        self.z_plane_slider.setValue(self._iz)
        self.x_cut_slider.blockSignals(False)
        self.y_cut_slider.blockSignals(False)
        self.z_plane_slider.blockSignals(False)

        self.set_cut_indices(
            self._nearest_index(field, "x", 0.0, nx),
            self._nearest_index(field, "y", 0.0, ny),
            emit=False,
        )
        self.volumeSelectionChanged.emit(str(key))
        self.cutChanged.emit(self._ix, self._iy)
        self.zPlaneChanged.emit(self._iz)

    def _cut_changed(self, _value: int) -> None:
        if self._updating_cuts:
            return
        self._ix = self.x_cut_slider.value()
        self._iy = self.y_cut_slider.value()
        self._update_views()
        self.cutChanged.emit(self._ix, self._iy)

    def _z_changed(self, value: int) -> None:
        if self._updating_cuts:
            return
        self._iz = int(value)
        self._update_views()
        self.zPlaneChanged.emit(self._iz)

    def _set_z_slider(self, value: int) -> None:
        self.z_plane_slider.blockSignals(True)
        self.z_plane_slider.setValue(int(value))
        self.z_plane_slider.blockSignals(False)

    def set_cut_indices(self, ix: int, iy: int, *, emit: bool = True) -> None:
        """Set longitudinal cut indices in LCProp field order: x index, y index."""
        if self._is_fixed_cut_selection():
            self._update_fixed_cut_views()
            return
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

        if key in self._fixed_cut_fields:
            self._update_fixed_cut_views()
            return

        field = self._run_data.fields[key]
        data = np.asarray(field.data)
        if data.ndim != 3:
            return

        _, nx, ny = data.shape
        ix = min(max(int(self._ix), 0), nx - 1)
        iy = min(max(int(self._iy), 0), ny - 1)
        self._iz = min(max(int(self._iz), 0), data.shape[0] - 1)

        x_value = self._coord_value(field, "x", ix)
        y_value = self._coord_value(field, "y", iy)
        z_value = self._coord_value(field, "z", self._iz)

        self.y_cut_label.setText(f"x-z cut at y = {format_number(y_value, quantity='coordinate')} µm")
        self.x_cut_label.setText(f"y-z cut at x = {format_number(x_value, quantity='coordinate')} µm")
        self.z_plane_label.setText(f"z = {format_number(z_value, quantity='coordinate')} µm")
        self.position_label.setText(
            f"x={format_number(x_value, quantity='coordinate')}, y={format_number(y_value, quantity='coordinate')}, z={format_number(z_value, quantity='coordinate')} µm"
        )

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
            coordinates={
                "z": self._coordinate_array(field, "z", data.shape[0]),
                "x": self._coordinate_array(field, "x", nx),
            },
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
            coordinates={
                "z": self._coordinate_array(field, "z", data.shape[0]),
                "y": self._coordinate_array(field, "y", ny),
            },
        )

        limits = self.scales.limits(scale_key(field), (self._current_vmin, self._current_vmax))
        self.scale_controls.show_scale(scale_key(field), limits)
        self.xz_view.set_field(
            xz_field,
            extent=self._field_extent(field, "z", "x"),
            vmin=limits[0],
            vmax=limits[1],
        )
        self.yz_view.set_field(
            yz_field,
            extent=self._field_extent(field, "z", "y"),
            vmin=limits[0],
            vmax=limits[1],
        )
        self._apply_guides()

    def _is_fixed_cut_selection(self) -> bool:
        return self.field_selector.currentData() in getattr(self, "_fixed_cut_fields", {})

    def _configure_fixed_cuts(self) -> None:
        xz_key, yz_key = self._fixed_cut_fields[self.field_selector.currentData()]
        xz_field = self._run_data.fields[xz_key]
        yz_field = self._run_data.fields[yz_key]
        xz = np.asarray(xz_field.data)
        yz = np.asarray(yz_field.data)
        if xz.ndim != 2 or yz.ndim != 2 or xz.shape[0] != yz.shape[0]:
            return
        self._iz = xz.shape[0] // 2
        self.z_plane_slider.blockSignals(True)
        self.z_plane_slider.setRange(0, xz.shape[0] - 1)
        self.z_plane_slider.setValue(self._iz)
        self.z_plane_slider.blockSignals(False)
        self._current_vmin, self._current_vmax = volume_limits(xz, yz)
        coordinates = xz_field.coordinates
        x_cut_um = float(coordinates["x_cut_um"])
        y_cut_um = float(coordinates["y_cut_um"])
        self.field_selector_label.setText("Longitudinal field")
        prefix = "Retained Fast" if self.field_selector.currentData() == self._RETAINED_FAST_CUTS else "Sampled fixed"
        self.y_cut_label.setText(
            f"{prefix} x-z cut at y = {format_number(y_cut_um, quantity='coordinate')} µm"
        )
        self.x_cut_label.setText(
            f"{prefix} y-z cut at x = {format_number(x_cut_um, quantity='coordinate')} µm"
        )
        self.x_cut_slider.hide()
        self.y_cut_slider.hide()
        self.show_guides.show()
        self.position_label.setText(
            f"{prefix} cuts: x={format_number(x_cut_um, quantity='coordinate')}, y={format_number(y_cut_um, quantity='coordinate')} µm"
        )
        self._update_fixed_cut_views()

    def _update_fixed_cut_views(self) -> None:
        xz_key, yz_key = self._fixed_cut_fields[self.field_selector.currentData()]
        xz_field = self._run_data.fields[xz_key]
        yz_field = self._run_data.fields[yz_key]
        z_value = self._coord_value(xz_field, "z", self._iz)
        self.z_plane_label.setText(
            f"z sample = {format_number(z_value, quantity='coordinate')} µm (fixed cuts)"
        )
        limits = self.scales.limits(scale_key(xz_field), (self._current_vmin, self._current_vmax))
        self.scale_controls.show_scale(scale_key(xz_field), limits)
        self.xz_view.set_field(
            xz_field,
            extent=self._field_extent(xz_field, "z", "x"),
            vmin=limits[0],
            vmax=limits[1],
        )
        self.yz_view.set_field(
            yz_field,
            extent=self._field_extent(yz_field, "z", "y"),
            vmin=limits[0],
            vmax=limits[1],
        )
        self._apply_guides()
        self.guidesChanged.emit()

    def _apply_guides(self) -> None:
        if self._is_fixed_cut_selection():
            if self._show_guides:
                x, y, z = self.guide_coordinates()
                self.xz_view.set_crosshair_coordinates(z, x)
                self.yz_view.set_crosshair_coordinates(z, y)
            else:
                self.xz_view.clear_crosshair()
                self.yz_view.clear_crosshair()
            return
        if self._show_guides:
            self.xz_view.set_crosshair(self._iz, self._ix)
            self.yz_view.set_crosshair(self._iz, self._iy)
        else:
            self.xz_view.clear_crosshair()
            self.yz_view.clear_crosshair()

    def guide_coordinates(self):
        key = self.field_selector.currentData()
        if self._run_data is None or key is None:
            return None
        if self._is_fixed_cut_selection():
            field = self._run_data.fields[self._fixed_cut_fields[key][0]]
            return (float(field.coordinates["x_cut_um"]), float(field.coordinates["y_cut_um"]),
                    self._coord_value(field, "z", self._iz))
        field = self._run_data.fields[key]
        return tuple(self._coord_value(field, axis, i)
                     for axis, i in zip(("x", "y", "z"), (self._ix, self._iy, self._iz)))

    def select_volume(self, key: str) -> None:
        index = self.field_selector.findData(key)
        if index >= 0 and index != self.field_selector.currentIndex():
            self.field_selector.setCurrentIndex(index)

    def _coord_value(self, field, axis: str, index: int) -> float:
        if self._run_data is None:
            return float(index)
        coordinates = getattr(field, "coordinates", {}) or {}
        values = coordinates.get(axis)
        if values is not None:
            values = np.asarray(values)
            if 0 <= index < values.size:
                return float(values[index])
        return self._run_data.geometry.value(axis, index)

    def _field_extent(self, field, horizontal: str, vertical: str):
        h = np.asarray((getattr(field, "coordinates", {}) or {}).get(
            horizontal, []
        ))
        v = np.asarray((getattr(field, "coordinates", {}) or {}).get(
            vertical, []
        ))
        if h.size and v.size:
            return [float(h[0]), float(h[-1]), float(v[0]), float(v[-1])]
        return self._run_data.geometry.extent(horizontal, vertical)

    def _nearest_index(self, field, axis: str, value: float, size: int) -> int:
        coordinates = (getattr(field, "coordinates", {}) or {}).get(axis)
        if coordinates is not None:
            values = np.asarray(coordinates)
            if values.size == size:
                return int(np.argmin(np.abs(values - value)))
        return min(max(self._run_data.geometry.nearest_index(axis, value), 0), size - 1)

    def _coordinate_array(self, field, axis: str, size: int) -> np.ndarray:
        coordinates = (getattr(field, "coordinates", {}) or {}).get(axis)
        if coordinates is not None:
            values = np.asarray(coordinates)
            if values.shape == (size,):
                return values
        values = self._run_data.geometry.coord(axis)
        if values is not None and values.shape == (size,):
            return values
        return np.arange(size, dtype=float)
