from __future__ import annotations

from lcprop.gui.runtime_status import RuntimeStatusLabel

from dataclasses import replace

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lcprop.gui.views.image_view import ImageView
from lcprop.gui.views.display_scale import DisplayScales, DisplayScaleControls, scale_key


ROBUST_INTENSITY_PERCENTILE = 99.5


def display_limits(field) -> tuple[float, float]:
    """Return finite display-only limits without changing stored field data."""

    data = np.asarray(field.data)
    finite = data[np.isfinite(data)]
    if str(getattr(field, "kind", "field")) in {
        "intensity",
        "intensity_preview",
    }:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return 0.0, 1.0
        # Ignore a tiny hot-pixel tail so the beam body remains visible.
        vmax = float(np.percentile(positive, ROBUST_INTENSITY_PERCENTILE))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.max(positive))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        return 0.0, vmax

    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        padding = max(abs(vmin) * 1e-12, 1e-15)
        vmin -= padding
        vmax += padding
    return vmin, vmax


class ImagePane(QWidget):
    """Field browser for 2-D image fields."""

    positionSelected = Signal(int, int)
    physicalPositionSelected = Signal(float, float)
    sourceVolumeSelected = Signal(str)
    fieldChanged = Signal()

    def __init__(self, scales=None):
        super().__init__()
        self.scales = scales if scales is not None else DisplayScales()
        self._run_data = None
        self._z_index = None
        self._scale_limits: dict[str, tuple[float, float]] = {}

        layout = QVBoxLayout(self)

        self.td_time_label = RuntimeStatusLabel()
        time_policy = self.td_time_label.sizePolicy()
        time_policy.setRetainSizeWhenHidden(True)
        self.td_time_label.setSizePolicy(time_policy)
        self.td_time_label.setVisible(False)
        layout.addWidget(self.td_time_label)

        self.field_selector = QComboBox()
        self.field_selector.setMinimumWidth(285)
        self.field_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.field_selector.setMinimumContentsLength(32)
        self.field_selector.currentIndexChanged.connect(self._field_changed)
        layout.addWidget(self.field_selector)

        view_controls = QHBoxLayout()
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Restore this product's recommended view")
        self.full_aperture_button = QPushButton("Full Aperture")
        self.full_aperture_button.setToolTip("Show the complete field extent")
        self.fit_button.clicked.connect(self.image_view_fit)
        self.full_aperture_button.clicked.connect(self.image_view_full_aperture)
        view_controls.addWidget(self.fit_button)
        view_controls.addWidget(self.full_aperture_button)
        view_controls.addStretch(1)
        layout.addLayout(view_controls)

        self.scale_controls = DisplayScaleControls(self.scales)
        layout.addWidget(self.scale_controls)
        self.scales.changed.connect(self._refresh_scale)
        self.image_view = ImageView()
        self.image_view.positionSelected.connect(self._position_selected)
        layout.addWidget(self.image_view)

    def _refresh_scale(self):
        self._field_changed(self.field_selector.currentIndex())

    def image_view_fit(self) -> None:
        self.image_view.fit_default()

    def image_view_full_aperture(self) -> None:
        self.image_view.fit_full_aperture()

    def set_run_data(self, run_data) -> None:
        previous_key = self.field_selector.currentData() or getattr(self, "_preferred_key", None)
        self._preferred_key = previous_key
        self._run_data = run_data
        self._z_index = None
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        for key, field in run_data.fields.items():
            if (getattr(field.data, "ndim", None) == 2
                    and not field.coordinates.get("paired_cut_key")):
                self.field_selector.addItem(field.display_name, key)

        self.field_selector.blockSignals(False)

        if not self.field_selector.count():
            self.image_view.clear_field()
        self.image_view.setVisible(self.field_selector.count() > 0)
        self.scale_controls.setEnabled(self.field_selector.count() > 0)
        if self.field_selector.count() > 0:
            default_index = 0
            preferred_found = False
            for index in range(self.field_selector.count()):
                key = self.field_selector.itemData(index)
                if bool(getattr(run_data.fields[key], "initially_selected", False)):
                    default_index = index
                    preferred_found = True
            if (
                not preferred_found
                and run_data.workflow in {"static", "timedependent"}
            ):
                final_index = self.field_selector.findData("final_intensity")
                if final_index >= 0:
                    default_index = final_index
            previous_index = self.field_selector.findData(previous_key)
            if previous_index >= 0:
                default_index = previous_index
            self.field_selector.setCurrentIndex(default_index)
            self._field_changed(default_index)

    def set_td_time_indicator(self, text: str | None) -> None:
        self.td_time_label.setText("" if text is None else text)
        self.td_time_label.setVisible(text is not None)

    def _field_changed(self, index: int) -> None:
        if self._run_data is None or index < 0:
            return

        key = self.field_selector.itemData(index)
        if key is None:
            return

        field = self._field_at_selected_z(self._run_data.fields[key])
        extent = self._field_extent(field)
        vmin, vmax = self._limits_for_field(field)
        self.image_view.set_field(
            field,
            extent=extent,
            default_display_extent=getattr(
                field, "default_display_extent", None
            ),
            vmin=vmin,
            vmax=vmax,
        )
        source_key = getattr(self._run_data.fields[key], "source_volume_key", None)
        if source_key is not None:
            self.sourceVolumeSelected.emit(str(source_key))
        self.fieldChanged.emit()

    def _field_extent(self, field):
        if len(field.axes) != 2:
            return None
        horizontal, vertical = field.axes
        coordinates = getattr(field, "coordinates", {}) or {}
        preview = coordinates.get("preview_metadata", {})
        if "original_extent" in preview:
            return preview["original_extent"]
        h = coordinates.get(horizontal)
        v = coordinates.get(vertical)
        if h is not None and v is not None:
            h = np.asarray(h)
            v = np.asarray(v)
            if h.size and v.size:
                return [float(h[0]), float(h[-1]), float(v[0]), float(v[-1])]
        if horizontal in ("x", "y", "z") and vertical in ("x", "y", "z"):
            return self._run_data.geometry.extent(horizontal, vertical)
        return None

    def reset_color_scales(self) -> None:
        """Start deterministic autoscaling for a fresh run."""

        self._scale_limits.clear()
        self._display_limits_cache = {}
        self.scales.reset_locks()

    def _limits_for_field(self, field) -> tuple[float, float]:
        kind = str(getattr(field, "kind", "field"))
        if not hasattr(self, "_display_limits_cache"):
            self._display_limits_cache = {}
        import weakref
        key = field.key
        cached = self._display_limits_cache.get(key)
        semantics = (field.content_revision, field.kind, field.quantity, field.value_unit,
                     ROBUST_INTENSITY_PERCENTILE)
        if cached is None or cached[0]() is not field.data or cached[2] != semantics:
            cached = (weakref.ref(field.data), display_limits(field), semantics)
            self._display_limits_cache[key] = cached
        limits = self.scales.limits(scale_key(field), cached[1])
        self.scale_controls.show_scale(scale_key(field), limits)
        self._scale_limits[kind] = limits
        return limits

    def set_z_index(self, index: int) -> None:
        """Select the z slice used by 2-D fields derived from a volume."""
        self._z_index = int(index)
        self._field_changed(self.field_selector.currentIndex())

    def _field_at_selected_z(self, field):
        source_key = getattr(field, "source_volume_key", None)
        if source_key is None:
            return field
        source = self._run_data.fields[source_key]
        volume = np.asarray(source.data)
        if self._z_index is None:
            self._z_index = volume.shape[0] // 2
        iz = min(max(self._z_index, 0), volume.shape[0] - 1)
        return replace(field, data=volume[iz])

    def select_source_volume(self, source_key: str) -> None:
        """Select the x-y field linked to a chosen MPR source volume."""
        for index in range(self.field_selector.count()):
            key = self.field_selector.itemData(index)
            field = self._run_data.fields[key]
            if getattr(field, "source_volume_key", None) == source_key:
                if index != self.field_selector.currentIndex():
                    self.field_selector.setCurrentIndex(index)
                return

    def set_crosshair(self, ix: int, iy: int) -> None:
        """Move the image crosshair to LCProp (x,y) indices."""
        self.image_view.set_crosshair(ix, iy)

    def clear_crosshair(self) -> None:
        self.image_view.clear_crosshair()

    def _position_selected(self, ix: int, iy: int) -> None:
        if self._run_data is None:
            return
        index = self.field_selector.currentIndex()
        if index < 0:
            return
        key = self.field_selector.itemData(index)
        if key is None:
            return
        field = self._field_at_selected_z(self._run_data.fields[key])
        if field.axes == ("x", "y"):
            self.positionSelected.emit(ix, iy)
            x, y = self.image_view._index_to_display_coordinates(ix, iy)
            self.physicalPositionSelected.emit(x, y)
