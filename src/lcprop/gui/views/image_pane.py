from __future__ import annotations

from dataclasses import replace

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from lcprop.gui.views.image_view import ImageView


ROBUST_INTENSITY_PERCENTILE = 99.5


def display_limits(field) -> tuple[float, float]:
    """Return finite display-only limits without changing stored field data."""

    data = np.asarray(field.data)
    finite = data[np.isfinite(data)]
    if str(getattr(field, "kind", "field")) == "intensity":
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

    def __init__(self):
        super().__init__()
        self._run_data = None
        self._z_index = None
        self._scale_limits: dict[str, tuple[float, float]] = {}

        layout = QVBoxLayout(self)

        self.td_time_label = QLabel()
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

        self.image_view = ImageView()
        self.image_view.positionSelected.connect(self._position_selected)
        layout.addWidget(self.image_view)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
        z = getattr(run_data.geometry, "z", None)
        self._z_index = None if z is None else len(z) // 2
        self.field_selector.blockSignals(True)
        self.field_selector.clear()

        for key, field in run_data.fields.items():
            if getattr(field.data, "ndim", None) == 2:
                self.field_selector.addItem(field.display_name, key)

        self.field_selector.blockSignals(False)

        if self.field_selector.count() > 0:
            default_index = 0
            if run_data.workflow in {"static", "timedependent"}:
                final_index = self.field_selector.findData("final_intensity")
                if final_index >= 0:
                    default_index = final_index
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
        extent = None
        if field.axes == ("x", "y"):
            extent = self._run_data.geometry.extent_xy()
        vmin, vmax = self._limits_for_field(field)
        self.image_view.set_field(
            field,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

    def reset_color_scales(self) -> None:
        """Start deterministic autoscaling for a fresh run."""

        self._scale_limits.clear()

    def _limits_for_field(self, field) -> tuple[float, float]:
        kind = str(getattr(field, "kind", "field"))
        limits = display_limits(field)
        self._scale_limits[kind] = limits
        return limits

    def set_z_index(self, index: int) -> None:
        """Select the z slice used by 2-D fields derived from a volume."""
        self._z_index = int(index)
        self._field_changed(self.field_selector.currentIndex())

    def _field_at_selected_z(self, field):
        source_key = getattr(field, "source_volume_key", None)
        if source_key is None or self._z_index is None:
            return field
        source = self._run_data.fields[source_key]
        volume = np.asarray(source.data)
        iz = min(max(self._z_index, 0), volume.shape[0] - 1)
        return replace(field, data=volume[iz])

    def set_crosshair(self, ix: int, iy: int) -> None:
        """Move the image crosshair to LCProp (x,y) indices."""
        self.image_view.set_crosshair(ix, iy)

    def clear_crosshair(self) -> None:
        self.image_view.clear_crosshair()

    def _position_selected(self, ix: int, iy: int) -> None:
        self.positionSelected.emit(ix, iy)
