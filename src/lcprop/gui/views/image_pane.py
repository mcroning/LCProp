from __future__ import annotations

from dataclasses import replace

import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from lcprop.gui.views.image_view import ImageView


class ImagePane(QWidget):
    """Field browser for 2-D image fields."""

    positionSelected = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._run_data = None
        self._z_index = None

        layout = QVBoxLayout(self)

        self.field_selector = QComboBox()
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
            self.field_selector.setCurrentIndex(0)
            self._field_changed(0)

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
        self.image_view.set_field(field, extent=extent)

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
