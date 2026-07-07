from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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

        self.xz_view = ImageView()
        self.yz_view = ImageView()

        layout.addWidget(QLabel("x-z cut at center y"))
        layout.addWidget(self.xz_view)
        layout.addWidget(QLabel("y-z cut at center x"))
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
        ix = nx // 2
        iy = ny // 2

        xz = data[:, :, iy].T
        yz = data[:, ix, :].T

        xz_field = FieldData(
            key=f"{field.key}_xz",
            display_name=f"{field.display_name} x-z",
            data=xz,
            axes=("z", "x"),
            kind=field.kind,
            units=field.units,
            default_view="image",
        )

        yz_field = FieldData(
            key=f"{field.key}_yz",
            display_name=f"{field.display_name} y-z",
            data=yz,
            axes=("z", "y"),
            kind=field.kind,
            units=field.units,
            default_view="image",
        )

        self.xz_view.set_field(xz_field)
        self.yz_view.set_field(yz_field)
