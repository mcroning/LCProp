from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from lcprop.gui.views.image_view import ImageView


class ImagePane(QWidget):
    """Field browser for 2-D image fields."""

    def __init__(self):
        super().__init__()
        self._run_data = None

        layout = QVBoxLayout(self)

        self.field_selector = QComboBox()
        self.field_selector.currentIndexChanged.connect(self._field_changed)
        layout.addWidget(self.field_selector)

        self.image_view = ImageView()
        layout.addWidget(self.image_view)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
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

        field = self._run_data.fields[key]
        extent = None
        if field.axes == ("x", "y"):
            extent = self._run_data.geometry.extent_xy()
        self.image_view.set_field(field, extent=extent)
