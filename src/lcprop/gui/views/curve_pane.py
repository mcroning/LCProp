from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from lcprop.gui.views.curve_view import CurveView


class CurvePane(QWidget):
    """Curve browser for RunData curves."""

    def __init__(self):
        super().__init__()
        self._run_data = None

        layout = QVBoxLayout(self)

        self.curve_selector = QComboBox()
        self.curve_selector.currentIndexChanged.connect(self._curve_changed)
        layout.addWidget(self.curve_selector)

        self.curve_view = CurveView()
        layout.addWidget(self.curve_view)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
        self._summary = None
        if run_data is not None:
            try:
                self._summary = run_data.diagnostics["summary"].values
            except Exception:
                self._summary = None
        self.curve_selector.blockSignals(True)
        self.curve_selector.clear()

        for key, curve in run_data.curves.items():
            self.curve_selector.addItem(curve.display_name, key)

        self.curve_selector.blockSignals(False)

        if self.curve_selector.count() > 0:
            self.curve_selector.setCurrentIndex(0)
            self._curve_changed(0)

    def _curve_changed(self, index: int) -> None:
        if self._run_data is None or index < 0:
            return

        key = self.curve_selector.itemData(index)
        if key is None:
            return

        curve = self._run_data.curves[key]

        title = curve.display_name
        if self._summary is not None:
            mode = self._summary.get("mode")
            sweep_parameter = self._summary.get("parameter")
            if sweep_parameter == "power_mW":
                sweep_parameter = "Power"
            elif isinstance(sweep_parameter, str):
                sweep_parameter = sweep_parameter.replace("_", " ").title()

            if sweep_parameter:
                title = f"{title} vs {sweep_parameter}"
            if mode:
                title = f"{title} (Mode {mode})"

        self.curve_view.set_curve(curve, title=title)
