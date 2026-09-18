from __future__ import annotations
from lcprop.gui.number_format import format_number

from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

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

        self.tolerance_label = QLabel()
        self.tolerance_label.setWordWrap(True)
        layout.addWidget(self.tolerance_label)
        self.curve_view = CurveView()
        layout.addWidget(self.curve_view)

    def set_run_data(self, run_data) -> None:
        self._run_data = run_data
        self.tolerance_label.clear()
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

        self.curve_view.setVisible(self.curve_selector.count() > 0)
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

        diagnostic = self._run_data.diagnostics.get("convergence_gates")
        gates = [] if diagnostic is None else diagnostic.values.get("rows", [])
        tolerances = sorted({row["tolerance"] for row in gates
                             if row["key"] == key and row["tolerance"] is not None})
        self.tolerance_label.setText(
            "Qualification uses the last completed outer iteration. Strict tolerance(s): "
            + ", ".join(format_number(value, quantity="tolerance") for value in tolerances)
            + ". Inspect exact values and pass/fail in Samples / Tables → Convergence gates."
            if tolerances else ""
        )
        self.curve_view.set_curve(curve, title=title, tolerances=tolerances)
