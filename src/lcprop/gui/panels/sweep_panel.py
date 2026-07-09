from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QVBoxLayout, QWidget


class SweepPanel(QWidget):
    """Controls for generic parameter sweeps."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.parameter = QComboBox()
        self.parameter.addItem("Power", "power_mW")

        self.values = QLineEdit("0.1, 0.2, 0.5, 1.0, 2.0")
        self.continuation = QCheckBox("Continuation")
        self.continuation.setChecked(True)

        self.execution = QComboBox()
        self.execution.addItem("Sequential", "sequential")

        form.addRow("Sweep parameter", self.parameter)
        form.addRow("Values", self.values)
        form.addRow("", self.continuation)
        form.addRow("Execution", self.execution)

        layout.addLayout(form)
        layout.addStretch(1)

    def sweep_parameter(self) -> str:
        return str(self.parameter.currentData())

    def sweep_values(self) -> tuple[float, ...]:
        text = self.values.text().replace(";", ",")
        values = []
        for part in text.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            values.append(float(stripped))
        if not values:
            raise ValueError("Sweep values must contain at least one number")
        return tuple(values)

    def use_continuation(self) -> bool:
        return bool(self.continuation.isChecked())

    def sweep_execution(self) -> str:
        return str(self.execution.currentData())
