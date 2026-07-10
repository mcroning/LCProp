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
        self.execution.addItem("Parallel", "parallel")
        self.continuation.toggled.connect(self._continuation_changed)

        form.addRow("Sweep parameter", self.parameter)
        form.addRow("Values", self.values)
        form.addRow("", self.continuation)
        form.addRow("Execution", self.execution)

        layout.addLayout(form)
        layout.addStretch(1)
        self._continuation_changed(self.continuation.isChecked())

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

    def _continuation_changed(self, checked: bool) -> None:
        """Parallel sweeps are only valid for independent, non-continuation runs."""
        parallel_index = self.execution.findData("parallel")
        if parallel_index < 0:
            return
        parallel_item = self.execution.model().item(parallel_index)
        parallel_item.setEnabled(not checked)
        if checked and self.execution.currentData() == "parallel":
            self.execution.setCurrentIndex(self.execution.findData("sequential"))
