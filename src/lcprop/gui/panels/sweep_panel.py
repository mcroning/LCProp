from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lcprop.workflows.sweep import DEFAULT_SWEEP_MAX_WORKERS


class SweepPanel(QWidget):
    """Controls for generic parameter sweeps."""

    def __init__(self, *, parallel_available: bool = False):
        super().__init__()
        self._parallel_available = bool(parallel_available)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.parameter = QComboBox()
        self.parameter.addItem("Power", "power_mW")

        self.values = QLineEdit("0.1, 0.2, 0.5, 1.0, 2.0")
        self.continuation = QCheckBox("Continuation")
        self.continuation.setChecked(not self._parallel_available)

        self.execution = QComboBox()
        self.execution.addItem("Sequential", "sequential")
        self.execution.addItem("Parallel", "parallel")
        if self._parallel_available:
            self.execution.setCurrentIndex(self.execution.findData("parallel"))
        self.worker_count = QSpinBox()
        self.worker_count.setRange(1, max(1, int(os.cpu_count() or 1)))
        self.worker_count.setValue(
            min(DEFAULT_SWEEP_MAX_WORKERS, int(os.cpu_count() or 1))
        )
        self.worker_count.setToolTip(
            "Maximum soliton worker processes. Numerical thread pools inside "
            "each process are limited to one thread."
        )
        self.continuation.toggled.connect(self._continuation_changed)
        self.execution.currentIndexChanged.connect(self._execution_changed)

        form.addRow("Sweep parameter", self.parameter)
        form.addRow("Values", self.values)
        form.addRow("", self.continuation)
        form.addRow("Execution", self.execution)
        form.addRow("Worker processes", self.worker_count)

        layout.addLayout(form)
        layout.addStretch(1)
        self._continuation_changed(self.continuation.isChecked())
        self._execution_changed()

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

    def sweep_worker_count(self) -> int:
        return int(self.worker_count.value())

    def _continuation_changed(self, checked: bool) -> None:
        """Parallel sweeps are only valid for independent, non-continuation runs."""
        parallel_index = self.execution.findData("parallel")
        if parallel_index < 0:
            return
        parallel_item = self.execution.model().item(parallel_index)
        parallel_item.setEnabled(self._parallel_available and not checked)
        if checked and self.execution.currentData() == "parallel":
            self.execution.setCurrentIndex(self.execution.findData("sequential"))
        self._execution_changed()

    def _execution_changed(self, *_args) -> None:
        self.worker_count.setEnabled(
            self._parallel_available
            and self.execution.currentData() == "parallel"
        )
