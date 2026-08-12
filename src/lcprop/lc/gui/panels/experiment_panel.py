"""Liquid-crystal experiment-selection panel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget


class ExperimentPanel(QWidget):
    experimentChanged = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.experiment = QComboBox()
        self.experiment.addItems([
            "Static propagation",
            "Time-dependent propagation",
            "Soliton",
            "Soliton existence curve",
        ])
        self.experiment.currentTextChanged.connect(self.experimentChanged.emit)

        form = QFormLayout()
        form.addRow("Experiment", self.experiment)

        layout.addLayout(form)
        layout.addWidget(QLabel("Static and time-dependent propagation are wired to LocalRunner."))
        layout.addStretch(1)

    def current_experiment(self) -> str:
        return self.experiment.currentText()

    def set_current_experiment(self, name: str) -> None:
        self.experiment.setCurrentText(name)
