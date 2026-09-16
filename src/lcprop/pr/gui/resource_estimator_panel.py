"""PR run-planning presentation for transparent resource estimates."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class PRResourceEstimatorPanel(QWidget):
    """Display an on-demand estimate without changing the scientific request."""

    estimateRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Planning estimates are broad ranges from versioned measurements and "
            "explicit scaling formulas. Queue and transfer time are not included."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.estimate_button = QPushButton("Estimate current configuration")
        self.estimate_button.clicked.connect(self.estimateRequested)
        layout.addWidget(self.estimate_button)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlainText("No estimate yet.")
        layout.addWidget(self.output, 1)
        caveat = QLabel(
            "An estimate is not a convergence claim. Grid, timestep, coupled-pass, "
            "Newton, and PCG settings remain explicit scientific choices."
        )
        caveat.setWordWrap(True)
        layout.addWidget(caveat)

    def set_estimate(self, text: str) -> None:
        self.output.setPlainText(text)

    def set_error(self, message: str) -> None:
        self.output.setPlainText(f"Estimate unavailable: {message}")

    def mark_stale(self) -> None:
        text = self.output.toPlainText()
        if text not in ("No estimate yet.", "Estimate is stale; recalculate."):
            self.output.setPlainText("Estimate is stale; recalculate.")


__all__ = ["PRResourceEstimatorPanel"]
