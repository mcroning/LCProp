from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from lcprop.gui.workspace import Workspace


class ResultsPanel(QWidget):
    """Thin panel wrapper around Workspace."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.workspace = Workspace()
        layout.addWidget(self.workspace)

    def set_request_summary(self, text: str) -> None:
        self.workspace.set_request_summary(text)

    def append_console(self, text: str) -> None:
        self.workspace.append_console(text)

    def set_td_time_indicator(self, text: str | None) -> None:
        self.workspace.set_td_time_indicator(text)

    def set_run_data(self, run_data) -> None:
        self.workspace.set_run_data(run_data)

    def reset_field_color_scales(self) -> None:
        self.workspace.reset_field_color_scales()
