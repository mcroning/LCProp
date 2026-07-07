from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget

from lcprop.gui.views import ImagePane, LongitudinalPane, CurvePane


class Workspace(QWidget):
    """Workflow-independent results workspace."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.image_pane = ImagePane()
        self.tabs.addTab(self.image_pane, "Images")

        self.longitudinal_pane = LongitudinalPane()
        self.tabs.addTab(self.longitudinal_pane, "Longitudinal")

        self.curve_pane = CurvePane()
        self.tabs.addTab(self.curve_pane, "Curves")

        self.diagnostics_view = QTextEdit()
        self.diagnostics_view.setReadOnly(True)
        self.tabs.addTab(self.diagnostics_view, "Diagnostics")

        self.request_summary = QTextEdit()
        self.request_summary.setReadOnly(True)
        self.tabs.addTab(self.request_summary, "Request")

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.tabs.addTab(self.console, "Console")

    def set_request_summary(self, text: str) -> None:
        self.request_summary.setPlainText(text)

    def append_console(self, text: str) -> None:
        self.console.append(text)

    def set_run_data(self, run_data) -> None:
        self.image_pane.set_run_data(run_data)
        self.longitudinal_pane.set_run_data(run_data)
        self.curve_pane.set_run_data(run_data)

        if self.curve_pane.curve_selector.count() > 0 and not run_data.fields:
            self.tabs.setCurrentWidget(self.curve_pane)
        elif self.longitudinal_pane.field_selector.count() > 0:
            self.tabs.setCurrentWidget(self.longitudinal_pane)
        else:
            self.tabs.setCurrentWidget(self.image_pane)

        self.diagnostics_view.setPlainText(self._format_diagnostics(run_data))

    def _format_diagnostics(self, run_data) -> str:
        lines: list[str] = [f"Workflow: {run_data.workflow}"]

        for name, diagnostic in run_data.diagnostics.items():
            lines.append("")
            lines.append(f"[{name}] {diagnostic.display_name}")
            for key, value in diagnostic.values.items():
                if key == "rows":
                    lines.append(f"{key}: {len(value)} rows")
                else:
                    lines.append(f"{key}: {value}")

        if run_data.curves:
            lines.append("")
            lines.append("Curves:")
            for key, curve in run_data.curves.items():
                lines.append(f"  {key}: {curve.display_name}")

        if run_data.fields:
            lines.append("")
            lines.append("Fields:")
            for key, field in run_data.fields.items():
                shape = getattr(field.data, "shape", None)
                lines.append(f"  {key}: {field.display_name}, shape={shape}")

        return "\n".join(lines)
