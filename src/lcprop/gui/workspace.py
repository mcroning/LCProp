from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTextEdit, QSplitter, QTabWidget, QVBoxLayout, QWidget

from lcprop.gui.views import ImagePane, LongitudinalPane, CurvePane


class Workspace(QWidget):
    """Workflow-independent results workspace."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.image_pane = ImagePane()
        self.longitudinal_pane = LongitudinalPane()

        self.fields_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.fields_splitter.addWidget(self.image_pane)
        self.fields_splitter.addWidget(self.longitudinal_pane)
        self.fields_splitter.setStretchFactor(0, 3)
        self.fields_splitter.setStretchFactor(1, 4)
        self.fields_splitter.setSizes([420, 560])
        self.tabs.addTab(self.fields_splitter, "Fields")

        self.image_pane.positionSelected.connect(
            self._image_position_selected
        )
        self.longitudinal_pane.cutChanged.connect(
            self._longitudinal_cut_changed
        )
        self.longitudinal_pane.guidesVisibilityChanged.connect(
            self._guides_visibility_changed
        )
        self.longitudinal_pane.zPlaneChanged.connect(
            self.image_pane.set_z_index
        )

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

    def set_td_time_indicator(self, text: str | None) -> None:
        self.image_pane.set_td_time_indicator(text)

    def reset_field_color_scales(self) -> None:
        self.image_pane.reset_color_scales()

    def _image_position_selected(self, ix: int, iy: int) -> None:
        self.longitudinal_pane.set_cut_indices(ix, iy)

    def _longitudinal_cut_changed(self, ix: int, iy: int) -> None:
        self.image_pane.set_crosshair(ix, iy)

    def _guides_visibility_changed(self, visible: bool) -> None:
        if visible:
            self.image_pane.set_crosshair(
                self.longitudinal_pane.x_cut_slider.value(),
                self.longitudinal_pane.y_cut_slider.value(),
            )
        else:
            self.image_pane.clear_crosshair()

    def set_run_data(self, run_data) -> None:
        if run_data.workflow == "timedependent":
            summary = run_data.diagnostics.get("summary")
            values = {} if summary is None else summary.values
            cumulative_time = values.get("cumulative_time")
            if cumulative_time is not None:
                prefix = (
                    "TD time at stop: "
                    if values.get("status") == "cancelled"
                    else "Final TD time: "
                )
                self.set_td_time_indicator(
                    prefix + f"{float(cumulative_time):.3f}"
                )
            else:
                self.set_td_time_indicator(None)
        elif run_data.workflow == "static":
            summary = run_data.diagnostics.get("summary")
            values = {} if summary is None else summary.values
            coordinate = values.get("z_reached_um")
            if coordinate is None:
                self.set_td_time_indicator(None)
            else:
                prefix = (
                    "z at stop: "
                    if values.get("status") == "stopped"
                    else "Final z: "
                )
                completed = values.get("completed_slices")
                total = values.get("total_slices")
                self.set_td_time_indicator(
                    prefix
                    + f"{float(coordinate):.3f} um; slices: {completed}/{total}"
                )
        else:
            self.set_td_time_indicator(None)

        self.image_pane.set_run_data(run_data)
        self.longitudinal_pane.set_run_data(run_data)
        self.curve_pane.set_run_data(run_data)

        if run_data.fields:
            self.tabs.setCurrentWidget(self.fields_splitter)
        elif self.curve_pane.curve_selector.count() > 0:
            self.tabs.setCurrentWidget(self.curve_pane)

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
