from __future__ import annotations
from lcprop.gui.number_format import format_number
from lcprop.gui.scientific_labels import scientific_text, axis_label

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class CurveView(FigureCanvasQTAgg):
    """Generic Qt/Matplotlib view for CurveData."""

    def __init__(self):
        self.figure = Figure(figsize=(5, 4))
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.line = None
        self.lines = []

    def set_curve(self, curve, title: str | None = None, *, tolerances=()) -> None:
        x = np.asarray(curve.x)
        y = np.asarray(curve.y)

        self.ax.clear()
        self.lines = list(self.ax.plot(x, y, marker="o"))
        self.line = self.lines[0] if self.lines else None
        series_labels = tuple(getattr(curve, "series_labels", ()))
        if series_labels and len(series_labels) == len(self.lines):
            for line, label in zip(self.lines, series_labels):
                line.set_label(label)
            self.ax.legend()

        self.ax.set_title(scientific_text(title if title is not None else curve.display_name))
        self.ax.set_xlabel(_label_with_unit(curve.x_label, curve.units))
        self.ax.set_ylabel(_label_with_unit(curve.y_label, curve.units))
        requested_scale = getattr(curve, "y_scale", "linear")
        if requested_scale == "log" and y.size > 0 and np.all(y > 0.0):
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")
        if curve.key == "overlap_abs":
            self.ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        for tolerance in tolerances:
            self.ax.axhline(tolerance, color="tab:red", linestyle="--",
                            label=f"Strict tolerance < {format_number(tolerance, quantity='tolerance')}")
        if tolerances:
            self.ax.legend()
        self.ax.grid(True)

        self.figure.tight_layout()
        self.draw_idle()


def _label_with_unit(label: str, units: dict[str, str]) -> str:
    return axis_label(label, units)
