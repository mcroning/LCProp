from __future__ import annotations

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

    def set_curve(self, curve, title: str | None = None) -> None:
        x = np.asarray(curve.x)
        y = np.asarray(curve.y)

        self.ax.clear()
        (self.line,) = self.ax.plot(x, y, marker="o")

        self.ax.set_title(title if title is not None else curve.display_name)
        self.ax.set_xlabel(_label_with_unit(curve.x_label, curve.units))
        self.ax.set_ylabel(_label_with_unit(curve.y_label, curve.units))
        requested_scale = getattr(curve, "y_scale", "linear")
        if requested_scale == "log" and y.size > 0 and np.all(y > 0.0):
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")
        self.ax.grid(True)

        self.figure.tight_layout()
        self.draw_idle()


def _label_with_unit(label: str, units: dict[str, str]) -> str:
    unit = units.get(label)
    if unit:
        return f"{label} ({unit})"
    return label
