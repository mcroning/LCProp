from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ImageView(FigureCanvasQTAgg):
    """Generic Qt/Matplotlib view for a 2-D FieldData object."""

    def __init__(self):
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.ax = None
        self.image = None
        self.colorbar = None

    def set_field(self, field, *, extent=None) -> None:
        raw = np.asarray(field.data)

        if raw.ndim != 2:
            raise ValueError(f"ImageView requires 2-D data, got shape {raw.shape}")

        # LCProp field convention is data[x, y]. Matplotlib imshow expects
        # image[row, column] = image[y, x], so transpose at the display boundary.
        data = raw.T

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        self.image = self.ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=getattr(field, "colormap", "viridis"),
        )

        self.ax.set_title(field.display_name)

        if len(field.axes) >= 2:
            self.ax.set_xlabel(_label_with_unit(field.axes[0], field.units))
            self.ax.set_ylabel(_label_with_unit(field.axes[1], field.units))

        self.colorbar = self.figure.colorbar(self.image, ax=self.ax)
        value_unit = getattr(field, "value_unit", "")
        if value_unit:
            self.colorbar.set_label(value_unit)
        self.figure.tight_layout()
        self.draw_idle()


def _label_with_unit(label: str, units: dict[str, str]) -> str:
    unit = units.get(label)
    if unit:
        return f"{label} ({unit})"
    return label
