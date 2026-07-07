from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ImageView(FigureCanvasQTAgg):
    """Generic Qt/Matplotlib view for a 2-D FieldData object."""

    def __init__(self):
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.image = None
        self.colorbar = None

    def set_field(self, field) -> None:
        data = np.asarray(field.data)

        if data.ndim != 2:
            raise ValueError(f"ImageView requires 2-D data, got shape {data.shape}")

        self.ax.clear()

        self.image = self.ax.imshow(
            data,
            origin="lower",
            aspect="auto",
        )

        self.ax.set_title(field.display_name)

        if len(field.axes) >= 2:
            self.ax.set_xlabel(field.axes[0])
            self.ax.set_ylabel(field.axes[1])

        if self.colorbar is not None:
            self.colorbar.remove()

        self.colorbar = self.figure.colorbar(self.image, ax=self.ax)
        self.figure.tight_layout()
        self.draw_idle()
