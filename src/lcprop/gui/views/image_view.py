from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSizePolicy


class ImageView(FigureCanvasQTAgg):
    """Generic Qt/Matplotlib view for a 2-D FieldData object.

    LCProp stores 2-D fields as data[x, y]. Matplotlib displays images as
    image[row, column] = image[y, x], so ImageView owns the display transpose
    and translates mouse clicks back to LCProp indices.
    """

    positionSelected = Signal(int, int)

    def __init__(self, *, compact_vertical: bool = False):
        self.figure = Figure(figsize=(5, 5))
        super().__init__(self.figure)
        self.setMinimumSize(360, 300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._field = None
        self._raw_shape = None
        self._extent = None
        if compact_vertical:
            # Longitudinal views share the available height. Raise the axes so
            # tick labels and the z-axis label stay inside a short canvas.
            self._axes_rect = (0.16, 0.23, 0.67, 0.67)
            self._colorbar_rect = (0.87, 0.23, 0.035, 0.67)
        else:
            # The transverse pane is narrower than the longitudinal pane.
            # Reserve a wider right gutter so colorbar ticks and units are not
            # clipped at the minimum application width.
            self._axes_rect = (0.13, 0.14, 0.63, 0.76)
            self._colorbar_rect = (0.81, 0.14, 0.035, 0.76)
        self.ax = self.figure.add_axes(self._axes_rect)
        colorbar_ax = self.figure.add_axes(self._colorbar_rect)
        self.image = self.ax.imshow(
            np.zeros((2, 2)),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        self.colorbar = self.figure.colorbar(self.image, cax=colorbar_ax)
        self.ax.tick_params(axis="both", labelsize=8)
        self.ax.xaxis.label.set_size(9)
        self.ax.yaxis.label.set_size(9)
        self.colorbar.ax.tick_params(labelsize=8)
        self._vline = None
        self._hline = None
        self._crosshair_index = None
        self._button_press_cid = self.mpl_connect("button_press_event", self._on_mouse_press)

    def set_field(self, field, *, extent=None, vmin=None, vmax=None) -> None:
        raw = np.asarray(field.data)

        if raw.ndim != 2:
            raise ValueError(f"ImageView requires 2-D data, got shape {raw.shape}")

        # LCProp field convention is data[x, y]. Matplotlib imshow expects
        # image[row, column] = image[y, x], so transpose at the display boundary.
        data = raw.T

        self._field = field
        self._raw_shape = raw.shape
        self._extent = extent

        # Figure, axes, image, and colorbar are created once. Live updates only
        # replace artist data and text, preserving the fixed axes rectangles.
        resolved_extent = (
            (-0.5, raw.shape[0] - 0.5, -0.5, raw.shape[1] - 0.5)
            if extent is None
            else tuple(float(value) for value in extent)
        )
        display_extent = _nondegenerate_extent(resolved_extent)
        self.image.set_data(data)
        self.image.set_extent(display_extent)
        self.image.set_cmap(getattr(field, "colormap", "viridis"))
        self.image.set_clim(vmin=vmin, vmax=vmax)
        self.ax.set_aspect(_image_aspect(field))
        self.ax.set_xlim(display_extent[0], display_extent[1])
        self.ax.set_ylim(display_extent[2], display_extent[3])

        self.ax.set_title(field.display_name, fontsize=10, pad=4)

        if len(field.axes) >= 2:
            self.ax.set_xlabel(_label_with_unit(field.axes[0], field.units))
            self.ax.set_ylabel(_label_with_unit(field.axes[1], field.units))
        else:
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")

        self.colorbar.update_normal(self.image)
        value_unit = getattr(field, "value_unit", "")
        self.colorbar.set_label(value_unit, fontsize=9)

        if self._crosshair_index is not None:
            ix, iy = self._crosshair_index
            self.set_crosshair(ix, iy, emit=False)

        self.draw_idle()

    def set_crosshair(self, ix: int, iy: int, *, emit: bool = False) -> None:
        """Move the crosshair to LCProp data indices (ix, iy)."""
        if self._raw_shape is None or self.ax is None:
            self._crosshair_index = (int(ix), int(iy))
            return

        nx, ny = self._raw_shape
        ix = min(max(int(ix), 0), nx - 1)
        iy = min(max(int(iy), 0), ny - 1)
        self._crosshair_index = (ix, iy)

        x_value, y_value = self._index_to_display_coordinates(ix, iy)

        if self._vline is None or self._hline is None:
            self._vline = self.ax.axvline(x_value, linewidth=1)
            self._hline = self.ax.axhline(y_value, linewidth=1)
        else:
            self._vline.set_xdata([x_value, x_value])
            self._hline.set_ydata([y_value, y_value])

        self.draw_idle()

        if emit:
            self.positionSelected.emit(ix, iy)

    def clear_crosshair(self) -> None:
        self._crosshair_index = None
        if self._vline is not None:
            self._vline.remove()
        if self._hline is not None:
            self._hline.remove()
        self._vline = None
        self._hline = None
        self.draw_idle()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._raw_shape is None:
            return

        ix, iy = self._display_coordinates_to_index(float(event.xdata), float(event.ydata))
        self.set_crosshair(ix, iy, emit=True)

    def _index_to_display_coordinates(self, ix: int, iy: int) -> tuple[float, float]:
        nx, ny = self._raw_shape
        if self._extent is None:
            return float(ix), float(iy)

        xmin, xmax, ymin, ymax = [float(v) for v in self._extent]
        x = _index_to_extent_value(ix, nx, xmin, xmax)
        y = _index_to_extent_value(iy, ny, ymin, ymax)
        return x, y

    def _display_coordinates_to_index(self, x_value: float, y_value: float) -> tuple[int, int]:
        nx, ny = self._raw_shape
        if self._extent is None:
            ix = int(round(x_value))
            iy = int(round(y_value))
        else:
            xmin, xmax, ymin, ymax = [float(v) for v in self._extent]
            ix = _extent_value_to_index(x_value, nx, xmin, xmax)
            iy = _extent_value_to_index(y_value, ny, ymin, ymax)

        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        return ix, iy


def _index_to_extent_value(index: int, n: int, lo: float, hi: float) -> float:
    if n <= 1:
        return 0.5 * (lo + hi)
    return lo + (hi - lo) * (float(index) / float(n - 1))


def _nondegenerate_extent(extent) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = extent
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    return xmin, xmax, ymin, ymax


def _extent_value_to_index(value: float, n: int, lo: float, hi: float) -> int:
    if n <= 1 or hi == lo:
        return 0
    frac = (float(value) - lo) / (hi - lo)
    return int(round(frac * float(n - 1)))


def _image_aspect(field) -> str:
    """Use a fixed physical aspect ratio for transverse x-y images."""
    axes = tuple(getattr(field, "axes", ()))
    return "equal" if axes == ("x", "y") else "auto"


def _label_with_unit(label: str, units: dict[str, str]) -> str:
    unit = units.get(label)
    if unit:
        return f"{label} ({unit})"
    return label
