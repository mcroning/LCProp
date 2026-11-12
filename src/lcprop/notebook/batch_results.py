"""Retained-product slicing without resampling or scientific execution."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    axes: tuple
    coordinates: tuple
    units: tuple
    values: np.ndarray
    caption: str
    value_unit: str
    colormap: str


def field_coordinates(field, geometry, axis):
    size = field.data.shape[field.axes.index(axis)]
    values = field.coordinates.get(axis)
    if values is None and geometry is not None:
        values = geometry.coord(axis)
    if values is None:
        raise ValueError(f"No retained {axis} coordinates")
    values = np.asarray(values)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"No matching retained {axis} coordinates")
    if not size or (size > 1 and not np.all(np.diff(values) > 0)):
        raise ValueError(f"Invalid retained {axis} coordinates")
    unit = field.units.get(axis, geometry.units if geometry is not None else "")
    return values, unit


def retained_plane(field, geometry, axes, index=None):
    """Extract only a plane that exists in the selected retained product."""
    values = np.asarray(field.data)
    order = list(field.axes)
    caption = field.display_name
    if len(order) != values.ndim or len(set(order)) != len(order):
        raise ValueError("Inconsistent retained axes")
    if values.ndim == 3 and set(order) == {"x", "y", "z"}:
        missing, = set(order) - set(axes)
        samples, unit = field_coordinates(field, geometry, missing)
        if index is None:
            index = 0 if missing == "z" else len(samples) // 2
        if not 0 <= index < len(samples):
            raise ValueError("Cut index outside retained coordinates")
        axis_index = order.index(missing)
        values = np.take(values, index, axis=axis_index)
        order.pop(axis_index)
        caption += f" · {missing} = {samples[index]:.6g} {unit}"
    elif values.ndim == 2 and set(order) == set(axes):
        for axis in ("x", "y"):
            coordinate = field.coordinates.get(f"{axis}_cut_um")
            if axis not in axes and coordinate is not None:
                caption += f" · fixed {axis} = {float(coordinate):.6g} µm"
    if values.ndim != 2 or set(order) != set(axes):
        raise ValueError(f"No retained {'–'.join(axes)} plane for this product")
    coordinates, units = zip(*(field_coordinates(field, geometry, axis) for axis in axes))
    values = values.transpose(order.index(axes[1]), order.index(axes[0]))
    return Plane(tuple(axes), coordinates, units, values, caption,
                 field.value_unit, field.colormap)


def draw_plane(ax, plane, square):
    artist = ax.pcolormesh(*plane.coordinates, plane.values, shading="nearest",
                           cmap=plane.colormap)
    ax.set(title=plane.caption, xlabel=f"{plane.axes[0]} ({plane.units[0]})",
           ylabel=f"{plane.axes[1]} ({plane.units[1]})")
    if square:
        ax.set_box_aspect(1)
    else:
        scales = {"m": 1., "mm": 1e-3, "um": 1e-6, "µm": 1e-6,
                  "μm": 1e-6, "nm": 1e-9}
        h, v = plane.units
        if h in scales and v in scales:
            ax.set_aspect(scales[v] / scales[h])
        elif h and h == v:
            ax.set_aspect("equal")
        else:
            raise ValueError("Physical aspect unavailable for these axis units")
    return artist
