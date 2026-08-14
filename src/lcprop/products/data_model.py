"""Material-neutral presentation records and LC compatibility adapters.

The data records and collections defined here are canonical shared APIs. The
lazy LC result adapters retained in ``__all__`` are compatibility exports; new
LC code should import those adapters from :mod:`lcprop.lc.products`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np




@dataclass(frozen=True)
class Geometry:
    """Physical coordinate vectors for a run."""

    x: Any | None = None
    y: Any | None = None
    z: Any | None = None
    units: str = "um"

    def dx(self) -> float | None:
        return self.spacing("x")

    def dy(self) -> float | None:
        return self.spacing("y")

    def dz(self) -> float | None:
        return self.spacing("z")

    def spacing(self, axis: str) -> float | None:
        coord = self.coord(axis)
        if coord is None or coord.size < 2:
            return None
        return float(coord[1] - coord[0])

    def coord(self, axis: str):
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown coordinate axis: {axis}")
        value = getattr(self, axis)
        if value is None:
            return None
        return np.asarray(value)

    def value(self, axis: str, index: int) -> float:
        coord = self.coord(axis)
        if coord is None:
            return float(index)
        if index < 0 or index >= coord.size:
            return float(index)
        return float(coord[index])

    def nearest_index(self, axis: str, value: float) -> int:
        coord = self.coord(axis)
        if coord is None or coord.size == 0:
            return int(round(value))
        return int(np.argmin(np.abs(coord - value)))

    def extent(self, horizontal: str, vertical: str):
        h = self.coord(horizontal)
        v = self.coord(vertical)
        if h is None or v is None:
            return None
        return [float(h[0]), float(h[-1]), float(v[0]), float(v[-1])]

    def extent_xy(self):
        return self.extent("x", "y")

    def extent_zx(self):
        return self.extent("z", "x")

    def extent_zy(self):
        return self.extent("z", "y")


@dataclass(frozen=True)
class FieldData:
    key: str
    display_name: str
    data: Any
    axes: tuple[str, ...]
    kind: str
    units: dict[str, str] = field(default_factory=dict)
    default_view: str = "image"
    quantity: str = ""
    value_unit: str = ""
    colormap: str = "viridis"
    source_volume_key: str | None = None


@dataclass(frozen=True)
class CurveData:
    key: str
    display_name: str
    x: Any
    y: Any
    x_label: str
    y_label: str
    units: dict[str, str] = field(default_factory=dict)
    y_scale: str = "linear"
    series_labels: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DiagnosticData:
    key: str
    display_name: str
    values: dict[str, Any]




def make_field(
    key: str,
    display_name: str,
    data: Any,
    axes: tuple[str, ...],
    kind: str,
    units: dict[str, str] | None = None,
    default_view: str = "image",
    *,
    quantity: str = "",
    value_unit: str = "",
    colormap: str = "viridis",
    source_volume_key: str | None = None,
) -> FieldData:
    return FieldData(
        key=key,
        display_name=display_name,
        data=data,
        axes=axes,
        kind=kind,
        units={} if units is None else units,
        default_view=default_view,
        quantity=quantity,
        value_unit=value_unit,
        colormap=colormap,
        source_volume_key=source_volume_key,
    )


class FieldCollection:
    def __init__(self, items: list[tuple[str, FieldData]] | None = None):
        self._items: OrderedDict[str, FieldData] = OrderedDict(items or [])

    def add(self, key: str, field: FieldData) -> None:
        self._items[key] = field

    def first(self) -> FieldData | None:
        return next(iter(self._items.values()), None)

    def by_kind(self, kind: str) -> list[FieldData]:
        return [field for field in self._items.values() if field.kind == kind]

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def get(self, key: str, default=None):
        return self._items.get(key, default)

    def __getitem__(self, key: str) -> FieldData:
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


class CurveCollection:
    def __init__(self, items: list[tuple[str, CurveData]] | None = None):
        self._items: OrderedDict[str, CurveData] = OrderedDict(items or [])

    def add(self, key: str, curve: CurveData) -> None:
        self._items[key] = curve

    def first(self) -> CurveData | None:
        return next(iter(self._items.values()), None)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def get(self, key: str, default=None):
        return self._items.get(key, default)

    def __getitem__(self, key: str) -> CurveData:
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


class DiagnosticCollection:
    def __init__(self, items: list[tuple[str, DiagnosticData]] | None = None):
        self._items: OrderedDict[str, DiagnosticData] = OrderedDict(items or [])

    def add(self, key: str, diagnostic: DiagnosticData) -> None:
        self._items[key] = diagnostic

    def first(self) -> DiagnosticData | None:
        return next(iter(self._items.values()), None)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def get(self, key: str, default=None):
        return self._items.get(key, default)

    def __getitem__(self, key: str) -> DiagnosticData:
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


@dataclass(frozen=True)
class RunData:
    workflow: str
    geometry: Geometry = field(default_factory=Geometry)
    fields: FieldCollection = field(default_factory=FieldCollection)
    curves: CurveCollection = field(default_factory=CurveCollection)
    diagnostics: DiagnosticCollection = field(default_factory=DiagnosticCollection)
    longitudinal_enabled: bool = True
    longitudinal_message: str | None = None


_LC_COMPAT_EXPORTS = (
    "from_static_result",
    "from_static_live_state",
    "from_timedependent_result",
    "from_timedependent_live_state",
    "from_soliton_result",
    "from_soliton_existence_result",
    "from_parameter_sweep_result",
    "to_run_data",
)


def __getattr__(name: str):
    """Resolve legacy LC adapters without making shared records LC-owned."""
    if name not in _LC_COMPAT_EXPORTS:
        raise AttributeError(name)
    from lcprop.lc import products as lc_products

    return getattr(lc_products, name)


__all__ = [
    "FieldData",
    "CurveData",
    "DiagnosticData",
    "FieldCollection",
    "CurveCollection",
    "DiagnosticCollection",
    "Geometry",
    "RunData",
    "make_field",
    *_LC_COMPAT_EXPORTS,
]
