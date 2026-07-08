from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.splitstep import total_intensity



@dataclass(frozen=True)
class Geometry:
    """Physical coordinate vectors for a run."""

    x: Any | None = None
    y: Any | None = None
    z: Any | None = None
    units: str = "um"

    def extent_xy(self):
        if self.x is None or self.y is None:
            return None
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    def extent_zx(self):
        if self.z is None or self.x is None:
            return None
        z = np.asarray(self.z)
        x = np.asarray(self.x)
        return [float(z[0]), float(z[-1]), float(x[0]), float(x[-1])]

    def extent_zy(self):
        if self.z is None or self.y is None:
            return None
        z = np.asarray(self.z)
        y = np.asarray(self.y)
        return [float(z[0]), float(z[-1]), float(y[0]), float(y[-1])]


@dataclass(frozen=True)
class FieldData:
    key: str
    display_name: str
    data: Any
    axes: tuple[str, ...]
    kind: str
    units: dict[str, str] = field(default_factory=dict)
    default_view: str = "image"


@dataclass(frozen=True)
class CurveData:
    key: str
    display_name: str
    x: Any
    y: Any
    x_label: str
    y_label: str
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticData:
    key: str
    display_name: str
    values: dict[str, Any]


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




def _geometry_from_grid_summary(grid_summary: dict) -> Geometry:
    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    nz = int(grid_summary.get("Nz", 1))

    dx = float(grid_summary["dx_um"])
    dy = float(grid_summary["dy_um"])
    dz = float(grid_summary.get("dz_um", 1.0))

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    z = np.arange(nz) * dz

    return Geometry(x=x, y=y, z=z, units="um")


def _intensity_from_A(A, *, coherent: bool = False):
    return asnumpy(total_intensity(A, coherent=coherent))


def from_static_result(result) -> RunData:
    return RunData(
        workflow="static",
        geometry=_geometry_from_grid_summary(result.grid_summary),
        fields=FieldCollection([
            ("intensity", FieldData("intensity", "Intensity", _intensity_from_A(result.A_final), ("x", "y"), "intensity", {"x": "um", "y": "um"})),
            ("theta", FieldData("theta", "Theta", asnumpy(result.theta_final), ("x", "y"), "theta", {"x": "um", "y": "um", "theta": "rad"})),
        ]),
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData(
                "summary",
                "Summary",
                {
                    "power_initial": result.power_initial,
                    "power_final": result.power_final,
                    "method": result.method,
                    "n_steps": result.n_steps,
                    "grid": result.grid_summary,
                },
            ))
        ]),
    )


def from_timedependent_result(result) -> RunData:
    theta_stack = asnumpy(result.theta_final)
    theta_bias = asnumpy(result.theta_bias)
    delta_theta_stack = theta_stack - theta_bias[None, :, :]

    return RunData(
        workflow="timedependent",
        geometry=_geometry_from_grid_summary(result.grid_summary),
        fields=FieldCollection([
            ("delta_theta_stack", FieldData("delta_theta_stack", "Delta theta stack", delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um", "theta": "rad"}, "longitudinal")),
            ("theta_stack", FieldData("theta_stack", "Theta stack", theta_stack, ("z", "x", "y"), "theta", {"z": "um", "x": "um", "y": "um", "theta": "rad"}, "longitudinal")),
            ("final_intensity", FieldData("final_intensity", "Final intensity", _intensity_from_A(result.A_final), ("x", "y"), "intensity", {"x": "um", "y": "um"})),
        ]),
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData(
                "summary",
                "Summary",
                {
                    "power_initial": result.power_initial,
                    "power_final": result.power_final,
                    "Nt": result.Nt,
                    "method": result.method,
                    "grid": result.grid_summary,
                },
            ))
        ]),
    )


def from_soliton_result(result) -> RunData:
    curves = CurveCollection()
    if result.history:
        outer = np.asarray([r["outer"] for r in result.history])
        curves.add("residual_rms", CurveData(
            "residual_rms",
            "Residual RMS",
            outer,
            np.asarray([r.get("residual_rms", np.nan) for r in result.history]),
            "outer",
            "residual_rms",
        ))
        curves.add("beta_history", CurveData(
            "beta_history",
            "Beta history",
            outer,
            np.asarray([r.get("beta", np.nan) for r in result.history]),
            "outer",
            "beta",
        ))

    return RunData(
        workflow="soliton",
        fields=FieldCollection([
            ("intensity", FieldData("intensity", "Intensity", asnumpy(result.intensity), ("x", "y"), "intensity", {"x": "um", "y": "um"})),
            ("theta", FieldData("theta", "Theta", asnumpy(result.theta), ("x", "y"), "theta", {"x": "um", "y": "um", "theta": "rad"})),
        ]),
        curves=curves,
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", dict(result.metrics)))
        ]),
    )


def from_soliton_existence_result(result) -> RunData:
    rows = result.samples
    curves = CurveCollection()

    if rows:
        P = np.asarray([r.get("requested_power_mW", np.nan) for r in rows])
        for key, label in [
            ("beta", "Beta"),
            ("theta_max", "Theta max"),
            ("Imax", "Imax"),
            ("residual_rms", "Residual RMS"),
        ]:
            curves.add(key, CurveData(
                key,
                label,
                P,
                np.asarray([r.get(key, np.nan) for r in rows]),
                "P",
                key,
                {"P": "mW"},
            ))

    return RunData(
        workflow="soliton_existence",
        curves=curves,
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", dict(result.metrics))),
            ("table", DiagnosticData("table", "Samples", {"rows": rows})),
        ]),
    )


def to_run_data(result) -> RunData:
    name = type(result).__name__
    if name == "StaticRunResult":
        return from_static_result(result)
    if name == "TimeDependentRunResult":
        return from_timedependent_result(result)
    if name == "SolitonResult":
        return from_soliton_result(result)
    if name == "SolitonExistenceResult":
        return from_soliton_existence_result(result)
    raise TypeError(f"Unsupported result type: {name}")


__all__ = [
    "FieldData",
    "CurveData",
    "DiagnosticData",
    "FieldCollection",
    "CurveCollection",
    "DiagnosticCollection",
    "Geometry",
    "RunData",
    "from_static_result",
    "from_timedependent_result",
    "from_soliton_result",
    "from_soliton_existence_result",
    "to_run_data",
]
