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




def _geometry_from_grid_summary(grid_summary: dict) -> Geometry:
    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    # Some result objects historically omitted/staled Nz; use physical length as a fallback.
    dx = float(grid_summary["dx_um"])
    dy = float(grid_summary["dy_um"])
    dz = float(grid_summary.get("dz_um", 1.0))

    nz_from_summary = int(grid_summary.get("Nz", 0) or 0)
    z_length_um = grid_summary.get("z_length_um")
    if z_length_um is not None and dz > 0.0:
        nz_from_length = max(1, int(round(float(z_length_um) / dz)))
    else:
        nz_from_length = 0
    nz = max(nz_from_summary, nz_from_length, 1)

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    z = np.arange(nz) * dz

    return Geometry(x=x, y=y, z=z, units="um")



def _intensity_from_A(A, *, coherent: bool = False, coherence_groups=None):
    return asnumpy(
        total_intensity(
            A,
            coherent=coherent,
            coherence_groups=coherence_groups,
        )
    )


def _result_coherence(result) -> tuple[bool, tuple[str, ...] | None]:
    """Return legacy and grouped coherence metadata from a workflow result."""

    summary = getattr(result, "launch_summary", {})
    groups = summary.get("coherence_groups")
    return summary.get("coherence") == "coherent", None if groups is None else tuple(groups)


def _as_zxy_stack(field, geometry: Geometry):
    data = asnumpy(field)
    if data.ndim == 3:
        return data
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D field, got shape {data.shape}")
    nz = 1 if geometry.z is None else len(np.asarray(geometry.z))
    return np.repeat(data[None, :, :], nz, axis=0)



def from_static_result(result) -> RunData:
    geometry = _geometry_from_grid_summary(result.grid_summary)
    theta_stack = _as_zxy_stack(result.theta_final, geometry)
    theta_2d = theta_stack[-1]
    coherent, coherence_groups = _result_coherence(result)

    fields = []
    theta_bias = getattr(result, "theta_bias", None)
    if theta_bias is not None:
        theta_bias_2d = asnumpy(theta_bias)
        delta_theta_stack = theta_stack - theta_bias_2d[None, :, :]
        fields.append(("delta_theta_stack", make_field("delta_theta_stack", "Δθ", delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")))


    fields.extend([
        ("theta_stack", make_field("theta_stack", "θ", theta_stack, ("z", "x", "y"), "theta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ("final_intensity", make_field("final_intensity", "Output Plane Intensity", _intensity_from_A(result.A_final, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="intensity", value_unit="mW/um²")),
        ("theta", make_field("theta", "Output Plane Theta", theta_2d, ("x", "y"), "theta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad")),
    ])

    return RunData(
        workflow="static",
        geometry=geometry,
        fields=FieldCollection(fields),
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
    coherent, coherence_groups = _result_coherence(result)

    fields = []

    theta_initial = getattr(result, "theta_initial", None)
    if theta_initial is not None:
        theta_initial_stack = asnumpy(theta_initial)
        initial_delta_theta_stack = theta_initial_stack - theta_bias[None, :, :]
        fields.extend([
            ("initial_delta_theta_stack", make_field("initial_delta_theta_stack", "Initial Δθ", initial_delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
            ("initial_theta_stack", make_field("initial_theta_stack", "Initial θ", theta_initial_stack, ("z", "x", "y"), "theta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ])

    A_initial = getattr(result, "A_initial", None)
    if A_initial is not None:
        fields.append(("initial_intensity", make_field("initial_intensity", "Initial Intensity", _intensity_from_A(A_initial, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="intensity", value_unit="mW/um²")))
    initial_intensity_stack = getattr(result, "initial_intensity_stack", None)
    if initial_intensity_stack is not None:
        fields.append((
            "initial_intensity_stack",
            make_field(
                "initial_intensity_stack",
                "Initial TD Source Intensity",
                asnumpy(initial_intensity_stack),
                ("z", "x", "y"),
                "intensity",
                {"z": "um", "x": "um", "y": "um"},
                "longitudinal",
                quantity="intensity",
                value_unit="mW/um²",
            ),
        ))

    fields.extend([
        ("delta_theta_stack", make_field("delta_theta_stack", "Final Δθ", delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ("theta_stack", make_field("theta_stack", "Final θ", theta_stack, ("z", "x", "y"), "theta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ("final_intensity", make_field("final_intensity", "Output Plane Intensity", _intensity_from_A(result.A_final, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="intensity", value_unit="mW/um²")),
    ])

    return RunData(
        workflow="timedependent",
        geometry=_geometry_from_grid_summary(result.grid_summary),
        fields=FieldCollection(fields),
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
                    "has_initial_A": A_initial is not None,
                    "has_initial_theta": theta_initial is not None,
                    "has_initial_td_source": initial_intensity_stack is not None,
                },
            ))
        ]),
    )


def from_soliton_result(result) -> RunData:
    grid_summary = result.metrics.get("grid")
    if grid_summary is None:
        A = asnumpy(result.A)
        nx, ny = A.shape[-2], A.shape[-1]
        grid_summary = {
            "Nx": nx,
            "Ny": ny,
            "Nz": 1,
            "dx_um": 1.0,
            "dy_um": 1.0,
            "dz_um": 1.0,
        }
    geometry = _geometry_from_grid_summary(grid_summary)

    theta_2d = asnumpy(result.theta)
    intensity = asnumpy(result.intensity)

    fields = [
        ("final_intensity", make_field("final_intensity", "Output Plane Intensity", intensity, ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="intensity", value_unit="mW/um²")),
        ("theta", make_field("theta", "Output Plane Theta", theta_2d, ("x", "y"), "theta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad")),
    ]

    curves = CurveCollection()
    history = getattr(result, "history", None) or getattr(result, "samples", None) or []
    if history:
        outer = np.asarray([r["outer"] for r in history])
        curves.add("residual_rms", CurveData(
            "residual_rms",
            "Residual RMS",
            outer,
            np.asarray([r.get("residual_rms", np.nan) for r in history]),
            "outer",
            "residual_rms",
        ))
        curves.add("beta_history", CurveData(
            "beta_history",
            "Beta history",
            outer,
            np.asarray([r.get("beta", np.nan) for r in history]),
            "outer",
            "beta",
        ))

    summary = dict(result.metrics)
    summary["mode"] = getattr(result, "mode", "00")

    return RunData(
        workflow="soliton",
        geometry=geometry,
        fields=FieldCollection(fields),
        curves=curves,
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", summary))
        ]),
    )




def from_soliton_existence_result(result) -> RunData:
    rows = list(getattr(result, "samples", []) or [])
    curves = CurveCollection()

    if rows:
        P = np.asarray([
            r.get("requested_power_mW", r.get("value", np.nan))
            for r in rows
        ], dtype=float)
        for key, label in [
            ("beta", "Beta"),
            ("theta_max", "Theta max"),
            
            ("Imax", "Imax"),
            ("residual_rms", "Residual RMS"),
            ("residual_max", "Residual max"),
            ("field_rel", "Field relative change"),
            ("overlap_abs", "Mode overlap"),
            ("dtheta_rms", "Theta update RMS"),
            ("elapsed_s", "Elapsed time"),
            ("converged", "Converged"),
        ]:
            y = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
            curves.add(key, CurveData(
                key,
                label,
                P,
                y,
                "P",
                key,
                {"P": "mW"},
            ))

    summary = dict(getattr(result, "metrics", {}) or {})
    summary.setdefault("n_rows", len(rows))
    summary.setdefault("kind", getattr(result, "kind", type(result).__name__))
    summary.setdefault("mode", getattr(result, "mode", summary.get("mode", "00")))

    return RunData(
        workflow="soliton_existence",
        curves=curves,
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData("summary", "Summary", summary)),
            ("table", DiagnosticData("table", "Samples", {"rows": rows})),
        ]),
    )


# Conversion for generic parameter sweep result, reusing soliton existence result logic.
def from_parameter_sweep_result(result) -> RunData:
    """Convert a generic parameter sweep result into GUI products."""
    run_data = from_soliton_existence_result(result)

    if getattr(result, "results", None):
        field_data = from_soliton_result(result.results[-1])
        fields = field_data.fields
        geometry = field_data.geometry
    else:
        fields = run_data.fields
        geometry = run_data.geometry

    return RunData(
        workflow="parameter_sweep",
        geometry=geometry,
        fields=fields,
        curves=run_data.curves,
        diagnostics=run_data.diagnostics,
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
    if name == "ParameterSweepResult":
        return from_parameter_sweep_result(result)
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
    "from_parameter_sweep_result",
    "to_run_data",
]
