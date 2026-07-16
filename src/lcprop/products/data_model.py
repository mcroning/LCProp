from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
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


def _result_power_diagnostics(result) -> dict[str, float | None]:
    """Separate normalized field integrals from physical power diagnostics."""
    return {
        "normalized_field_integral_initial": result.power_initial,
        "normalized_field_integral_final": result.power_final,
        "physical_power_initial_mW": getattr(result, "physical_power_initial_mW", None),
        "physical_power_final_mW": getattr(result, "physical_power_final_mW", None),
    }


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
    if theta_stack.shape[0] == 0:
        coherent, coherence_groups = _result_coherence(result)
        A_initial = getattr(result, "A_initial", None)
        if A_initial is None:
            raise ValueError("static result does not contain the input optical field")
        geometry = replace(geometry, z=np.asarray(geometry.z)[:0])
        return RunData(
            workflow="static",
            geometry=geometry,
            fields=FieldCollection([
                ("input_intensity", make_field(
                    "input_intensity", "Input Plane Intensity",
                    _intensity_from_A(
                        A_initial,
                        coherent=coherent,
                        coherence_groups=coherence_groups,
                    ),
                    ("x", "y"), "intensity", {"x": "um", "y": "um"},
                    quantity="normalized_intensity", value_unit="1/µm²",
                )),
            ]),
            diagnostics=DiagnosticCollection([
                ("summary", DiagnosticData("summary", "Summary", {
                    "status": getattr(result, "status", "stopped"),
                    "completed_slices": 0,
                    "total_slices": getattr(result, "total_slices", 0),
                    "z_reached_um": getattr(result, "z_reached_um", 0.0),
                    "grid": result.grid_summary,
                })),
            ]),
        )
    geometry = replace(geometry, z=np.asarray(geometry.z)[: theta_stack.shape[0]])
    coherent, coherence_groups = _result_coherence(result)
    theta_bias_2d = asnumpy(result.theta_bias)
    delta_theta_stack = theta_stack - theta_bias_2d[None, :, :]

    A_initial = getattr(result, "A_initial", None)
    if A_initial is None:
        raise ValueError("static result does not contain the input optical field")
    intensity_stack = getattr(result, "intensity_stack", None)
    if intensity_stack is None:
        raise ValueError("static result does not contain a z-dependent intensity volume")

    stopped = getattr(result, "status", "completed") == "stopped"
    final_intensity_label = (
        "Intensity at current z" if stopped else "Output Plane Intensity"
    )
    final_delta_theta_label = (
        "Δθ at current z" if stopped else "Output Plane Δθ"
    )
    fields = [
        ("input_intensity", make_field("input_intensity", "Input Plane Intensity", _intensity_from_A(A_initial, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="normalized_intensity", value_unit="1/µm²")),
        ("final_intensity", make_field("final_intensity", final_intensity_label, _intensity_from_A(result.A_final, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="normalized_intensity", value_unit="1/µm²")),
        ("input_delta_theta", make_field("input_delta_theta", "Input Plane Δθ", delta_theta_stack[0], ("x", "y"), "theta_delta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad")),
        ("output_delta_theta", make_field("output_delta_theta", final_delta_theta_label, delta_theta_stack[-1], ("x", "y"), "theta_delta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad")),
        ("intensity_stack", make_field("intensity_stack", "Intensity", asnumpy(intensity_stack), ("z", "x", "y"), "intensity", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="normalized_intensity", value_unit="1/µm²")),
        ("delta_theta_stack", make_field("delta_theta_stack", "Δθ", delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
    ]

    slice_summaries = tuple(getattr(result, "slice_summaries", ()) or ())
    iteration_records = tuple(getattr(result, "iteration_records", ()) or ())
    curves = CurveCollection()
    convergence_rows = []
    for item in slice_summaries:
        convergence_rows.append({
            "z_index": item.z_index,
            "z_um": item.z_um,
            "converged": item.converged,
            "iterations": item.relaxation_iterations,
            "optical_passes": item.optical_passes,
            "final_residual_rms": item.final_residual_rms,
            "final_residual_max": item.final_residual_max,
            "delta_theta_rms": item.final_delta_theta_rms,
            "delta_theta_max": item.final_delta_theta_max,
            "theta_max": item.theta_max,
            "termination_reason": item.termination_reason,
        })

    if slice_summaries:
        z_um = np.asarray([item.z_um for item in slice_summaries], dtype=float)
        curve_specs = (
            ("static_final_residual_rms", "Final Residual RMS", "residual RMS", [item.final_residual_rms for item in slice_summaries], "log"),
            ("static_final_residual_max", "Final Residual Max", "residual max", [item.final_residual_max for item in slice_summaries], "log"),
            ("static_relaxation_iterations", "Relaxation Iterations", "iterations", [item.relaxation_iterations for item in slice_summaries], "linear"),
            ("static_theta_max", "Theta Max", "theta max", [item.theta_max for item in slice_summaries], "linear"),
        )
        for key, display_name, y_label, values, y_scale in curve_specs:
            curves.add(key, CurveData(
                key,
                display_name,
                z_um,
                np.asarray(values, dtype=float),
                "z",
                y_label,
                {"z": "um", "theta max": "rad"},
                y_scale=y_scale,
            ))

    summary_values = {
        **_result_power_diagnostics(result),
        "method": result.method,
        "n_steps": result.n_steps,
        "status": getattr(result, "status", "completed"),
        "completed_slices": getattr(result, "completed_slices", theta_stack.shape[0]),
        "total_slices": getattr(result, "total_slices", result.grid_summary.get("Nz")),
        "z_reached_um": getattr(result, "z_reached_um", None),
        "intensity_volume_sampling": "accepted slice midpoint (average of entrance and exit plane intensities)",
        "grid": result.grid_summary,
        "all_slices_converged": getattr(result, "all_slices_converged", None),
        "max_final_residual_rms": getattr(result, "max_final_residual_rms", None),
        "median_final_residual_rms": getattr(result, "median_final_residual_rms", None),
        "rms_over_z_final_residual": getattr(result, "rms_over_z_final_residual", None),
        "max_final_residual_max": getattr(result, "max_final_residual_max", None),
        "worst_slice_index": getattr(result, "worst_slice_index", None),
    }
    diagnostics = [
        ("summary", DiagnosticData("summary", "Summary", summary_values)),
    ]
    if slice_summaries:
        diagnostics.append((
            "static_convergence",
            DiagnosticData(
                "static_convergence",
                "Static Convergence by Slice",
                {"rows": convergence_rows},
            ),
        ))
    if iteration_records:
        diagnostics.append((
            "static_iteration_history",
            DiagnosticData(
                "static_iteration_history",
                "Static Iteration History",
                {"rows": [asdict(item) for item in iteration_records]},
            ),
        ))

    return RunData(
        workflow="static",
        geometry=geometry,
        fields=FieldCollection(fields),
        curves=curves,
        diagnostics=DiagnosticCollection(diagnostics),
    )


def from_static_live_state(state: dict[str, Any]) -> RunData:
    """Expose only the immutable input and latest completed static slice."""

    geometry = _geometry_from_grid_summary(state["grid_summary"])
    completed = int(state["completed_slices"])
    geometry = replace(geometry, z=np.asarray(geometry.z)[:completed])
    groups = tuple(state["launch_summary"].get("coherence_groups", ())) or None
    bias = np.asarray(state["theta_bias"])
    fields = FieldCollection([
        ("input_intensity", make_field(
            "input_intensity", "Input Plane Intensity",
            _intensity_from_A(state["A_initial"], coherence_groups=groups),
            ("x", "y"), "intensity", {"x": "um", "y": "um"},
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("final_intensity", make_field(
            "final_intensity", "Intensity at current z",
            _intensity_from_A(state["A_current"], coherence_groups=groups),
            ("x", "y"), "intensity", {"x": "um", "y": "um"},
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("input_delta_theta", make_field(
            "input_delta_theta", "Input Plane Δθ",
            np.asarray(state["theta_input"]) - bias,
            ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
            quantity="theta", value_unit="rad",
        )),
        ("output_delta_theta", make_field(
            "output_delta_theta", "Δθ at current z",
            np.asarray(state["theta_current"]) - bias,
            ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
            quantity="theta", value_unit="rad",
        )),
    ])
    return RunData(workflow="static", geometry=geometry, fields=fields)


def _resolved_static_reference(static_reference, *, fallback=None):
    reference = fallback if static_reference is None else static_reference
    if reference is None:
        raise ValueError(
            "TD-from-static display requires its immutable static reference"
        )
    required = (
        "intensity_stack",
        "theta_stack",
        "theta_bias",
        "output_plane_intensity",
    )
    missing = [
        name
        for name in required
        if name not in reference or reference[name] is None
    ]
    if missing:
        raise ValueError(
            "TD-from-static display reference is missing: " + ", ".join(missing)
        )
    return reference


def from_timedependent_result(
    result,
    *,
    td_from_static: bool = False,
    static_reference=None,
) -> RunData:
    geometry = _geometry_from_grid_summary(result.grid_summary)
    theta_stack = asnumpy(result.theta_final)
    theta_bias = asnumpy(result.theta_bias)
    delta_theta_stack = theta_stack - theta_bias[None, :, :]
    coherent, coherence_groups = _result_coherence(result)
    theta_initial = getattr(result, "theta_initial", None)
    A_initial = getattr(result, "A_initial", None)
    initial_intensity_stack = getattr(result, "initial_intensity_stack", None)
    final_intensity_stack = getattr(result, "final_intensity_stack", None)
    if theta_initial is None or A_initial is None:
        raise ValueError("time-dependent result does not contain its initial state")
    if initial_intensity_stack is None or final_intensity_stack is None:
        raise ValueError("time-dependent result does not contain initial and final intensity volumes")

    theta_initial_stack = asnumpy(theta_initial)
    initial_delta_theta_stack = theta_initial_stack - theta_bias[None, :, :]
    # The 2-D TD delta-theta products use the longitudinal pane's default
    # selected plane. The full volumes remain available for every z slice.
    selected_z_index = theta_stack.shape[0] // 2
    stopped = getattr(result, "status", "completed") == "cancelled"
    final_intensity_label = "Intensity at Stop" if stopped else "Final Intensity"
    final_delta_theta_label = "Δθ at Stop" if stopped else "Final Δθ"

    if td_from_static:
        reference = _resolved_static_reference(
            static_reference,
            fallback={
                "intensity_stack": initial_intensity_stack,
                "theta_stack": theta_initial,
                "theta_bias": theta_bias,
                "output_plane_intensity": getattr(
                    result, "initial_output_plane_intensity", None
                ),
            },
        )
        static_intensity_stack = np.asarray(reference["intensity_stack"])
        static_theta_stack = np.asarray(reference["theta_stack"])
        static_bias = np.asarray(reference["theta_bias"])
        static_delta_theta_stack = static_theta_stack - static_bias[None, :, :]
        if static_intensity_stack.shape != theta_stack.shape:
            raise ValueError(
                "initial static intensity reference shape does not match TD grid"
            )
        if static_theta_stack.shape != theta_stack.shape:
            raise ValueError(
                "initial static theta reference shape does not match TD grid"
            )
        final_intensity_label = (
            "Output Plane Intensity at Stop"
            if stopped
            else "Final Output Plane Intensity"
        )
        final_delta_theta_label = (
            "Output Plane Δθ at Stop"
            if stopped
            else "Final Output Plane Δθ"
        )
        fields = [
            ("initial_static_output_intensity", make_field(
                "initial_static_output_intensity",
                "Initial Static Output Plane Intensity",
                np.asarray(reference["output_plane_intensity"]),
                ("x", "y"), "intensity", {"x": "um", "y": "um"},
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("final_intensity", make_field(
                "final_intensity", final_intensity_label,
                _intensity_from_A(
                    result.A_final,
                    coherent=coherent,
                    coherence_groups=coherence_groups,
                ),
                ("x", "y"), "intensity", {"x": "um", "y": "um"},
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("initial_static_output_delta_theta", make_field(
                "initial_static_output_delta_theta",
                "Initial Static Output Plane Δθ",
                static_delta_theta_stack[-1],
                ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
                quantity="theta", value_unit="rad",
            )),
            ("final_delta_theta", make_field(
                "final_delta_theta", final_delta_theta_label,
                delta_theta_stack[-1],
                ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
                quantity="theta", value_unit="rad",
            )),
            ("initial_static_intensity_stack", make_field(
                "initial_static_intensity_stack", "Initial Static Intensity",
                static_intensity_stack, ("z", "x", "y"), "intensity",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("initial_static_delta_theta_stack", make_field(
                "initial_static_delta_theta_stack", "Initial Static Delta Theta",
                static_delta_theta_stack, ("z", "x", "y"), "theta_delta",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="theta", value_unit="rad",
            )),
            ("final_intensity_stack", make_field(
                "final_intensity_stack",
                "Intensity at Stop" if stopped else "Final Intensity",
                asnumpy(final_intensity_stack),
                ("z", "x", "y"), "intensity",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("final_delta_theta_stack", make_field(
                "final_delta_theta_stack",
                "Delta Theta at Stop" if stopped else "Final Delta Theta",
                delta_theta_stack,
                ("z", "x", "y"), "theta_delta",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="theta", value_unit="rad",
            )),
        ]
    else:
        fields = [
        ("initial_intensity", make_field("initial_intensity", "Initial Intensity", _intensity_from_A(A_initial, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="normalized_intensity", value_unit="1/µm²")),
        ("final_intensity", make_field("final_intensity", final_intensity_label, _intensity_from_A(result.A_final, coherent=coherent, coherence_groups=coherence_groups), ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="normalized_intensity", value_unit="1/µm²")),
        ("initial_delta_theta", make_field("initial_delta_theta", "Initial Δθ", initial_delta_theta_stack[selected_z_index], ("x", "y"), "theta_delta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad", source_volume_key="initial_delta_theta_stack")),
        ("final_delta_theta", make_field("final_delta_theta", final_delta_theta_label, delta_theta_stack[selected_z_index], ("x", "y"), "theta_delta", {"x": "um", "y": "um"}, quantity="theta", value_unit="rad", source_volume_key="final_delta_theta_stack")),
        ("initial_intensity_stack", make_field("initial_intensity_stack", "Initial Intensity", asnumpy(initial_intensity_stack), ("z", "x", "y"), "intensity", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="normalized_intensity", value_unit="1/µm²")),
        ("final_intensity_stack", make_field("final_intensity_stack", "Intensity at Stop" if stopped else "Final Intensity", asnumpy(final_intensity_stack), ("z", "x", "y"), "intensity", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="normalized_intensity", value_unit="1/µm²")),
        ("initial_delta_theta_stack", make_field("initial_delta_theta_stack", "Initial Delta Theta", initial_delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ("final_delta_theta_stack", make_field("final_delta_theta_stack", "Delta Theta at Stop" if stopped else "Final Delta Theta", delta_theta_stack, ("z", "x", "y"), "theta_delta", {"z": "um", "x": "um", "y": "um"}, "longitudinal", quantity="theta", value_unit="rad")),
        ]

    return RunData(
        workflow="timedependent",
        geometry=geometry,
        fields=FieldCollection(fields),
        diagnostics=DiagnosticCollection([
            ("summary", DiagnosticData(
                "summary",
                "Summary",
                {
                    **_result_power_diagnostics(result),
                    "Nt": result.Nt,
                    "status": getattr(result, "status", "completed"),
                    "completed_steps": getattr(result, "completed_steps", result.Nt),
                    "requested_steps": getattr(result, "requested_steps", result.Nt),
                    "current_time": getattr(result, "current_time", None),
                    "current_time_semantics": "cumulative nondimensional TD simulation time",
                    "segment_start_time": getattr(result, "segment_start_time", 0.0),
                    "segment_elapsed_time": getattr(result, "segment_elapsed_time", None),
                    "cumulative_time": getattr(result, "cumulative_time", getattr(result, "current_time", None)),
                    "prior_completed_steps": getattr(result, "prior_completed_steps", 0),
                    "segment_completed_steps": getattr(result, "segment_completed_steps", result.Nt),
                    "cumulative_completed_steps": getattr(result, "cumulative_completed_steps", getattr(result, "completed_steps", result.Nt)),
                    "method": result.method,
                    "grid": result.grid_summary,
                    "has_initial_A": A_initial is not None,
                    "has_initial_theta": theta_initial is not None,
                    "transverse_delta_theta_z_index": selected_z_index,
                    "intensity_volume_sampling": "slice midpoint (average of entrance and exit plane intensities)",
                    "td_source_intensity_retained_on_result": getattr(result, "initial_source_intensity_stack", None) is not None,
                    "initial_condition_source": (
                        "static result" if td_from_static else "default or explicit TD state"
                    ),
                },
            ))
        ]),
    )


def from_timedependent_live_state(
    state: dict[str, Any],
    *,
    td_from_static: bool = False,
    static_reference=None,
) -> RunData:
    """Expose the latest completed TD-step state without result construction."""

    geometry = _geometry_from_grid_summary(state["grid_summary"])
    groups = tuple(state["launch_summary"].get("coherence_groups", ())) or None
    bias = np.asarray(state["theta_bias"])
    theta_initial = np.asarray(state["theta_initial"])
    theta_current = np.asarray(state["theta_current"])
    iz = theta_current.shape[0] // 2
    if td_from_static:
        reference = _resolved_static_reference(
            static_reference,
            fallback={
                "intensity_stack": state["initial_intensity_stack"],
                "theta_stack": state["theta_initial"],
                "theta_bias": state["theta_bias"],
                "output_plane_intensity": state[
                    "initial_output_plane_intensity"
                ],
            },
        )
        static_theta = np.asarray(reference["theta_stack"])
        static_bias = np.asarray(reference["theta_bias"])
        fields = FieldCollection([
            ("initial_static_output_intensity", make_field(
                "initial_static_output_intensity",
                "Initial Static Output Plane Intensity",
                np.asarray(reference["output_plane_intensity"]),
                ("x", "y"), "intensity", {"x": "um", "y": "um"},
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("final_intensity", make_field(
                "final_intensity", "Current Output Plane Intensity",
                np.asarray(state["output_plane_intensity"]),
                ("x", "y"), "intensity", {"x": "um", "y": "um"},
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("initial_static_output_delta_theta", make_field(
                "initial_static_output_delta_theta",
                "Initial Static Output Plane Δθ",
                static_theta[-1] - static_bias,
                ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
                quantity="theta", value_unit="rad",
            )),
            ("final_delta_theta", make_field(
                "final_delta_theta", "Current Output Plane Δθ",
                theta_current[-1] - bias,
                ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
                quantity="theta", value_unit="rad",
            )),
            ("initial_static_intensity_stack", make_field(
                "initial_static_intensity_stack", "Initial Static Intensity",
                np.asarray(reference["intensity_stack"]),
                ("z", "x", "y"), "intensity",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("initial_static_delta_theta_stack", make_field(
                "initial_static_delta_theta_stack", "Initial Static Delta Theta",
                static_theta - static_bias[None, :, :],
                ("z", "x", "y"), "theta_delta",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="theta", value_unit="rad",
            )),
            ("final_intensity_stack", make_field(
                "final_intensity_stack", "Intensity at current t",
                np.asarray(state["current_intensity_stack"]),
                ("z", "x", "y"), "intensity",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="normalized_intensity", value_unit="1/µm²",
            )),
            ("final_delta_theta_stack", make_field(
                "final_delta_theta_stack", "Delta Theta at current t",
                theta_current - bias[None, :, :],
                ("z", "x", "y"), "theta_delta",
                {"z": "um", "x": "um", "y": "um"}, "longitudinal",
                quantity="theta", value_unit="rad",
            )),
        ])
        return RunData(
            workflow="timedependent",
            geometry=geometry,
            fields=fields,
        )

    fields = FieldCollection([
        ("initial_intensity", make_field(
            "initial_intensity", "Initial Intensity",
            _intensity_from_A(state["A_initial"], coherence_groups=groups),
            ("x", "y"), "intensity", {"x": "um", "y": "um"},
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("final_intensity", make_field(
            "final_intensity", "Intensity at current t",
            _intensity_from_A(state["A_current"], coherence_groups=groups),
            ("x", "y"), "intensity", {"x": "um", "y": "um"},
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("initial_delta_theta", make_field(
            "initial_delta_theta", "Initial Δθ",
            theta_initial[iz] - bias,
            ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
            quantity="theta", value_unit="rad",
        )),
        ("final_delta_theta", make_field(
            "final_delta_theta", "Δθ at current t",
            theta_current[iz] - bias,
            ("x", "y"), "theta_delta", {"x": "um", "y": "um"},
            quantity="theta", value_unit="rad",
        )),
        ("initial_intensity_stack", make_field(
            "initial_intensity_stack", "Initial Intensity",
            np.asarray(state["initial_intensity_stack"]),
            ("z", "x", "y"), "intensity",
            {"z": "um", "x": "um", "y": "um"}, "longitudinal",
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("initial_delta_theta_stack", make_field(
            "initial_delta_theta_stack", "Initial Delta Theta",
            theta_initial - bias[None, :, :],
            ("z", "x", "y"), "theta_delta",
            {"z": "um", "x": "um", "y": "um"}, "longitudinal",
            quantity="theta", value_unit="rad",
        )),
        ("final_intensity_stack", make_field(
            "final_intensity_stack", "Intensity at current t",
            np.asarray(state["current_intensity_stack"]),
            ("z", "x", "y"), "intensity",
            {"z": "um", "x": "um", "y": "um"}, "longitudinal",
            quantity="normalized_intensity", value_unit="1/µm²",
        )),
        ("final_delta_theta_stack", make_field(
            "final_delta_theta_stack", "Delta Theta at current t",
            theta_current - bias[None, :, :],
            ("z", "x", "y"), "theta_delta",
            {"z": "um", "x": "um", "y": "um"}, "longitudinal",
            quantity="theta", value_unit="rad",
        )),
    ])
    return RunData(workflow="timedependent", geometry=geometry, fields=fields)


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
        ("final_intensity", make_field("final_intensity", "Normalized Intensity", intensity, ("x", "y"), "intensity", {"x": "um", "y": "um"}, quantity="normalized_intensity", value_unit="1/um²")),
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

def to_run_data(result, **kwargs) -> RunData:
    name = type(result).__name__
    if name == "StaticRunResult":
        return from_static_result(result)
    if name == "TimeDependentRunResult":
        return from_timedependent_result(result, **kwargs)
    if kwargs:
        raise TypeError(
            f"display options are not supported for result type: {name}"
        )
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
    "from_static_live_state",
    "from_timedependent_result",
    "from_timedependent_live_state",
    "from_soliton_result",
    "from_soliton_existence_result",
    "from_parameter_sweep_result",
    "to_run_data",
]
