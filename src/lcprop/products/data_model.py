from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.splitstep import total_intensity


@dataclass(frozen=True)
class FieldData:
    name: str
    data: Any
    axes: tuple[str, ...]
    kind: str
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CurveData:
    name: str
    x: Any
    y: Any
    x_label: str
    y_label: str
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticData:
    name: str
    values: dict[str, Any]


@dataclass(frozen=True)
class RunData:
    workflow: str
    fields: dict[str, FieldData] = field(default_factory=dict)
    curves: dict[str, CurveData] = field(default_factory=dict)
    diagnostics: dict[str, DiagnosticData] = field(default_factory=dict)


def _intensity_from_A(A, *, coherent: bool = False):
    return asnumpy(total_intensity(A, coherent=coherent))


def from_static_result(result) -> RunData:
    return RunData(
        workflow="static",
        fields={
            "intensity": FieldData("Intensity", _intensity_from_A(result.A_final), ("x", "y"), "intensity", {"x": "um", "y": "um"}),
            "theta": FieldData("Theta", asnumpy(result.theta_final), ("x", "y"), "theta", {"x": "um", "y": "um", "theta": "rad"}),
        },
        diagnostics={
            "summary": DiagnosticData(
                "Summary",
                {
                    "power_initial": result.power_initial,
                    "power_final": result.power_final,
                    "method": result.method,
                    "n_steps": result.n_steps,
                    "grid": result.grid_summary,
                },
            )
        },
    )


def from_timedependent_result(result) -> RunData:
    return RunData(
        workflow="timedependent",
        fields={
            "final_intensity": FieldData("Final intensity", _intensity_from_A(result.A_final), ("x", "y"), "intensity", {"x": "um", "y": "um"}),
            "theta_stack": FieldData("Theta stack", asnumpy(result.theta_final), ("z", "x", "y"), "theta", {"z": "um", "x": "um", "y": "um", "theta": "rad"}),
        },
        diagnostics={
            "summary": DiagnosticData(
                "Summary",
                {
                    "power_initial": result.power_initial,
                    "power_final": result.power_final,
                    "Nt": result.Nt,
                    "method": result.method,
                    "grid": result.grid_summary,
                },
            )
        },
    )


def from_soliton_result(result) -> RunData:
    curves = {}
    if result.history:
        outer = np.asarray([r["outer"] for r in result.history])
        curves["residual_rms"] = CurveData(
            "Residual RMS",
            outer,
            np.asarray([r.get("residual_rms", np.nan) for r in result.history]),
            "outer",
            "residual_rms",
        )
        curves["beta_history"] = CurveData(
            "Beta history",
            outer,
            np.asarray([r.get("beta", np.nan) for r in result.history]),
            "outer",
            "beta",
        )

    return RunData(
        workflow="soliton",
        fields={
            "intensity": FieldData("Intensity", asnumpy(result.intensity), ("x", "y"), "intensity", {"x": "um", "y": "um"}),
            "theta": FieldData("Theta", asnumpy(result.theta), ("x", "y"), "theta", {"x": "um", "y": "um", "theta": "rad"}),
        },
        curves=curves,
        diagnostics={"summary": DiagnosticData("Summary", dict(result.metrics))},
    )


def from_soliton_existence_result(result) -> RunData:
    rows = result.samples
    curves = {}

    if rows:
        P = np.asarray([r.get("requested_power_mW", np.nan) for r in rows])
        for key, label in [
            ("beta", "Beta"),
            ("theta_max", "Theta max"),
            ("Imax", "Imax"),
            ("residual_rms", "Residual RMS"),
        ]:
            curves[key] = CurveData(
                label,
                P,
                np.asarray([r.get(key, np.nan) for r in rows]),
                "P",
                key,
                {"P": "mW"},
            )

    return RunData(
        workflow="soliton_existence",
        curves=curves,
        diagnostics={
            "summary": DiagnosticData("Summary", dict(result.metrics)),
            "table": DiagnosticData("Samples", {"rows": rows}),
        },
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
