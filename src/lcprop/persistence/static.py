"""Versioned, portable checkpoints for slice-local static propagation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.core.results import StaticIterationRecord, StaticSliceSummary


STATIC_CHECKPOINT_SCHEMA_VERSION = 1
StaticCheckpointStatus = Literal["stopped", "completed"]


@dataclass(frozen=True)
class StaticCheckpoint:
    """State required to continue at the next uncomputed z slice."""

    request: StaticRunRequest
    next_slice_index: int
    completed_slices: int
    z_reached_um: float
    A_next: np.ndarray
    theta_seed: np.ndarray
    theta_stack: np.ndarray
    intensity_stack: np.ndarray
    theta_intensity_stack: np.ndarray | None
    slice_summaries: tuple[StaticSliceSummary, ...]
    iteration_records: tuple[StaticIterationRecord, ...]
    relax_steps: int
    grid_summary: dict[str, Any]
    launch_summary: dict[str, Any]
    normalized_power_initial: float
    physical_power_initial_mW: float
    A_dtype: str
    theta_dtype: str
    status: StaticCheckpointStatus
    request_fingerprint: str
    schema_version: int = STATIC_CHECKPOINT_SCHEMA_VERSION
    workflow: Literal["static"] = "static"


def _request_to_dict(request: StaticRunRequest) -> dict[str, Any]:
    return {
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "bias": asdict(request.bias),
        "beams": {
            "coherence": request.beams.coherence,
            "channels": [asdict(channel) for channel in request.beams.channels],
        },
        "solver": {
            **asdict(request.solver),
            "workflow": asdict(request.solver.workflow),
        },
        "output": {
            "run_dir": None,
            "save_slices": bool(request.output.save_slices),
            "save_full": bool(request.output.save_full),
        },
        "runtime": asdict(request.runtime),
        "initial_conditions": {
            "A": "replaced by checkpoint.npz:A_next",
            "theta": "replaced by checkpoint.npz:theta_seed",
        },
    }


def _request_from_dict(values: dict[str, Any]) -> StaticRunRequest:
    beam_values = values["beams"]
    solver_values = dict(values["solver"])
    workflow = StaticWorkflowOptions(**solver_values.pop("workflow"))
    output_values = dict(values["output"])
    output_values["run_dir"] = None
    return StaticRunRequest(
        grid=GridSpec(**values["grid"]),
        material=LCMaterial(**values["material"]),
        bias=BiasSpec(**values["bias"]),
        beams=BeamStack(
            channels=tuple(
                BeamChannel(**channel) for channel in beam_values["channels"]
            ),
            coherence=beam_values["coherence"],
        ),
        solver=StaticSolverOptions(workflow=workflow, **solver_values),
        output=OutputOptions(**output_values),
        runtime=RuntimeOptions(**values["runtime"]),
    )


def static_request_fingerprint(request: StaticRunRequest) -> str:
    document = json.dumps(
        _request_to_dict(request), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def validate_static_checkpoint(checkpoint: StaticCheckpoint) -> None:
    if checkpoint.schema_version != STATIC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported static checkpoint schema version: "
            f"{checkpoint.schema_version}"
        )
    if checkpoint.workflow != "static":
        raise ValueError("invalid static checkpoint workflow")
    if checkpoint.status not in ("stopped", "completed"):
        raise ValueError(f"invalid static checkpoint status: {checkpoint.status}")
    if checkpoint.next_slice_index != checkpoint.completed_slices:
        raise ValueError("static checkpoint next slice must equal completed slices")
    n = checkpoint.completed_slices
    for name, array in (
        ("theta_stack", checkpoint.theta_stack),
        ("intensity_stack", checkpoint.intensity_stack),
    ):
        if array.ndim != 3 or array.shape[0] != n:
            raise ValueError(f"{name} must contain exactly the completed prefix")
    if checkpoint.theta_intensity_stack is not None:
        if checkpoint.theta_intensity_stack.shape != checkpoint.theta_stack.shape:
            raise ValueError("theta intensity prefix shape does not match theta prefix")
    if checkpoint.A_next.ndim != 3 or checkpoint.theta_seed.ndim != 2:
        raise ValueError("invalid static continuation field shapes")
    if checkpoint.A_next.shape[1:] != checkpoint.theta_seed.shape:
        raise ValueError("static optical and theta spatial shapes do not match")
    if str(checkpoint.A_next.dtype) != checkpoint.A_dtype:
        raise ValueError("static checkpoint A dtype metadata does not match array")
    if str(checkpoint.theta_seed.dtype) != checkpoint.theta_dtype:
        raise ValueError("static checkpoint theta dtype metadata does not match array")
    if checkpoint.request_fingerprint != static_request_fingerprint(checkpoint.request):
        raise ValueError("static checkpoint request fingerprint does not match request")
    if not np.isfinite(checkpoint.normalized_power_initial):
        raise ValueError("static checkpoint normalized power must be finite")
    if not np.isfinite(checkpoint.physical_power_initial_mW):
        raise ValueError("static checkpoint physical power must be finite")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_static_checkpoint(checkpoint: StaticCheckpoint, run_dir) -> Path:
    """Atomically save a static completed-prefix checkpoint."""

    validate_static_checkpoint(checkpoint)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request_document = {
        "workflow": "static",
        "schema_version": checkpoint.schema_version,
        "request": _request_to_dict(checkpoint.request),
    }
    provenance = {
        "workflow": "static",
        "schema_version": checkpoint.schema_version,
        "status": checkpoint.status,
        "completed_units": checkpoint.completed_slices,
        "current_coordinate": checkpoint.z_reached_um,
        "coordinate_name": "z",
        "coordinate_unit": "um",
        "next_slice_index": checkpoint.next_slice_index,
        "completed_slices": checkpoint.completed_slices,
        "z_reached_um": checkpoint.z_reached_um,
        "relax_steps": checkpoint.relax_steps,
        "grid_summary": checkpoint.grid_summary,
        "launch_summary": checkpoint.launch_summary,
        "normalized_power_initial": checkpoint.normalized_power_initial,
        "physical_power_initial_mW": checkpoint.physical_power_initial_mW,
        "A_dtype": checkpoint.A_dtype,
        "theta_dtype": checkpoint.theta_dtype,
        "request_fingerprint": checkpoint.request_fingerprint,
        "slice_summaries": [asdict(item) for item in checkpoint.slice_summaries],
        "iteration_records": [asdict(item) for item in checkpoint.iteration_records],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "request": "request.json",
            "arrays": "checkpoint.npz",
            "provenance": "provenance.json",
        },
    }
    arrays_path = directory / "checkpoint.npz"
    arrays_temporary = arrays_path.with_suffix(".npz.tmp")
    with arrays_temporary.open("wb") as stream:
        arrays = {
            "A_next": np.asarray(checkpoint.A_next),
            "theta_seed": np.asarray(checkpoint.theta_seed),
            "theta_stack": np.asarray(checkpoint.theta_stack),
            "intensity_stack": np.asarray(checkpoint.intensity_stack),
        }
        if checkpoint.theta_intensity_stack is not None:
            arrays["theta_intensity_stack"] = np.asarray(
                checkpoint.theta_intensity_stack
            )
        np.savez_compressed(stream, **arrays)
    arrays_temporary.replace(arrays_path)
    _atomic_json(directory / "request.json", request_document)
    _atomic_json(directory / "provenance.json", provenance)
    return directory


def load_static_checkpoint(run_dir) -> StaticCheckpoint:
    directory = Path(run_dir)
    request_document = json.loads(
        (directory / "request.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    if request_document.get("workflow") != "static" or provenance.get("workflow") != "static":
        raise ValueError("checkpoint directory does not contain a static workflow")
    with np.load(directory / "checkpoint.npz", allow_pickle=False) as arrays:
        theta_intensity = (
            np.asarray(arrays["theta_intensity_stack"]).copy()
            if "theta_intensity_stack" in arrays.files
            else None
        )
        checkpoint = StaticCheckpoint(
            request=_request_from_dict(request_document["request"]),
            next_slice_index=int(provenance["next_slice_index"]),
            completed_slices=int(provenance["completed_slices"]),
            z_reached_um=float(provenance["z_reached_um"]),
            A_next=np.asarray(arrays["A_next"]).copy(),
            theta_seed=np.asarray(arrays["theta_seed"]).copy(),
            theta_stack=np.asarray(arrays["theta_stack"]).copy(),
            intensity_stack=np.asarray(arrays["intensity_stack"]).copy(),
            theta_intensity_stack=theta_intensity,
            slice_summaries=tuple(
                StaticSliceSummary(**item)
                for item in provenance["slice_summaries"]
            ),
            iteration_records=tuple(
                StaticIterationRecord(**item)
                for item in provenance["iteration_records"]
            ),
            relax_steps=int(provenance["relax_steps"]),
            grid_summary=dict(provenance["grid_summary"]),
            launch_summary=dict(provenance["launch_summary"]),
            normalized_power_initial=float(
                provenance["normalized_power_initial"]
            ),
            physical_power_initial_mW=float(
                provenance["physical_power_initial_mW"]
            ),
            A_dtype=str(provenance["A_dtype"]),
            theta_dtype=str(provenance["theta_dtype"]),
            status=str(provenance["status"]),
            request_fingerprint=str(provenance["request_fingerprint"]),
            schema_version=int(provenance["schema_version"]),
        )
    validate_static_checkpoint(checkpoint)
    return checkpoint


__all__ = [
    "STATIC_CHECKPOINT_SCHEMA_VERSION",
    "StaticCheckpoint",
    "load_static_checkpoint",
    "save_static_checkpoint",
    "static_request_fingerprint",
    "validate_static_checkpoint",
]
