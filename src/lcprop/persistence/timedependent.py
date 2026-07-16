"""Versioned, portable checkpoints for the current TD workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)


TD_CHECKPOINT_SCHEMA_VERSION = 1
CheckpointStatus = Literal["completed", "cancelled"]


@dataclass(frozen=True)
class TimeDependentCheckpoint:
    """Minimal state required to resume the current TD algorithm."""

    request: TimeDependentRunRequest
    theta: np.ndarray
    A0: np.ndarray
    completed_steps: int
    requested_steps: int
    current_time: float
    grid_summary: dict[str, float | int | str]
    theta_dtype: str
    A0_dtype: str
    status: CheckpointStatus
    schema_version: int = TD_CHECKPOINT_SCHEMA_VERSION
    workflow: Literal["timedependent"] = "timedependent"


def _request_to_dict(request: TimeDependentRunRequest) -> dict[str, Any]:
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
            "A0": "checkpoint.npz:A0",
            "theta": "checkpoint.npz:theta",
        },
    }


def _request_from_dict(values: dict[str, Any]) -> TimeDependentRunRequest:
    beam_values = values["beams"]
    solver_values = dict(values["solver"])
    workflow = StaticWorkflowOptions(**solver_values.pop("workflow"))
    output_values = dict(values["output"])
    output_values["run_dir"] = None
    return TimeDependentRunRequest(
        grid=GridSpec(**values["grid"]),
        material=LCMaterial(**values["material"]),
        bias=BiasSpec(**values["bias"]),
        beams=BeamStack(
            channels=tuple(
                BeamChannel(**channel) for channel in beam_values["channels"]
            ),
            coherence=beam_values["coherence"],
        ),
        solver=TimeDependentSolverOptions(workflow=workflow, **solver_values),
        output=OutputOptions(**output_values),
        runtime=RuntimeOptions(**values["runtime"]),
    )


def _validate_checkpoint(checkpoint: TimeDependentCheckpoint) -> None:
    if checkpoint.schema_version != TD_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported TD checkpoint schema version: "
            f"{checkpoint.schema_version}"
        )
    if checkpoint.workflow != "timedependent":
        raise ValueError("invalid TD checkpoint workflow")
    if checkpoint.status not in ("completed", "cancelled"):
        raise ValueError(f"invalid TD checkpoint status: {checkpoint.status}")
    if checkpoint.completed_steps < 0:
        raise ValueError("completed_steps must be nonnegative")
    if checkpoint.requested_steps < checkpoint.completed_steps:
        raise ValueError("requested_steps must be >= completed_steps")
    if checkpoint.theta.ndim != 3:
        raise ValueError("checkpoint theta must have shape (Nz, Nx, Ny)")
    if checkpoint.A0.ndim != 3:
        raise ValueError("checkpoint A0 must have shape (Nch, Nx, Ny)")
    if checkpoint.theta.shape[1:] != checkpoint.A0.shape[1:]:
        raise ValueError("checkpoint theta and A0 spatial shapes must match")
    if str(checkpoint.theta.dtype) != checkpoint.theta_dtype:
        raise ValueError("checkpoint theta dtype metadata does not match array")
    if str(checkpoint.A0.dtype) != checkpoint.A0_dtype:
        raise ValueError("checkpoint A0 dtype metadata does not match array")


def save_timedependent_checkpoint(
    checkpoint: TimeDependentCheckpoint,
    run_dir,
) -> Path:
    """Save a checkpoint using relative JSON/NPZ files in ``run_dir``."""

    _validate_checkpoint(checkpoint)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    request_document = {
        "workflow": "timedependent",
        "schema_version": checkpoint.schema_version,
        "request": _request_to_dict(checkpoint.request),
    }
    provenance = {
        "workflow": "timedependent",
        "schema_version": checkpoint.schema_version,
        "status": checkpoint.status,
        "completed_steps": checkpoint.completed_steps,
        "requested_steps": checkpoint.requested_steps,
        "current_time": checkpoint.current_time,
        "completed_units": checkpoint.completed_steps,
        "current_coordinate": checkpoint.current_time,
        "coordinate_name": "t",
        "coordinate_unit": "",
        "grid_summary": checkpoint.grid_summary,
        "theta_dtype": checkpoint.theta_dtype,
        "A0_dtype": checkpoint.A0_dtype,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "request": "request.json",
            "arrays": "checkpoint.npz",
            "provenance": "provenance.json",
        },
    }

    request_path = directory / "request.json"
    request_temporary = request_path.with_suffix(".json.tmp")
    request_temporary.write_text(
        json.dumps(request_document, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    request_temporary.replace(request_path)
    arrays_path = directory / "checkpoint.npz"
    arrays_temporary = arrays_path.with_suffix(".npz.tmp")
    with arrays_temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=np.asarray(checkpoint.theta),
            A0=np.asarray(checkpoint.A0),
        )
    arrays_temporary.replace(arrays_path)
    provenance_path = directory / "provenance.json"
    provenance_temporary = provenance_path.with_suffix(".json.tmp")
    provenance_temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance_temporary.replace(provenance_path)
    return directory


def load_timedependent_checkpoint(run_dir) -> TimeDependentCheckpoint:
    """Load and validate a checkpoint directory."""

    directory = Path(run_dir)
    request_document = json.loads(
        (directory / "request.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    schema_version = int(provenance["schema_version"])
    if int(request_document["schema_version"]) != schema_version:
        raise ValueError("TD checkpoint schema versions do not agree")

    with np.load(directory / "checkpoint.npz", allow_pickle=False) as arrays:
        theta = np.asarray(arrays["theta"]).copy()
        A0 = np.asarray(arrays["A0"]).copy()

    checkpoint = TimeDependentCheckpoint(
        request=_request_from_dict(request_document["request"]),
        theta=theta,
        A0=A0,
        completed_steps=int(provenance["completed_steps"]),
        requested_steps=int(provenance["requested_steps"]),
        current_time=float(provenance["current_time"]),
        grid_summary=dict(provenance["grid_summary"]),
        theta_dtype=str(provenance["theta_dtype"]),
        A0_dtype=str(provenance["A0_dtype"]),
        status=str(provenance["status"]),
        schema_version=schema_version,
    )
    _validate_checkpoint(checkpoint)
    return checkpoint


__all__ = [
    "CheckpointStatus",
    "TD_CHECKPOINT_SCHEMA_VERSION",
    "TimeDependentCheckpoint",
    "load_timedependent_checkpoint",
    "save_timedependent_checkpoint",
]
