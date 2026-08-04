"""Versioned disk codec for PR time-dependent checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_checkpoint,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_MATERIAL_ID,
    PR_TIMEDEPENDENT_WORKFLOW,
)


PR_CHECKPOINT_SCHEMA_VERSION = 1
PR_CHECKPOINT_MATERIAL = PR_MATERIAL_ID
PR_CHECKPOINT_FORMAT = "lcprop-checkpoint"


def _request_to_dict(request: PRRunRequest) -> dict[str, Any]:
    return {
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": {
            "coherence": request.beams.coherence,
            "channels": [asdict(channel) for channel in request.beams.channels],
        },
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "initial_conditions": {
            "A0": "checkpoint.npz:A0",
            "E_initial": "checkpoint.npz:E_initial",
            "E_current": "checkpoint.npz:E_current",
        },
    }


def _request_from_dict(values: dict[str, Any]) -> PRRunRequest:
    beam_values = values["beams"]
    request = PRRunRequest(
        grid=GridSpec(**values["grid"]),
        material=PRMaterialSpec(**values["material"]),
        beams=BeamStack(
            channels=tuple(
                BeamChannel(**channel) for channel in beam_values["channels"]
            ),
            coherence=beam_values["coherence"],
        ),
        solver=PRSolverOptions(**values["solver"]),
        backend=BackendSpec(**values["backend"]),
    )
    request.grid.validate()
    request.material.validate()
    request.beams.validate()
    request.solver.validate()
    request.backend.validate()
    return request


def _identity(document: dict[str, Any], *, name: str) -> int:
    if document.get("format") != PR_CHECKPOINT_FORMAT:
        raise ValueError(f"invalid PR checkpoint format in {name}")
    if document.get("material") != PR_CHECKPOINT_MATERIAL:
        raise ValueError(f"invalid PR checkpoint material in {name}")
    if document.get("workflow") != PR_TIMEDEPENDENT_WORKFLOW:
        raise ValueError(f"invalid PR checkpoint workflow in {name}")
    version = int(document["schema_version"])
    if version != PR_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported PR checkpoint schema version: {version}")
    return version


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_pr_checkpoint(
    checkpoint: PRTimeDependentCheckpoint,
    run_dir: str | Path,
) -> Path:
    """Encode a validated PR checkpoint in a portable JSON/NPZ directory."""

    validate_pr_checkpoint(checkpoint)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    identity = {
        "format": PR_CHECKPOINT_FORMAT,
        "material": PR_CHECKPOINT_MATERIAL,
        "workflow": PR_TIMEDEPENDENT_WORKFLOW,
        "schema_version": PR_CHECKPOINT_SCHEMA_VERSION,
    }
    request_document = {
        **identity,
        "request": _request_to_dict(checkpoint.request),
    }
    provenance = {
        **identity,
        "status": checkpoint.status,
        "completed_steps": int(checkpoint.completed_steps),
        "requested_steps": int(checkpoint.requested_steps),
        "time_normalized": float(checkpoint.time_normalized),
        "completed_units": int(checkpoint.completed_steps),
        "current_coordinate": float(checkpoint.time_normalized),
        "coordinate_name": "t_normalized",
        "coordinate_unit": "",
        "grid_summary": checkpoint.grid_summary,
        "E_dtype": checkpoint.E_dtype,
        "A0_dtype": checkpoint.A0_dtype,
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
        np.savez_compressed(
            stream,
            E_initial=np.asarray(asnumpy(checkpoint.E_initial)),
            E_current=np.asarray(asnumpy(checkpoint.E_current)),
            A0=np.asarray(asnumpy(checkpoint.A0)),
        )
    arrays_temporary.replace(arrays_path)
    _atomic_json(directory / "request.json", request_document)
    _atomic_json(directory / "provenance.json", provenance)
    return directory


def load_pr_checkpoint(run_dir: str | Path) -> PRTimeDependentCheckpoint:
    """Decode and validate a PR checkpoint directory."""

    directory = Path(run_dir)
    request_document = json.loads(
        (directory / "request.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    request_version = _identity(request_document, name="request.json")
    provenance_version = _identity(provenance, name="provenance.json")
    if request_version != provenance_version:
        raise ValueError("PR checkpoint schema versions do not agree")

    with np.load(directory / "checkpoint.npz", allow_pickle=False) as arrays:
        expected_arrays = {"E_initial", "E_current", "A0"}
        if set(arrays.files) != expected_arrays:
            raise ValueError(
                "PR checkpoint arrays must be exactly E_initial, E_current, and A0"
            )
        E_initial = np.asarray(arrays["E_initial"]).copy()
        E_current = np.asarray(arrays["E_current"]).copy()
        A0 = np.asarray(arrays["A0"]).copy()

    checkpoint = PRTimeDependentCheckpoint(
        request=_request_from_dict(request_document["request"]),
        E_initial=E_initial,
        E_current=E_current,
        A0=A0,
        completed_steps=int(provenance["completed_steps"]),
        requested_steps=int(provenance["requested_steps"]),
        time_normalized=float(provenance["time_normalized"]),
        grid_summary=dict(provenance["grid_summary"]),
        E_dtype=str(provenance["E_dtype"]),
        A0_dtype=str(provenance["A0_dtype"]),
        status=str(provenance["status"]),
    )
    validate_pr_checkpoint(checkpoint)
    return checkpoint


__all__ = [
    "PR_CHECKPOINT_FORMAT",
    "PR_CHECKPOINT_MATERIAL",
    "PR_CHECKPOINT_SCHEMA_VERSION",
    "load_pr_checkpoint",
    "save_pr_checkpoint",
]
