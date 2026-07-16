"""Portable local persistence helpers."""

import json
from pathlib import Path

from lcprop.persistence.timedependent import (
    TD_CHECKPOINT_SCHEMA_VERSION,
    TimeDependentCheckpoint,
    load_timedependent_checkpoint,
    save_timedependent_checkpoint,
)
from lcprop.persistence.static import (
    STATIC_CHECKPOINT_SCHEMA_VERSION,
    StaticCheckpoint,
    load_static_checkpoint,
    save_static_checkpoint,
)


def save_run_checkpoint(checkpoint, run_dir):
    """Save either workflow checkpoint using the shared run-directory layout."""

    if isinstance(checkpoint, StaticCheckpoint):
        return save_static_checkpoint(checkpoint, run_dir)
    if isinstance(checkpoint, TimeDependentCheckpoint):
        return save_timedependent_checkpoint(checkpoint, run_dir)
    raise TypeError(f"unsupported checkpoint type: {type(checkpoint).__name__}")


def load_run_checkpoint(run_dir):
    """Dispatch a checkpoint loader using workflow metadata only."""

    directory = Path(run_dir)
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    workflow = provenance.get("workflow")
    if workflow == "static":
        return load_static_checkpoint(directory)
    if workflow == "timedependent":
        return load_timedependent_checkpoint(directory)
    raise ValueError(f"unsupported or missing checkpoint workflow: {workflow!r}")

__all__ = [
    "TD_CHECKPOINT_SCHEMA_VERSION",
    "TimeDependentCheckpoint",
    "load_timedependent_checkpoint",
    "save_timedependent_checkpoint",
    "STATIC_CHECKPOINT_SCHEMA_VERSION",
    "StaticCheckpoint",
    "load_static_checkpoint",
    "save_static_checkpoint",
    "load_run_checkpoint",
    "save_run_checkpoint",
]
