"""Portable local persistence helpers."""

from lcprop.lc import LC_MATERIAL_ID
from lcprop.persistence.composition import (
    CheckpointCodec,
    CheckpointCodecRegistry,
)

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
from lcprop.pr.checkpoint import PRTimeDependentCheckpoint
from lcprop.pr.persistence import load_pr_checkpoint, save_pr_checkpoint
from lcprop.pr.specs import PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW


LC_STATIC_CHECKPOINT_CODEC = CheckpointCodec(
    material_id=LC_MATERIAL_ID,
    workflow_id="static",
    checkpoint_type=StaticCheckpoint,
    save=save_static_checkpoint,
    load=load_static_checkpoint,
)
LC_TIMEDEPENDENT_CHECKPOINT_CODEC = CheckpointCodec(
    material_id=LC_MATERIAL_ID,
    workflow_id="timedependent",
    checkpoint_type=TimeDependentCheckpoint,
    save=save_timedependent_checkpoint,
    load=load_timedependent_checkpoint,
)
PR_TIMEDEPENDENT_CHECKPOINT_CODEC = CheckpointCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    checkpoint_type=PRTimeDependentCheckpoint,
    save=save_pr_checkpoint,
    load=load_pr_checkpoint,
)

CHECKPOINT_CODECS = CheckpointCodecRegistry()
for _codec in (
    LC_STATIC_CHECKPOINT_CODEC,
    LC_TIMEDEPENDENT_CHECKPOINT_CODEC,
    PR_TIMEDEPENDENT_CHECKPOINT_CODEC,
):
    CHECKPOINT_CODECS.register(_codec)
CHECKPOINT_CODECS.register_legacy_workflow("static", material_id=LC_MATERIAL_ID)
CHECKPOINT_CODECS.register_legacy_workflow(
    "timedependent",
    material_id=LC_MATERIAL_ID,
)


def save_run_checkpoint(checkpoint, run_dir):
    """Save a checkpoint through its explicitly registered material codec."""

    return CHECKPOINT_CODECS.save_checkpoint(checkpoint, run_dir)


def load_run_checkpoint(run_dir):
    """Load a material-identified or supported legacy checkpoint."""

    return CHECKPOINT_CODECS.load_checkpoint(run_dir)

__all__ = [
    "CHECKPOINT_CODECS",
    "CheckpointCodec",
    "CheckpointCodecRegistry",
    "LC_STATIC_CHECKPOINT_CODEC",
    "LC_TIMEDEPENDENT_CHECKPOINT_CODEC",
    "PR_TIMEDEPENDENT_CHECKPOINT_CODEC",
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
