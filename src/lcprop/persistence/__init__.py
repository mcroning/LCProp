"""Portable local persistence helpers."""

from lcprop.lc import LC_MATERIAL_ID
from lcprop.persistence.composition import (
    CheckpointCodec,
    CheckpointCodecRegistry,
)
from lcprop.persistence.experiments import (
    EXPERIMENT_FILE_EXTENSION,
    EXPERIMENT_FORMAT,
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentCodecRegistry,
    ExperimentError,
    ExperimentFormatError,
    ExperimentMaterialError,
    ExperimentPayloadError,
    ExperimentPresentationError,
    ExperimentRequestCodec,
    ExperimentRuntimeStateError,
    ExperimentSchemaError,
    ExperimentWorkflowError,
    LoadedExperiment,
    read_experiment_file,
    write_experiment_file,
)

from lcprop.lc.persistence.timedependent import (
    TD_CHECKPOINT_SCHEMA_VERSION,
    TimeDependentCheckpoint,
    load_timedependent_checkpoint,
    save_timedependent_checkpoint,
)
from lcprop.lc.persistence.static import (
    STATIC_CHECKPOINT_SCHEMA_VERSION,
    StaticCheckpoint,
    load_static_checkpoint,
    save_static_checkpoint,
)
from lcprop.pr.checkpoint import PRTimeDependentCheckpoint
from lcprop.pr.persistence import load_pr_checkpoint, save_pr_checkpoint
from lcprop.pr.specs import PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.lc.experiment_codec import (
    LC_STATIC_EXPERIMENT_CODEC,
    LC_TIMEDEPENDENT_EXPERIMENT_CODEC,
)
from lcprop.pr.experiment_codec import (
    PR_STATIC_EXPERIMENT_CODEC,
    PR_TIMEDEPENDENT_EXPERIMENT_CODEC,
)


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

EXPERIMENT_CODECS = ExperimentCodecRegistry()
for _experiment_codec in (
    LC_STATIC_EXPERIMENT_CODEC,
    LC_TIMEDEPENDENT_EXPERIMENT_CODEC,
    PR_STATIC_EXPERIMENT_CODEC,
    PR_TIMEDEPENDENT_EXPERIMENT_CODEC,
):
    EXPERIMENT_CODECS.register(_experiment_codec)


def save_run_checkpoint(checkpoint, run_dir):
    """Save a checkpoint through its explicitly registered material codec."""

    return CHECKPOINT_CODECS.save_checkpoint(checkpoint, run_dir)


def load_run_checkpoint(run_dir):
    """Load a material-identified or supported legacy checkpoint."""

    return CHECKPOINT_CODECS.load_checkpoint(run_dir)


def save_experiment(
    request,
    path,
    *,
    material_id,
    workflow_id,
    presentation_payload=None,
):
    """Save one request through the registered material/workflow codec."""

    return write_experiment_file(
        path,
        request,
        material_id=material_id,
        workflow_id=workflow_id,
        registry=EXPERIMENT_CODECS,
        presentation_payload=presentation_payload,
    )


def load_experiment(path, *, expected_material_id=None) -> LoadedExperiment:
    """Load one experiment with optional active-material validation."""

    return read_experiment_file(
        path,
        registry=EXPERIMENT_CODECS,
        expected_material_id=expected_material_id,
    )

__all__ = [
    "CHECKPOINT_CODECS",
    "EXPERIMENT_CODECS",
    "EXPERIMENT_FILE_EXTENSION",
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_SCHEMA_VERSION",
    "CheckpointCodec",
    "CheckpointCodecRegistry",
    "ExperimentCodecRegistry",
    "ExperimentError",
    "ExperimentFormatError",
    "ExperimentMaterialError",
    "ExperimentPayloadError",
    "ExperimentPresentationError",
    "ExperimentRequestCodec",
    "ExperimentRuntimeStateError",
    "ExperimentSchemaError",
    "ExperimentWorkflowError",
    "LC_STATIC_CHECKPOINT_CODEC",
    "LC_STATIC_EXPERIMENT_CODEC",
    "LC_TIMEDEPENDENT_CHECKPOINT_CODEC",
    "LC_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "LoadedExperiment",
    "PR_STATIC_EXPERIMENT_CODEC",
    "PR_TIMEDEPENDENT_CHECKPOINT_CODEC",
    "PR_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "TD_CHECKPOINT_SCHEMA_VERSION",
    "TimeDependentCheckpoint",
    "load_timedependent_checkpoint",
    "save_timedependent_checkpoint",
    "STATIC_CHECKPOINT_SCHEMA_VERSION",
    "StaticCheckpoint",
    "load_static_checkpoint",
    "save_static_checkpoint",
    "load_run_checkpoint",
    "load_experiment",
    "save_run_checkpoint",
    "save_experiment",
]
