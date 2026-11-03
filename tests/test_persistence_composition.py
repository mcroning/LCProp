from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lcprop.lc import LC_MATERIAL_ID
from lcprop.persistence import (
    CHECKPOINT_CODECS,
    LC_STATIC_CHECKPOINT_CODEC,
    LC_TIMEDEPENDENT_CHECKPOINT_CODEC,
    PR_TIMEDEPENDENT_CHECKPOINT_CODEC,
    CheckpointCodec,
    CheckpointCodecRegistry,
    StaticCheckpoint,
    TimeDependentCheckpoint,
    load_run_checkpoint,
    save_run_checkpoint,
)
from lcprop.pr.checkpoint import PRTimeDependentCheckpoint
from lcprop.pr.specs import PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.workflow import continue_pr_timedependent, run_pr_timedependent
from lcprop.workflows.timedependent import run_timedependent
from tests.test_pr_checkpoint import (
    _assert_same_cumulative_physics,
    _request_with_initial_state,
)
from tests.test_unified_execution import (
    _static_request,
    _stop_static_after,
    _td_request,
)


@dataclass(frozen=True)
class _ExampleCheckpoint:
    value: int


@dataclass(frozen=True)
class _ExampleCheckpointSubclass(_ExampleCheckpoint):
    pass


def _example_codec(
    *,
    material_id: str = "example",
    workflow_id: str = "evolve",
    checkpoint_type: type = _ExampleCheckpoint,
    loaded_value=_ExampleCheckpoint(7),
) -> CheckpointCodec:
    def save(checkpoint, run_dir):
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "provenance.json").write_text(
            json.dumps(
                {"material": material_id, "workflow": workflow_id}
            ),
            encoding="utf-8",
        )
        return directory

    def load(_run_dir):
        return loaded_value

    return CheckpointCodec(
        material_id=material_id,
        workflow_id=workflow_id,
        checkpoint_type=checkpoint_type,
        save=save,
        load=load,
    )


def test_builtin_checkpoint_codecs_have_material_owned_identities():
    assert CHECKPOINT_CODECS.codecs == (
        LC_STATIC_CHECKPOINT_CODEC,
        LC_TIMEDEPENDENT_CHECKPOINT_CODEC,
        PR_TIMEDEPENDENT_CHECKPOINT_CODEC,
    )
    assert LC_STATIC_CHECKPOINT_CODEC.key == (LC_MATERIAL_ID, "static")
    assert LC_TIMEDEPENDENT_CHECKPOINT_CODEC.key == (
        LC_MATERIAL_ID,
        "timedependent",
    )
    assert PR_TIMEDEPENDENT_CHECKPOINT_CODEC.key == (
        PR_MATERIAL_ID,
        PR_TIMEDEPENDENT_WORKFLOW,
    )
    assert LC_STATIC_CHECKPOINT_CODEC.checkpoint_type is StaticCheckpoint
    assert (
        LC_TIMEDEPENDENT_CHECKPOINT_CODEC.checkpoint_type
        is TimeDependentCheckpoint
    )
    assert (
        PR_TIMEDEPENDENT_CHECKPOINT_CODEC.checkpoint_type
        is PRTimeDependentCheckpoint
    )


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"material_id": ""}, ValueError, "material_id"),
        ({"workflow_id": " evolve"}, ValueError, "workflow_id"),
        ({"checkpoint_type": object()}, TypeError, "checkpoint_type"),
        ({"save": None}, TypeError, "save"),
        ({"load": None}, TypeError, "load"),
    ],
)
def test_checkpoint_codec_rejects_invalid_contract(changes, error, match):
    values = {
        "material_id": "example",
        "workflow_id": "evolve",
        "checkpoint_type": _ExampleCheckpoint,
        "save": lambda checkpoint, run_dir: Path(run_dir),
        "load": lambda run_dir: _ExampleCheckpoint(1),
    }
    values.update(changes)
    with pytest.raises(error, match=match):
        CheckpointCodec(**values)


def test_registry_rejects_duplicate_identity_and_checkpoint_type():
    registry = CheckpointCodecRegistry()
    registry.register(_example_codec())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_example_codec(checkpoint_type=dict))
    with pytest.raises(ValueError, match="checkpoint type already registered"):
        registry.register(
            _example_codec(material_id="other", workflow_id="other")
        )


def test_registry_requires_registered_target_for_legacy_workflow():
    registry = CheckpointCodecRegistry()
    with pytest.raises(ValueError, match="not a registered"):
        registry.register_legacy_workflow(
            "evolve",
            material_id="example",
        )


def test_materialless_load_requires_one_explicit_legacy_alias(tmp_path):
    registry = CheckpointCodecRegistry()
    registry.register(_example_codec())
    registry.save_checkpoint(_ExampleCheckpoint(1), tmp_path)
    path = tmp_path / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance.pop("material")
    path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported legacy"):
        registry.load_checkpoint(tmp_path)

    registry.register_legacy_workflow("evolve", material_id="example")
    assert registry.load_checkpoint(tmp_path) == _ExampleCheckpoint(7)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_legacy_workflow("evolve", material_id="example")


def test_registry_validates_loaded_checkpoint_type(tmp_path):
    registry = CheckpointCodecRegistry()
    registry.register(_example_codec(loaded_value=object()))
    registry.save_checkpoint(_ExampleCheckpoint(1), tmp_path)

    with pytest.raises(TypeError, match="incompatible type"):
        registry.load_checkpoint(tmp_path)


def test_registry_preserves_subclass_save_compatibility(tmp_path):
    registry = CheckpointCodecRegistry()
    registry.register(_example_codec())

    returned = registry.save_checkpoint(_ExampleCheckpointSubclass(3), tmp_path)

    assert returned == tmp_path
    assert (tmp_path / "provenance.json").is_file()


def test_shared_dispatch_round_trips_pr_and_continues_exactly(tmp_path):
    request = _request_with_initial_state(steps=3)
    first = run_pr_timedependent(
        replace(request, solver=replace(request.solver, Nt=1))
    )

    save_run_checkpoint(first.checkpoint, tmp_path)
    loaded = load_run_checkpoint(tmp_path)
    resumed = continue_pr_timedependent(request, loaded, additional_steps=2)
    uninterrupted = run_pr_timedependent(request)

    assert isinstance(loaded, PRTimeDependentCheckpoint)
    _assert_same_cumulative_physics(resumed, uninterrupted)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["material"] == PR_MATERIAL_ID
    assert provenance["workflow"] == PR_TIMEDEPENDENT_WORKFLOW


def test_shared_dispatch_loads_legacy_lc_static_without_format_change(tmp_path):
    stopped, _ = _stop_static_after(_static_request(), 1)

    save_run_checkpoint(stopped.checkpoint, tmp_path)
    loaded = load_run_checkpoint(tmp_path)

    assert isinstance(loaded, StaticCheckpoint)
    np.testing.assert_array_equal(loaded.A_next, stopped.checkpoint.A_next)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    request_document = json.loads((tmp_path / "request.json").read_text())
    assert provenance["workflow"] == "static"
    assert "material" not in provenance
    assert "format" not in provenance
    assert "material" not in request_document
    assert "format" not in request_document


def test_static_checkpoint_serialization_omits_removed_theta_weight(tmp_path):
    stopped, _ = _stop_static_after(_static_request(), 1)
    save_run_checkpoint(stopped.checkpoint, tmp_path)

    request_document = json.loads((tmp_path / "request.json").read_text())

    assert (
        "theta_weight"
        not in request_document["request"]["beams"]["channels"][0]
    )


@pytest.mark.parametrize("legacy_weight", [1.0, 2.0])
def test_static_checkpoint_legacy_theta_weight_ingestion_is_explicit(
    tmp_path,
    legacy_weight,
):
    stopped, _ = _stop_static_after(_static_request(), 1)
    save_run_checkpoint(stopped.checkpoint, tmp_path)
    request_path = tmp_path / "request.json"
    provenance_path = tmp_path / "provenance.json"
    request_document = json.loads(request_path.read_text())
    request_values = request_document["request"]
    request_values["beams"]["channels"][0]["theta_weight"] = legacy_weight
    serialized = json.dumps(
        request_values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance = json.loads(provenance_path.read_text())
    provenance["request_fingerprint"] = hashlib.sha256(serialized).hexdigest()
    request_path.write_text(json.dumps(request_document), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    if legacy_weight == 1.0:
        loaded = load_run_checkpoint(tmp_path)
        assert not hasattr(loaded.request.beams.channels[0], "theta_weight")
    else:
        with pytest.raises(ValueError, match="legacy non-unit theta_weight"):
            load_run_checkpoint(tmp_path)


def test_shared_dispatch_loads_legacy_lc_timedependent(tmp_path):
    base_request = _static_request(slices=2)
    result = run_timedependent(_td_request(base_request))

    save_run_checkpoint(result.checkpoint, tmp_path)
    loaded = load_run_checkpoint(tmp_path)

    assert isinstance(loaded, TimeDependentCheckpoint)
    np.testing.assert_array_equal(loaded.theta, result.checkpoint.theta)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["workflow"] == "timedependent"
    assert "material" not in provenance


def test_shared_dispatch_rejects_cross_material_identity(tmp_path):
    checkpoint = run_pr_timedependent(
        _request_with_initial_state(steps=1)
    ).checkpoint
    save_run_checkpoint(checkpoint, tmp_path)
    path = tmp_path / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["material"] = LC_MATERIAL_ID
    path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported checkpoint codec identity"):
        load_run_checkpoint(tmp_path)


def test_shared_dispatch_rejects_unknown_checkpoint_type(tmp_path):
    with pytest.raises(TypeError, match="unsupported checkpoint type"):
        save_run_checkpoint(object(), tmp_path)
