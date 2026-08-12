from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from lcprop.pr.checkpoint import validate_pr_checkpoint
from lcprop.pr.persistence import (
    PR_CHECKPOINT_FORMAT,
    PR_CHECKPOINT_MATERIAL,
    PR_CHECKPOINT_SCHEMA_VERSION,
    load_pr_checkpoint,
    save_pr_checkpoint,
)
from lcprop.pr.specs import (
    PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.workflow import (
    continue_pr_timedependent,
    run_pr_timedependent,
)
from tests.test_pr_checkpoint import (
    _assert_same_cumulative_physics,
    _request_with_initial_state,
)


def _rewrite_json(path, change) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_pr_checkpoint_disk_round_trip_preserves_physical_state_and_request(
    tmp_path,
):
    result = run_pr_timedependent(_request_with_initial_state(steps=2))

    returned = save_pr_checkpoint(result.checkpoint, tmp_path)
    loaded = load_pr_checkpoint(tmp_path)

    assert returned == tmp_path
    assert {path.name for path in tmp_path.iterdir()} == {
        "request.json",
        "checkpoint.npz",
        "provenance.json",
    }
    validate_pr_checkpoint(loaded)
    assert loaded.request == result.checkpoint.request
    assert loaded.request.initial_A is None
    assert loaded.request.initial_E is None
    assert loaded.completed_steps == 2
    assert loaded.requested_steps == 2
    assert loaded.time_normalized == result.checkpoint.time_normalized
    assert loaded.status == "completed"
    assert loaded.E_dtype == result.checkpoint.E_dtype
    assert loaded.A0_dtype == result.checkpoint.A0_dtype
    np.testing.assert_array_equal(loaded.E_initial, result.checkpoint.E_initial)
    np.testing.assert_array_equal(loaded.E_current, result.checkpoint.E_current)
    np.testing.assert_array_equal(loaded.A0, result.checkpoint.A0)

    request_document = json.loads((tmp_path / "request.json").read_text())
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    for document in (request_document, provenance):
        assert document["format"] == PR_CHECKPOINT_FORMAT
        assert document["material"] == PR_CHECKPOINT_MATERIAL
        assert document["workflow"] == PR_TIMEDEPENDENT_WORKFLOW
        assert document["schema_version"] == PR_CHECKPOINT_SCHEMA_VERSION
    assert request_document["request"]["beams"]["channels"][0][
        "tilt_y_rad_per_um"
    ] == result.checkpoint.request.beams.channels[0].tilt_y_rad_per_um
    assert (
        "theta_weight"
        not in request_document["request"]["beams"]["channels"][0]
    )


@pytest.mark.parametrize("legacy_weight", [1.0, 2.0])
def test_pr_checkpoint_legacy_theta_weight_ingestion_is_explicit(
    tmp_path,
    legacy_weight,
):
    result = run_pr_timedependent(_request_with_initial_state(steps=1))
    save_pr_checkpoint(result.checkpoint, tmp_path)

    def add_legacy_weight(document):
        document["request"]["beams"]["channels"][0][
            "theta_weight"
        ] = legacy_weight

    _rewrite_json(tmp_path / "request.json", add_legacy_weight)

    if legacy_weight == 1.0:
        loaded = load_pr_checkpoint(tmp_path)
        assert not hasattr(loaded.request.beams.channels[0], "theta_weight")
    else:
        with pytest.raises(ValueError, match="legacy non-unit theta_weight"):
            load_pr_checkpoint(tmp_path)


def test_disk_loaded_pr_checkpoint_continues_exactly(tmp_path):
    request = _request_with_initial_state(steps=4)
    first = run_pr_timedependent(
        replace(request, solver=replace(request.solver, Nt=1))
    )
    save_pr_checkpoint(first.checkpoint, tmp_path)

    resumed = continue_pr_timedependent(
        request,
        load_pr_checkpoint(tmp_path),
        additional_steps=3,
    )
    uninterrupted = run_pr_timedependent(request)

    _assert_same_cumulative_physics(resumed, uninterrupted)


def test_disk_loaded_semi_implicit_checkpoint_continues_exactly(tmp_path):
    request = _request_with_initial_state(
        steps=4,
        integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
    )
    first = run_pr_timedependent(
        replace(request, solver=replace(request.solver, Nt=1))
    )
    save_pr_checkpoint(first.checkpoint, tmp_path)

    loaded = load_pr_checkpoint(tmp_path)
    resumed = continue_pr_timedependent(
        request,
        loaded,
        additional_steps=3,
    )
    uninterrupted = run_pr_timedependent(request)

    assert loaded.request.solver.integrator == PR_SEMI_IMPLICIT_INTEGRATOR
    _assert_same_cumulative_physics(resumed, uninterrupted)


def test_schema_version_one_checkpoint_loads_as_explicit_euler(tmp_path):
    result = run_pr_timedependent(_request_with_initial_state(steps=1))
    save_pr_checkpoint(result.checkpoint, tmp_path)

    def make_request_v1(document):
        document["schema_version"] = 1
        document["request"]["solver"].pop("integrator")

    _rewrite_json(tmp_path / "request.json", make_request_v1)
    _rewrite_json(
        tmp_path / "provenance.json",
        lambda document: document.__setitem__("schema_version", 1),
    )

    loaded = load_pr_checkpoint(tmp_path)

    assert loaded.request.solver.integrator == PR_EULER_INTEGRATOR
    np.testing.assert_array_equal(loaded.E_current, result.E_final)


def test_schema_version_two_requires_integrator_identity(tmp_path):
    result = run_pr_timedependent(_request_with_initial_state(steps=1))
    save_pr_checkpoint(result.checkpoint, tmp_path)
    _rewrite_json(
        tmp_path / "request.json",
        lambda document: document["request"]["solver"].pop("integrator"),
    )

    with pytest.raises(ValueError, match="missing integrator identity"):
        load_pr_checkpoint(tmp_path)


def test_schema_version_one_rejects_new_integrator_field(tmp_path):
    result = run_pr_timedependent(_request_with_initial_state(steps=1))
    save_pr_checkpoint(result.checkpoint, tmp_path)
    _rewrite_json(
        tmp_path / "request.json",
        lambda document: document.__setitem__("schema_version", 1),
    )
    _rewrite_json(
        tmp_path / "provenance.json",
        lambda document: document.__setitem__("schema_version", 1),
    )

    with pytest.raises(ValueError, match="must not contain integrator"):
        load_pr_checkpoint(tmp_path)


@pytest.mark.parametrize(
    ("filename", "field", "value", "match"),
    [
        ("request.json", "format", "other", "format"),
        ("request.json", "material", "lc", "material"),
        ("provenance.json", "workflow", "timedependent", "workflow"),
        ("provenance.json", "schema_version", 999, "schema version"),
    ],
)
def test_pr_checkpoint_rejects_wrong_disk_identity(
    tmp_path,
    filename,
    field,
    value,
    match,
):
    checkpoint = run_pr_timedependent(
        _request_with_initial_state(steps=1)
    ).checkpoint
    save_pr_checkpoint(checkpoint, tmp_path)
    _rewrite_json(
        tmp_path / filename,
        lambda document: document.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=match):
        load_pr_checkpoint(tmp_path)


def test_pr_checkpoint_rejects_array_dtype_metadata_mismatch(tmp_path):
    checkpoint = run_pr_timedependent(
        _request_with_initial_state(steps=1)
    ).checkpoint
    save_pr_checkpoint(checkpoint, tmp_path)
    _rewrite_json(
        tmp_path / "provenance.json",
        lambda document: document.__setitem__("E_dtype", "float32"),
    )

    with pytest.raises(ValueError, match="dtype metadata"):
        load_pr_checkpoint(tmp_path)


def test_save_pr_checkpoint_rejects_invalid_state_before_writing(tmp_path):
    checkpoint = run_pr_timedependent(
        _request_with_initial_state(steps=1)
    ).checkpoint
    invalid = replace(checkpoint, E_current=np.asarray(checkpoint.E_current)[:-1])

    with pytest.raises(ValueError, match="shape"):
        save_pr_checkpoint(invalid, tmp_path)
    assert list(tmp_path.iterdir()) == []
