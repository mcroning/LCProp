from __future__ import annotations

from enum import Enum
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.checks import (
    pr_full_transverse_nonlinear_td_h200_commissioning as harness,
)


class _EvidenceLabel(Enum):
    PRIMARY = "primary"


def _complete_scientific_object():
    primary = {
        "numpy_timing": {
            "scientific_total_seconds": np.float64(1.25),
            "optical_pass_seconds": np.float32(0.25),
            "nonlinear_material_seconds": np.float64(0.5),
            "optical_pass_calls": np.int64(4),
            "instrumented_material_calls": np.int32(3),
        },
        "cupy_timing": {
            "scientific_total_seconds": np.float64(0.75),
            "optical_pass_seconds": np.float64(0.2),
            "nonlinear_material_seconds": np.float64(0.3),
            "optical_pass_calls": 4,
            "instrumented_material_calls": 3,
        },
        "comparisons": {
            "psi": {
                "relative_l2": np.float64(2.0e-14),
                "maximum_absolute": np.float64(4.0e-15),
                "shape": np.array([2, 256, 256], dtype=np.int64),
                "passed": np.bool_(True),
            }
        },
        "diagnostic_comparisons": {
            "nonlinear_td_rhs_max": {
                "relative_l2": np.float64(1.0e-14),
                "maximum_absolute": np.float64(2.0e-15),
                "passed": np.bool_(True),
            }
        },
        "cupy_diagnostics": {
            "carrier_integrals_per_z": np.array([65536.0, 65536.0]),
            "physical_state_valid": np.bool_(True),
        },
    }
    static = {
        "relative_psi_change_after_one_td_step": np.float64(1.0e-13),
        "physical_state_valid": np.bool_(True),
    }
    timestep = {
        "step_counts": (10, 20, 40),
        "successive_difference_ratio": np.float64(1.97),
    }
    cancellation = {
        "completed_steps": np.int64(0),
        "candidate_discarded": np.bool_(True),
    }
    fast = {
        "policy": "fast",
        "omitted_fields": ("psi_initial", "psi_final"),
        "package_bytes": np.int64(4_207_630),
        "files": {"not-retained-in-scientific-rows": "checksum"},
    }
    pool = {
        "before": {"used_bytes": np.int64(0), "total_bytes": np.int64(0)},
        "after": {
            "used_bytes": np.int64(0),
            "total_bytes": np.int64(64 * 1024 * 1024),
        },
    }
    return harness._scientific_evidence_object(
        primary=primary,
        static=static,
        timestep=timestep,
        cancellation=cancellation,
        fast=fast,
        cupy_memory_pool=pool,
    )


def test_job_3662453_ndarray_failure_is_reproduced_and_normalized():
    failed_value = {"carrier_integrals_per_z": np.array([1.0, 2.0])}

    with pytest.raises(TypeError, match="ndarray is not JSON serializable"):
        json.dumps(failed_value)

    encoded = harness._canonical_json(failed_value)
    assert json.loads(encoded) == {"carrier_integrals_per_z": [1.0, 2.0]}


def test_complete_scientific_object_json_write_read_roundtrip(tmp_path):
    evidence = _complete_scientific_object()
    normalized = harness._normalize_json_value(evidence)
    json.dumps(normalized, sort_keys=True, allow_nan=False)

    path = tmp_path / "scientific_rows.json"
    written = harness._write_json(path, evidence, canonical=True)
    retained = json.loads(path.read_text())

    assert retained == written == normalized
    assert retained["primary"]["comparisons"]["psi"]["shape"] == [2, 256, 256]
    assert retained["primary"]["numpy_timing"]["optical_pass_calls"] == 4
    assert retained["cupy_memory_pool"]["after"]["total_bytes"] == 67_108_864
    assert retained["static_sanity"]["physical_state_valid"] is True
    assert retained["timestep_sanity"]["step_counts"] == [10, 20, 40]


def test_normalization_is_explicit_for_paths_enums_and_unsupported_types():
    normalized = harness._normalize_json_value(
        {"path": Path("evidence/data.json"), "label": _EvidenceLabel.PRIMARY}
    )
    assert normalized == {
        "path": "evidence/data.json",
        "label": "primary",
    }

    with pytest.raises(TypeError, match=r"unsupported JSON evidence type at \$\.bad"):
        harness._normalize_json_value({"bad": object()})
    with pytest.raises(TypeError, match="key.*must be str"):
        harness._normalize_json_value({1: "bad key"})
    with pytest.raises(ValueError, match="non-finite.*\\$.value"):
        harness._normalize_json_value({"value": np.float64(np.inf)})


def test_durable_intermediate_survives_late_schema_failure(tmp_path):
    primary_path = tmp_path / "primary_quantitative.json"
    harness._write_json(
        primary_path,
        _complete_scientific_object()["primary"],
    )

    with pytest.raises(TypeError, match="unsupported JSON evidence type"):
        harness._write_json(tmp_path / "late_manifest.json", {"bad": object()})

    retained = json.loads(primary_path.read_text())
    assert retained["comparisons"]["psi"]["passed"] is True
    assert not (tmp_path / "late_manifest.json").exists()
    assert not (tmp_path / "late_manifest.json.tmp").exists()


def test_cupy_values_normalize_conditionally():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    value = {
        "array": cp.asarray([1.0, 2.0], dtype=cp.float64),
        "scalar": cp.asarray(3, dtype=cp.int64),
    }
    assert harness._normalize_json_value(value) == {
        "array": [1.0, 2.0],
        "scalar": 3,
    }
