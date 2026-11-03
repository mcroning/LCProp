import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "checks"
    / "pr_static_image_amplification_gpu_pilot.py"
)


def _load_driver():
    name = "lcprop_pr_static_image_amplification_gpu_pilot_check"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> str:
    y, x = np.mgrid[:24, :24]
    pixels = (((x // 3) + (y // 4)) % 2 * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_td_reference(path: Path) -> str:
    record = {
        "git_sha": "transient-reference-sha",
        "scheduler": {"slurm_job_id": "transient-reference-job"},
        "metrics": {
            "analytic_absolute_signal_gain": 9.9,
            "measured_absolute_signal_gain": 4.5,
            "image_intensity_correlation": 0.98,
            "zero_response_image_intensity_correlation": 1.0,
            "normalized_image_rmse": 0.18,
            "normalized_power_relative_drift": 0.0,
            "finite_outputs": True,
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_threshold_miss_preserves_operational_success(
    tmp_path,
    monkeypatch,
):
    driver = _load_driver()
    image_path = tmp_path / "target.png"
    td_reference_path = tmp_path / "td_metrics.json"
    monkeypatch.setattr(driver, "IMAGE_SHA256", _write_image(image_path))
    monkeypatch.setattr(
        driver,
        "TD_REFERENCE_SHA256",
        _write_td_reference(td_reference_path),
    )
    monkeypatch.setattr(driver, "MINIMUM_GAIN", 1.0e6)

    record = driver.run_pilot(
        image_path=image_path,
        td_reference_path=td_reference_path,
        output_dir=tmp_path / "products",
        cpu_only=True,
    )

    assert record["status"] == "passed"
    assert record["operational_status"] == "passed"
    assert record["passed"]
    assert record["exit_code"] == 0
    assert record["scientific_assessment"]["classification"] == (
        "not_supported"
    )
    assert not record["scientific_assessment"]["controls"][
        "signal_amplification"
    ]["supported"]
    assert record["metrics"]["measured_absolute_signal_gain"] < 1.0e6
    assert record["static_diagnostics"]["converged"]
    assert "products" in record
    assert "error" not in record


def test_operational_failure_leaves_scientific_assessment_unevaluated(
    tmp_path,
    monkeypatch,
):
    driver = _load_driver()
    image_path = tmp_path / "target.png"
    td_reference_path = tmp_path / "td_metrics.json"
    monkeypatch.setattr(driver, "IMAGE_SHA256", _write_image(image_path))
    monkeypatch.setattr(
        driver,
        "TD_REFERENCE_SHA256",
        _write_td_reference(td_reference_path),
    )

    def reject_operational_outcome(*args, **kwargs):
        raise RuntimeError("controlled operational validation failure")

    monkeypatch.setattr(
        driver,
        "_validate_operational_outcome",
        reject_operational_outcome,
    )

    record = driver.run_pilot(
        image_path=image_path,
        td_reference_path=td_reference_path,
        output_dir=tmp_path / "products",
        cpu_only=True,
    )

    assert record["status"] == "failed"
    assert record["operational_status"] == "failed"
    assert not record["passed"]
    assert record["exit_code"] == 1
    assert record["scientific_assessment"] == {
        "classification": "not_evaluated",
        "controls": {},
    }
    assert record["metrics"]["finite_outputs"]
    assert record["static_diagnostics"]["converged"]
    assert "products" not in record
    assert record["error"]["message"] == (
        "controlled operational validation failure"
    )
