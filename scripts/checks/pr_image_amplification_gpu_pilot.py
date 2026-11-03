#!/usr/bin/env python3
"""Run the bounded CuPy PR image-amplification pilot.

Full mode requires a scheduler-assigned GPU and writes compact scientific
products plus a machine-readable metrics record. ``--cpu-only`` substitutes a
small geometry and exists only for pre-submission pipeline validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from time import perf_counter
import traceback
from typing import Any

import numpy as np
from PIL import Image

from lcprop.core.backend import BackendSpec
from lcprop.pr.image_amplification import (
    PRImageAmplificationSpec,
    run_image_amplification,
)


IMAGE_SHA256 = "e424f1070587d79d2a91f4a3bd14e7c11e80adfe7c723e45432d9d4fc5bb85b7"
POWER_DRIFT_LIMIT = 1.0e-9
MINIMUM_GAIN = 1.1
MINIMUM_CORRELATION = 0.8
MAXIMUM_NORMALIZED_RMSE = 1.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def pilot_spec() -> PRImageAmplificationSpec:
    """Return the single approved scaled-pilot parameter set."""

    return PRImageAmplificationSpec(
        Nx=256,
        Ny=256,
        x_aperture_um=256.0,
        y_aperture_um=256.0,
        interaction_length_um=400.0,
        dz_um=20.0,
        wavelength_um=0.633,
        refractive_index=2.4,
        positive_mode_index=8,
        beam_waist_um=48.0,
        image_size_factor=1.0,
        input_peak_ratio=1.0e-3,
        saturated_small_signal_gain=10.0,
        signal_gain_sign=1,
        dark_intensity=0.05,
        characteristic_wavenumber_per_um=0.5,
        Nt=200,
        dt_normalized=0.05,
        invert_image=False,
    )


def cpu_validation_spec() -> PRImageAmplificationSpec:
    """Return a fast geometrically similar request for local validation."""

    return replace(
        pilot_spec(),
        Nx=64,
        Ny=64,
        x_aperture_um=64.0,
        y_aperture_um=64.0,
        interaction_length_um=100.0,
        dz_um=20.0,
        positive_mode_index=2,
        beam_waist_um=12.0,
        Nt=2,
    )


def _read_image(path: Path) -> tuple[np.ndarray, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != IMAGE_SHA256:
        raise ValueError(
            f"image SHA-256 mismatch: expected {IMAGE_SHA256}, received {digest}"
        )
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return image, digest


def _save_normalized_image(values: np.ndarray, path: Path) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"cannot save nonfinite image product {path.name}")
    array = np.maximum(array, 0.0)
    maximum = float(np.max(array))
    if maximum <= 0.0:
        raise ValueError(f"cannot save zero image product {path.name}")
    pixels = np.rint(255.0 * array / maximum).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _scheduler_record() -> dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _require_gpu_allocation(record: dict[str, Any]) -> None:
    scheduler = record["scheduler"]
    if not scheduler["slurm_job_id"]:
        raise RuntimeError("GPU pilot requires a Slurm allocation")
    markers = (
        scheduler["slurm_job_gpus"],
        scheduler["slurm_gpus_on_node"],
        scheduler["cuda_visible_devices"],
    )
    if not any(value not in (None, "", "NoDevFiles", "-1") for value in markers):
        raise RuntimeError("Slurm allocation does not expose an assigned GPU")


def _initialize_cupy() -> tuple[Any, dict[str, Any]]:
    started = perf_counter()
    import cupy as cp

    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy reports no CUDA devices")
    device = cp.cuda.Device(0)
    device.use()
    probe = cp.arange(32, dtype=cp.float64)
    probe.sum()
    cp.cuda.get_current_stream().synchronize()
    cold_seconds = perf_counter() - started
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return cp, {
        "device_count": count,
        "device_id": int(device.id),
        "device_name": str(name),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": str(cp.__version__),
        "cold_initialization_seconds": cold_seconds,
        "free_memory_before_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def _base_record(mode: str, image_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "lcprop.pr_image_amplification_gpu_pilot.v1",
        "mode": mode,
        "status": "failed",
        "passed": False,
        "exit_code": 1,
        "started_utc": _utc_now(),
        "ended_utc": None,
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "repository": str(_repository_root()),
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "scheduler": _scheduler_record(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pillow": importlib.metadata.version("pillow"),
            "lcprop": importlib.metadata.version("lcprop"),
        },
        "limits": {
            "power_relative_drift": POWER_DRIFT_LIMIT,
            "minimum_gain": MINIMUM_GAIN,
            "minimum_image_correlation": MINIMUM_CORRELATION,
            "maximum_normalized_rmse": MAXIMUM_NORMALIZED_RMSE,
        },
    }


def _validate_result(result: Any, *, expected_backend: str) -> None:
    backend = result.run_result.diagnostics["backend"]
    if backend["backend"] != expected_backend:
        raise AssertionError(
            f"requested {expected_backend}, workflow reported {backend!r}"
        )
    if expected_backend == "cupy" and not backend["is_gpu"]:
        raise AssertionError("CuPy workflow did not report is_gpu=True")
    arrays = (
        result.run_result.A_final,
        result.run_result.E_final,
        result.output_signal_field,
        result.backpropagated_signal_field,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise AssertionError("pilot produced nonfinite fields")
    if result.measured_absolute_signal_gain <= MINIMUM_GAIN:
        raise AssertionError("pilot did not produce image-signal amplification")
    if result.image_intensity_correlation < MINIMUM_CORRELATION:
        raise AssertionError("pilot reconstruction correlation is too low")
    if result.normalized_image_rmse > MAXIMUM_NORMALIZED_RMSE:
        raise AssertionError("pilot normalized reconstruction error is too high")
    if abs(result.normalized_power_relative_drift) > POWER_DRIFT_LIMIT:
        raise AssertionError("pilot optical power drift exceeds its limit")


def _write_products(result: Any, image: np.ndarray, output_dir: Path) -> dict[str, int]:
    input_intensity = np.abs(result.input_signal_field) ** 2
    reconstructed_intensity = np.abs(result.backpropagated_signal_field) ** 2
    zero_intensity = np.abs(result.zero_response_backpropagated_signal_field) ** 2
    output_signal_intensity = np.abs(result.output_signal_field) ** 2

    _save_normalized_image(image, output_dir / "input_target.png")
    _save_normalized_image(input_intensity, output_dir / "input_signal.png")
    _save_normalized_image(
        reconstructed_intensity,
        output_dir / "reconstructed_signal.png",
    )
    _save_normalized_image(
        zero_intensity,
        output_dir / "zero_response_reconstruction.png",
    )
    _save_normalized_image(
        output_signal_intensity,
        output_dir / "output_signal.png",
    )
    midpoint = result.run_result.E_final.shape[0] // 2
    np.save(
        output_dir / "E_final_midplane.npy",
        np.asarray(result.run_result.E_final[midpoint], dtype=np.float32),
    )
    np.save(
        output_dir / "reconstructed_signal_intensity.npy",
        np.asarray(reconstructed_intensity, dtype=np.float32),
    )
    return {
        path.name: path.stat().st_size
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }


def run_pilot(
    *,
    image_path: Path,
    output_dir: Path,
    cpu_only: bool,
) -> dict[str, Any]:
    mode = "cpu_only_validation" if cpu_only else "gpu_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _base_record(mode, image_path, output_dir)
    total_started = perf_counter()
    try:
        image, image_digest = _read_image(image_path)
        spec = cpu_validation_spec() if cpu_only else pilot_spec()
        record["image_sha256"] = image_digest
        record["image_shape"] = list(image.shape)
        record["parameters"] = asdict(spec)
        Nz = int(round(spec.interaction_length_um / spec.dz_um))
        record["sampling"] = {
            "Nz": Nz,
            "dx_um": spec.x_aperture_um / spec.Nx,
            "dy_um": spec.y_aperture_um / spec.Ny,
            "samples_per_waist_x": spec.beam_waist_um * spec.Nx / spec.x_aperture_um,
            "samples_per_waist_y": spec.beam_waist_um * spec.Ny / spec.y_aperture_um,
            "grating_samples_per_period": spec.Nx / (2.0 * spec.positive_mode_index),
            "longitudinal_steps": Nz,
        }
        record["memory_estimate"] = {
            "one_float64_E_state_bytes": spec.Nx * spec.Ny * Nz * 8,
        }

        if cpu_only:
            backend = BackendSpec("numpy", "float64", False)
            cp = None
        else:
            _require_gpu_allocation(record)
            cp, gpu = _initialize_cupy()
            record["gpu"] = gpu
            record["versions"]["cupy"] = gpu["cupy_version"]
            backend = BackendSpec("cupy", "float64", False)
            cp.cuda.get_current_stream().synchronize()

        execution_started = perf_counter()
        result = run_image_amplification(image, spec, backend=backend)
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
        synchronized_execution = perf_counter() - execution_started
        expected_backend = "numpy" if cpu_only else "cupy"
        _validate_result(result, expected_backend=expected_backend)

        record["backend"] = {
            "requested": expected_backend,
            "reported": result.run_result.diagnostics["backend"]["backend"],
            "is_gpu": result.run_result.diagnostics["backend"]["is_gpu"],
        }
        record["metrics"] = {
            "analytic_absolute_signal_gain": result.analytic_absolute_signal_gain,
            "measured_absolute_signal_gain": result.measured_absolute_signal_gain,
            "image_intensity_correlation": result.image_intensity_correlation,
            "zero_response_image_intensity_correlation": (
                result.zero_response_image_intensity_correlation
            ),
            "normalized_image_rmse": result.normalized_image_rmse,
            "normalized_power_relative_drift": (
                result.normalized_power_relative_drift
            ),
            "finite_outputs": True,
        }
        record["timing_seconds"] = {
            "synchronized_execution": synchronized_execution,
            "pr_evolution_including_optical_passes": result.pr_workflow_runtime_s,
            "reconstruction_optical_propagation": (
                result.reconstruction_optical_runtime_s
            ),
            "reconstruction_total": result.reconstruction_runtime_s,
            "benchmark_total": result.runtime_s,
        }
        if cp is not None:
            free_after, total_after = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            record["gpu"].update(
                {
                    "free_memory_after_bytes": int(free_after),
                    "total_memory_after_bytes": int(total_after),
                    "memory_pool_used_after_bytes": int(pool.used_bytes()),
                    "memory_pool_peak_proxy_bytes": int(pool.total_bytes()),
                }
            )
        record["products"] = _write_products(result, image, output_dir)
        record.update({"status": "passed", "passed": True, "exit_code": 0})
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        record["total_wall_seconds"] = perf_counter() - total_started
        record["ended_utc"] = _utc_now()
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = run_pilot(
        image_path=args.image,
        output_dir=args.output_dir,
        cpu_only=bool(args.cpu_only),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
