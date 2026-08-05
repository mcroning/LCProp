#!/usr/bin/env python3
"""Fail-hard CPU/CuPy commissioning check for the PR workflow.

The default mode requires a Slurm allocation and a real CUDA device.  It runs
one identical, nontrivial PR request on NumPy and CuPy, compares the resulting
optical/material fields, and writes a self-contained JSON record.  The
``--cpu-only`` mode exists solely for local validation before cluster
submission; it never claims that GPU commissioning passed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.workflow import run_pr_timedependent


EXPECTED_NX = 32
EXPECTED_NY = 32
OPTICAL_RTOL = 1.0e-9
OPTICAL_ATOL = 1.0e-10
PR_RTOL = 1.0e-9
PR_ATOL = 1.0e-10
POWER_DRIFT_LIMIT = 1.0e-9


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_sha(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initial_optical_field(grid: GridSpec) -> np.ndarray:
    dx_um = grid.x_aperture_um / grid.Nx
    dy_um = grid.y_aperture_um / grid.Ny
    x_um = (np.arange(grid.Nx) - 0.5 * grid.Nx + 0.5) * dx_um
    y_um = (np.arange(grid.Ny) - 0.5 * grid.Ny + 0.5) * dy_um
    X_um, Y_um = np.meshgrid(x_um, y_um, indexing="ij")
    envelope = np.exp(
        -(
            ((X_um + 3.0) / 10.0) ** 2
            + ((Y_um - 2.0) / 8.0) ** 2
        )
    )
    phase = 0.08 * X_um - 0.05 * Y_um + 0.15 * np.cos(
        2.0 * np.pi * X_um / grid.x_aperture_um
    )
    return (envelope * np.exp(1j * phase))[None, ...].astype(np.complex128)


def build_request() -> PRRunRequest:
    """Return the single deterministic request shared by CPU and GPU runs."""

    grid = GridSpec(
        Nx=EXPECTED_NX,
        Ny=EXPECTED_NY,
        x_aperture_um=64.0,
        y_aperture_um=64.0,
        dz_um=2.0,
        z_length_um=8.0,
    )
    beam = BeamChannel(
        name="GPU commissioning beam",
        wavelength_um=0.633,
        power_mW=1.0,
        waist_x_um=10.0,
        waist_y_um=8.0,
        x0_um=-3.0,
        y0_um=2.0,
        tilt_x_rad_per_um=0.08,
        tilt_y_rad_per_um=-0.05,
        coherence_group="gpu-smoke",
    )
    material = PRMaterialSpec(
        dark_intensity=0.2,
        uniform_background_intensity=0.1,
        applied_field=0.5,
        gain_length_product=0.4,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.1,
    )
    solver = PRSolverOptions(
        Nt=2,
        dt_normalized=0.02,
        optical_substeps=2,
        integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
    )
    return PRRunRequest(
        grid=grid,
        beams=BeamStack(channels=(beam,)),
        material=material,
        solver=solver,
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
        initial_A=_initial_optical_field(grid),
    )


def _power_drift(result: Any) -> float:
    denominator = max(abs(float(result.power_initial)), np.finfo(float).tiny)
    return abs(float(result.power_final) - float(result.power_initial)) / denominator


def _difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(candidate) - np.asarray(reference))
    max_absolute = float(np.max(absolute))
    scale = max(
        float(np.max(np.abs(reference))),
        float(np.max(np.abs(candidate))),
        np.finfo(float).tiny,
    )
    return {
        "max_absolute": max_absolute,
        "max_relative": max_absolute / scale,
    }


def _run_cpu(request: PRRunRequest) -> tuple[Any, float]:
    cpu_request = replace(
        request,
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
    )
    started = perf_counter()
    result = run_pr_timedependent(cpu_request)
    elapsed = perf_counter() - started
    if result.diagnostics["backend"]["backend"] != "numpy":
        raise AssertionError("NumPy request did not report the numpy backend")
    return result, elapsed


def _initialize_cupy() -> tuple[Any, dict[str, Any]]:
    started = perf_counter()
    import cupy as cp  # imported here so cold initialization is measurable

    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise RuntimeError("CuPy reports no visible CUDA devices")
    device = cp.cuda.Device(0)
    device.use()
    allocation = cp.arange(16, dtype=cp.float64)
    allocation.sum()
    cp.cuda.get_current_stream().synchronize()
    cold_seconds = perf_counter() - started

    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return cp, {
        "device_count": device_count,
        "device_id": int(device.id),
        "device_name": str(name),
        "compute_capability": (
            f"{int(properties['major'])}.{int(properties['minor'])}"
        ),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": str(cp.__version__),
        "cold_initialization_seconds": cold_seconds,
    }


def _run_gpu(request: PRRunRequest, cp: Any) -> tuple[Any, float]:
    gpu_request = replace(
        request,
        backend=BackendSpec(
            backend="cupy",
            precision="float64",
            verbose=False,
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    started = perf_counter()
    result = run_pr_timedependent(gpu_request)
    cp.cuda.get_current_stream().synchronize()
    elapsed = perf_counter() - started
    backend = result.diagnostics["backend"]
    if backend["backend"] != "cupy" or not backend["is_gpu"]:
        raise AssertionError(f"CuPy request reported unexpected backend: {backend!r}")
    return result, elapsed


def _base_record(mode: str) -> dict[str, Any]:
    repository = _repository_root()
    return {
        "schema": "lcprop.pr_gpu_smoke.v1",
        "mode": mode,
        "status": "failed",
        "passed": False,
        "exit_code": 1,
        "started_utc": _utc_now(),
        "ended_utc": None,
        "git_sha": _git_sha(repository),
        "repository": str(repository),
        "hostname": socket.gethostname(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "lcprop": importlib.metadata.version("lcprop"),
        },
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_gpus": os.environ.get("SLURM_GPUS"),
            "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "request": {
            "shape": [EXPECTED_NX, EXPECTED_NY],
            "Nz": 4,
            "material_steps": 2,
            "integrator": PR_SEMI_IMPLICIT_INTEGRATOR,
            "precision": "float64",
        },
        "tolerances": {
            "optical_rtol": OPTICAL_RTOL,
            "optical_atol": OPTICAL_ATOL,
            "pr_rtol": PR_RTOL,
            "pr_atol": PR_ATOL,
            "power_relative_drift_limit": POWER_DRIFT_LIMIT,
            "relative_difference_definition": (
                "max_absolute_difference divided by the larger field maximum"
            ),
        },
    }


def _validate_complete_gpu_record(record: dict[str, Any]) -> None:
    required_paths = (
        ("git_sha",),
        ("hostname",),
        ("versions", "python"),
        ("versions", "numpy"),
        ("versions", "scipy"),
        ("versions", "lcprop"),
        ("versions", "cupy"),
        ("scheduler", "slurm_job_id"),
        ("gpu", "device_name"),
        ("gpu", "cuda_driver_version"),
        ("gpu", "cuda_runtime_version"),
        ("timing_seconds", "cupy_cold_initialization"),
        ("timing_seconds", "cpu"),
        ("timing_seconds", "gpu_warmed"),
        ("backend", "gpu_reported"),
        ("differences", "optical", "max_absolute"),
        ("differences", "optical", "max_relative"),
        ("differences", "pr", "max_absolute"),
        ("differences", "pr", "max_relative"),
        ("power", "cpu_relative_drift"),
        ("power", "gpu_relative_drift"),
    )
    for path in required_paths:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise AssertionError("incomplete provenance: " + ".".join(path))
            value = value[key]
        if value is None or value == "":
            raise AssertionError("empty provenance: " + ".".join(path))


def run_check(*, cpu_only: bool) -> dict[str, Any]:
    mode = "cpu_only_validation" if cpu_only else "gpu_commissioning"
    record = _base_record(mode)
    request = build_request()
    try:
        cpu_result, cpu_seconds = _run_cpu(request)
        cpu_drift = _power_drift(cpu_result)
        if cpu_drift > POWER_DRIFT_LIMIT:
            raise AssertionError(
                f"CPU power drift {cpu_drift:.3e} exceeds {POWER_DRIFT_LIMIT:.3e}"
            )
        record.update(
            {
                "backend": {
                    "cpu_requested": "numpy",
                    "cpu_reported": cpu_result.diagnostics["backend"]["backend"],
                },
                "timing_seconds": {"cpu": cpu_seconds},
                "power": {"cpu_relative_drift": cpu_drift},
            }
        )

        if cpu_only:
            record.update(
                {
                    "status": "cpu_validation_passed",
                    "passed": True,
                    "exit_code": 0,
                }
            )
            return record

        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU commissioning requires a Slurm job allocation")
        allocation_markers = (
            os.environ.get("SLURM_JOB_GPUS"),
            os.environ.get("SLURM_GPUS"),
            os.environ.get("SLURM_GPUS_ON_NODE"),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if not any(value not in (None, "", "NoDevFiles", "-1") for value in allocation_markers):
            raise RuntimeError("Slurm allocation does not report an assigned GPU")

        cp, gpu_info = _initialize_cupy()
        record["gpu"] = gpu_info
        record["versions"]["cupy"] = gpu_info["cupy_version"]
        record["timing_seconds"]["cupy_cold_initialization"] = gpu_info[
            "cold_initialization_seconds"
        ]

        gpu_result, gpu_seconds = _run_gpu(request, cp)
        gpu_drift = _power_drift(gpu_result)
        optical_difference = _difference_metrics(
            cpu_result.A_final,
            gpu_result.A_final,
        )
        pr_difference = _difference_metrics(
            cpu_result.E_final,
            gpu_result.E_final,
        )
        record["backend"].update(
            {
                "gpu_requested": "cupy",
                "gpu_reported": gpu_result.diagnostics["backend"]["backend"],
                "gpu_is_gpu": gpu_result.diagnostics["backend"]["is_gpu"],
            }
        )
        record["timing_seconds"]["gpu_warmed"] = gpu_seconds
        record["differences"] = {
            "optical": optical_difference,
            "pr": pr_difference,
        }
        record["power"]["gpu_relative_drift"] = gpu_drift

        np.testing.assert_allclose(
            gpu_result.A_final,
            cpu_result.A_final,
            rtol=OPTICAL_RTOL,
            atol=OPTICAL_ATOL,
        )
        np.testing.assert_allclose(
            gpu_result.E_final,
            cpu_result.E_final,
            rtol=PR_RTOL,
            atol=PR_ATOL,
        )
        if gpu_drift > POWER_DRIFT_LIMIT:
            raise AssertionError(
                f"GPU power drift {gpu_drift:.3e} exceeds {POWER_DRIFT_LIMIT:.3e}"
            )

        _validate_complete_gpu_record(record)
        record.update(
            {
                "status": "passed",
                "passed": True,
                "exit_code": 0,
            }
        )
        return record
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return record
    finally:
        record["ended_utc"] = _utc_now()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="destination for the complete machine-readable commissioning record",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="validate only the local NumPy path; never marks GPU commissioning complete",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = run_check(cpu_only=bool(args.cpu_only))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
