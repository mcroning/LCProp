#!/usr/bin/env python3
"""Run the bounded coupled-static PR image-amplification pilot.

Full mode requires a scheduler-assigned GPU and preserves the physical,
optical, image-processing, reconstruction, and metric definitions used by the
transient GPU pilot. ``--cpu-only`` substitutes the transient pilot's small
preflight geometry and exists only for local pipeline validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
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
from lcprop.core.grid import make_grid
from lcprop.pr.coupling import analytic_plane_wave_gain_length
from lcprop.pr.image_amplification import (
    PRImageAmplificationSpec,
    _intensity_metrics,
    _linear_propagate,
    isolate_signal_carrier,
    make_image_amplification_request,
    paper_absolute_signal_gain,
    signal_carrier_mask,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)


IMAGE_SHA256 = "e424f1070587d79d2a91f4a3bd14e7c11e80adfe7c723e45432d9d4fc5bb85b7"
TD_REFERENCE_SHA256 = (
    "1f686ec06875b66d19b20c87b8108684b971ef9afbd35dde963c91979e3eb9b9"
)
POWER_DRIFT_LIMIT = 1.0e-9
MINIMUM_GAIN = 1.1
MINIMUM_CORRELATION = 0.8
MAXIMUM_NORMALIZED_RMSE = 1.0
ZERO_RESPONSE_CORRELATION_ATOL = 2.0e-12


@dataclass(frozen=True)
class StaticImageAmplificationOutcome:
    """Static workflow result and shared image-amplification diagnostics."""

    request: PRStaticRunRequest
    run_result: Any
    input_signal_field: np.ndarray
    output_signal_field: np.ndarray
    backpropagated_signal_field: np.ndarray
    zero_response_backpropagated_signal_field: np.ndarray
    analytic_absolute_signal_gain: float
    measured_absolute_signal_gain: float
    image_intensity_correlation: float
    zero_response_image_intensity_correlation: float
    normalized_image_rmse: float
    normalized_power_relative_drift: float
    workflow_runtime_s: float
    reconstruction_optical_runtime_s: float
    reconstruction_runtime_s: float
    runtime_s: float


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
    """Return the exact physical and optical parameters of TD job 2209653."""

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
    """Return the geometrically similar preflight used by the TD pilot."""

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


def static_solver_options() -> PRStaticWorkflowOptions:
    """Return the commissioned precision-aware coupled-static controls."""

    return PRStaticWorkflowOptions(
        max_coupled_passes=20,
        max_backtracks=16,
        minimum_step_scale=2.0**-16,
        armijo_fraction=1.0e-4,
        optical_substeps=1,
        record_iteration_history=True,
    )


def _read_checksummed_file(path: Path, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"received {digest}"
        )
    return payload


def _read_image(path: Path) -> tuple[np.ndarray, str]:
    payload = _read_checksummed_file(path, IMAGE_SHA256)
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return image, hashlib.sha256(payload).hexdigest()


def _read_td_reference(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_checksummed_file(path, TD_REFERENCE_SHA256)
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


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
    cp.arange(32, dtype=cp.float64).sum()
    cp.cuda.get_current_stream().synchronize()
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
        "cold_initialization_seconds": perf_counter() - started,
        "free_memory_before_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def _base_record(
    mode: str,
    image_path: Path,
    td_reference_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "lcprop.pr_static_image_amplification_gpu_pilot.v1",
        "mode": mode,
        "status": "failed",
        "operational_status": "failed",
        "scientific_assessment": {
            "classification": "not_evaluated",
            "controls": {},
        },
        "passed": False,
        "exit_code": 1,
        "started_utc": _utc_now(),
        "ended_utc": None,
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "repository": str(_repository_root()),
        "image_path": str(image_path),
        "td_reference_path": str(td_reference_path),
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
            "zero_response_correlation_atol": ZERO_RESPONSE_CORRELATION_ATOL,
        },
    }


def _make_static_request(
    image: np.ndarray,
    spec: PRImageAmplificationSpec,
    backend: BackendSpec,
) -> tuple[PRStaticRunRequest, float, float]:
    transient_request, _transmission, normalized_grating, gain_length = (
        make_image_amplification_request(image, spec)
    )
    return (
        PRStaticRunRequest(
            grid=transient_request.grid,
            beams=transient_request.beams,
            material=transient_request.material,
            solver=static_solver_options(),
            backend=backend,
            initial_A=transient_request.initial_A,
        ),
        normalized_grating,
        gain_length,
    )


def run_static_image_amplification(
    image: np.ndarray,
    spec: PRImageAmplificationSpec,
    *,
    backend: BackendSpec,
) -> StaticImageAmplificationOutcome:
    """Run the static solver with the established image reconstruction."""

    request, normalized_grating, gain_length = _make_static_request(
        image,
        spec,
        backend,
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = float(request.beams.channels[0].tilt_x_rad_per_um)
    signal_kx = float(request.beams.channels[1].tilt_x_rad_per_um)
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    coherent_input = np.sum(np.asarray(request.initial_A), axis=0)
    input_signal = isolate_signal_carrier(coherent_input, mask)

    benchmark_started = perf_counter()
    workflow_started = perf_counter()
    run_result = run_pr_static(request)
    workflow_runtime = perf_counter() - workflow_started

    reconstruction_started = perf_counter()
    coherent_output = np.sum(np.asarray(run_result.A_final), axis=0)
    output_signal = isolate_signal_carrier(coherent_output, mask)
    reconstruction_optical_runtime = 0.0

    optical_started = perf_counter()
    backpropagated = _linear_propagate(
        output_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started

    optical_started = perf_counter()
    zero_output = _linear_propagate(
        coherent_input,
        grid,
        distance_um=float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started
    zero_signal = isolate_signal_carrier(zero_output, mask)
    optical_started = perf_counter()
    zero_backpropagated = _linear_propagate(
        zero_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started

    signal_intensity = np.abs(np.asarray(request.initial_A)[1]) ** 2
    roi = signal_intensity > 1.0e-4 * float(np.max(signal_intensity))
    correlation, normalized_rmse = _intensity_metrics(
        input_signal,
        backpropagated,
        roi,
    )
    zero_correlation, _ = _intensity_metrics(
        input_signal,
        zero_backpropagated,
        roi,
    )

    dxdy = float(grid.dx_um) * float(grid.dy_um)
    input_signal_power = float(np.sum(np.abs(input_signal) ** 2) * dxdy)
    output_signal_power = float(np.sum(np.abs(output_signal) ** 2) * dxdy)
    internal_angle = math.asin(
        pump_kx
        * float(spec.wavelength_um)
        / (2.0 * math.pi * float(spec.refractive_index))
    )
    analytic_gamma = analytic_plane_wave_gain_length(
        gain_length_product=gain_length,
        signed_grating_k_normalized=normalized_grating,
        internal_half_angle_rad=internal_angle,
    )
    analytic_gain = paper_absolute_signal_gain(
        input_ratio=float(spec.input_peak_ratio),
        gamma_p_L=analytic_gamma,
    )
    power_drift = float(
        (run_result.power_final - run_result.power_initial)
        / run_result.power_initial
    )
    reconstruction_runtime = perf_counter() - reconstruction_started
    return StaticImageAmplificationOutcome(
        request=request,
        run_result=run_result,
        input_signal_field=input_signal,
        output_signal_field=output_signal,
        backpropagated_signal_field=backpropagated,
        zero_response_backpropagated_signal_field=zero_backpropagated,
        analytic_absolute_signal_gain=analytic_gain,
        measured_absolute_signal_gain=output_signal_power / input_signal_power,
        image_intensity_correlation=correlation,
        zero_response_image_intensity_correlation=zero_correlation,
        normalized_image_rmse=normalized_rmse,
        normalized_power_relative_drift=power_drift,
        workflow_runtime_s=workflow_runtime,
        reconstruction_optical_runtime_s=reconstruction_optical_runtime,
        reconstruction_runtime_s=reconstruction_runtime,
        runtime_s=perf_counter() - benchmark_started,
    )


def _validate_operational_outcome(
    outcome: StaticImageAmplificationOutcome,
    *,
    expected_backend: str,
) -> None:
    result = outcome.run_result
    backend = result.backend_summary
    if backend["backend"] != expected_backend:
        raise AssertionError(
            f"requested {expected_backend}, workflow reported {backend!r}"
        )
    if expected_backend == "cupy" and not backend["is_gpu"]:
        raise AssertionError("CuPy workflow did not report is_gpu=True")
    if not result.converged or result.status != "converged":
        raise AssertionError("coupled-static workflow did not converge")
    if not all(summary.converged for summary in result.slice_summaries):
        raise AssertionError("one or more coupled-static slices did not converge")
    replay = result.replay_diagnostics
    for key in (
        "field_consistent",
        "source_consistent",
        "residual_consistent",
        "residual_converged",
    ):
        if not replay[key]:
            raise AssertionError(f"static replay check {key!r} failed")
    arrays = (
        result.A_final,
        result.E_final,
        result.source_intensity_stack,
        result.residual_stack,
        outcome.output_signal_field,
        outcome.backpropagated_signal_field,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise AssertionError("pilot produced nonfinite fields")
    metrics = (
        outcome.analytic_absolute_signal_gain,
        outcome.measured_absolute_signal_gain,
        outcome.image_intensity_correlation,
        outcome.zero_response_image_intensity_correlation,
        outcome.normalized_image_rmse,
        outcome.normalized_power_relative_drift,
    )
    if not all(math.isfinite(value) for value in metrics):
        raise AssertionError("pilot produced nonfinite scientific metrics")
    if abs(outcome.normalized_power_relative_drift) > POWER_DRIFT_LIMIT:
        raise AssertionError("pilot optical power drift exceeds its limit")
    if (
        abs(outcome.zero_response_image_intensity_correlation - 1.0)
        > ZERO_RESPONSE_CORRELATION_ATOL
    ):
        raise AssertionError("zero-response reconstruction control failed")


def _scientific_assessment(
    outcome: StaticImageAmplificationOutcome,
) -> dict[str, Any]:
    controls = {
        "signal_amplification": {
            "measured": outcome.measured_absolute_signal_gain,
            "criterion": f"> {MINIMUM_GAIN}",
            "supported": outcome.measured_absolute_signal_gain > MINIMUM_GAIN,
        },
        "image_correlation": {
            "measured": outcome.image_intensity_correlation,
            "criterion": f">= {MINIMUM_CORRELATION}",
            "supported": outcome.image_intensity_correlation
            >= MINIMUM_CORRELATION,
        },
        "normalized_image_rmse": {
            "measured": outcome.normalized_image_rmse,
            "criterion": f"<= {MAXIMUM_NORMALIZED_RMSE}",
            "supported": outcome.normalized_image_rmse
            <= MAXIMUM_NORMALIZED_RMSE,
        },
    }
    classification = (
        "supported"
        if all(control["supported"] for control in controls.values())
        else "not_supported"
    )
    return {
        "classification": classification,
        "controls": controls,
        "transient_comparison_is_acceptance_criterion": False,
    }


def _static_diagnostics(result: Any) -> dict[str, Any]:
    accepted = sum(record.accepted for record in result.iteration_records)
    rejected = len(result.iteration_records) - accepted
    return {
        "status": result.status,
        "converged": result.converged,
        "completed_slices": result.completed_slices,
        "coupled_iteration_records": len(result.iteration_records),
        "accepted_coupled_updates": accepted,
        "rejected_coupled_updates": rejected,
        "maximum_coupled_passes_used": max(
            summary.coupled_passes for summary in result.slice_summaries
        ),
        "maximum_slice_residual_rms": max(
            summary.final_residual_rms for summary in result.slice_summaries
        ),
        "maximum_slice_residual_max": max(
            summary.final_residual_max for summary in result.slice_summaries
        ),
        "termination_reasons": sorted(
            {summary.termination_reason for summary in result.slice_summaries}
        ),
        "replay": result.replay_diagnostics,
        "tolerance_provenance": result.tolerance_provenance,
    }


def _write_products(
    outcome: StaticImageAmplificationOutcome,
    image: np.ndarray,
    output_dir: Path,
) -> dict[str, int]:
    input_intensity = np.abs(outcome.input_signal_field) ** 2
    reconstructed_intensity = np.abs(outcome.backpropagated_signal_field) ** 2
    zero_intensity = (
        np.abs(outcome.zero_response_backpropagated_signal_field) ** 2
    )
    output_signal_intensity = np.abs(outcome.output_signal_field) ** 2
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
    midpoint = outcome.run_result.E_final.shape[0] // 2
    np.save(
        output_dir / "E_final_midplane.npy",
        np.asarray(outcome.run_result.E_final[midpoint], dtype=np.float32),
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


def _metric_record(outcome: StaticImageAmplificationOutcome) -> dict[str, Any]:
    return {
        "analytic_absolute_signal_gain": outcome.analytic_absolute_signal_gain,
        "measured_absolute_signal_gain": outcome.measured_absolute_signal_gain,
        "image_intensity_correlation": outcome.image_intensity_correlation,
        "zero_response_image_intensity_correlation": (
            outcome.zero_response_image_intensity_correlation
        ),
        "normalized_image_rmse": outcome.normalized_image_rmse,
        "normalized_power_relative_drift": (
            outcome.normalized_power_relative_drift
        ),
        "finite_outputs": True,
    }


def _td_comparison(
    static_metrics: dict[str, Any],
    td_metrics: dict[str, Any],
) -> dict[str, Any]:
    compared_names = (
        "measured_absolute_signal_gain",
        "image_intensity_correlation",
        "zero_response_image_intensity_correlation",
        "normalized_image_rmse",
        "normalized_power_relative_drift",
    )
    return {
        name: {
            "static": float(static_metrics[name]),
            "transient_job_2209653": float(td_metrics[name]),
            "static_minus_transient": float(
                static_metrics[name] - td_metrics[name]
            ),
        }
        for name in compared_names
    }


def run_pilot(
    *,
    image_path: Path,
    td_reference_path: Path,
    output_dir: Path,
    cpu_only: bool,
) -> dict[str, Any]:
    mode = "cpu_only_validation" if cpu_only else "gpu_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _base_record(mode, image_path, td_reference_path, output_dir)
    total_started = perf_counter()
    try:
        image, image_digest = _read_image(image_path)
        td_reference, td_digest = _read_td_reference(td_reference_path)
        spec = cpu_validation_spec() if cpu_only else pilot_spec()
        record["image_sha256"] = image_digest
        record["image_shape"] = list(image.shape)
        record["td_reference"] = {
            "sha256": td_digest,
            "git_sha": td_reference["git_sha"],
            "slurm_job_id": td_reference["scheduler"]["slurm_job_id"],
            "metrics": td_reference["metrics"],
            "directly_comparable_geometry": not cpu_only,
        }
        record["parameters"] = asdict(spec)
        record["static_solver"] = asdict(static_solver_options())
        record["transient_controls_not_used_by_static_solver"] = {
            "Nt": spec.Nt,
            "dt_normalized": spec.dt_normalized,
        }
        Nz = int(round(spec.interaction_length_um / spec.dz_um))
        record["sampling"] = {
            "Nz": Nz,
            "dx_um": spec.x_aperture_um / spec.Nx,
            "dy_um": spec.y_aperture_um / spec.Ny,
            "samples_per_waist_x": (
                spec.beam_waist_um * spec.Nx / spec.x_aperture_um
            ),
            "samples_per_waist_y": (
                spec.beam_waist_um * spec.Ny / spec.y_aperture_um
            ),
            "grating_samples_per_period": (
                spec.Nx / (2.0 * spec.positive_mode_index)
            ),
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
        outcome = run_static_image_amplification(
            image,
            spec,
            backend=backend,
        )
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
        synchronized_execution = perf_counter() - execution_started
        expected_backend = "numpy" if cpu_only else "cupy"

        record["backend"] = {
            "requested": expected_backend,
            "reported": outcome.run_result.backend_summary["backend"],
            "is_gpu": outcome.run_result.backend_summary["is_gpu"],
        }
        record["metrics"] = _metric_record(outcome)
        record["static_diagnostics"] = _static_diagnostics(outcome.run_result)
        if not cpu_only:
            record["comparison_with_transient_job_2209653"] = _td_comparison(
                record["metrics"],
                td_reference["metrics"],
            )
        record["timing_seconds"] = {
            "synchronized_execution": synchronized_execution,
            "coupled_static_workflow": outcome.workflow_runtime_s,
            "reconstruction_optical_propagation": (
                outcome.reconstruction_optical_runtime_s
            ),
            "reconstruction_total": outcome.reconstruction_runtime_s,
            "benchmark_total": outcome.runtime_s,
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
        _validate_operational_outcome(
            outcome,
            expected_backend=expected_backend,
        )
        record["scientific_assessment"] = _scientific_assessment(outcome)
        record["products"] = _write_products(outcome, image, output_dir)
        record.update(
            {
                "status": "passed",
                "operational_status": "passed",
                "passed": True,
                "exit_code": 0,
            }
        )
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
    parser.add_argument("--td-reference-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    record = run_pilot(
        image_path=args.image,
        td_reference_path=args.td_reference_metrics,
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
