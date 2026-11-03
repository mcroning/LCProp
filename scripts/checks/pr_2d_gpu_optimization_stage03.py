#!/usr/bin/env python3
"""Profile Stage-03 canonical PR optical propagation on a CUDA backend."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

import numpy as np

import lcprop.optics.splitstep as splitstep_module
import lcprop.pr.workflow as pr_workflow_module
import lcprop.pr.transverse.static_workflow as workflow_module
import lcprop.pr.transverse.workflow as transverse_workflow_module
from lcprop.pr.transverse.products import pr_transverse_static_result_to_run_data
from lcprop.pr.transverse.static_workflow import run_pr_transverse_static
from pr_2d_gpu_optimization_stage01 import make_request


SCHEMA = "lcprop.pr_2d_gpu_optimization_stage03.v1"


def _git_sha() -> str:
    override = os.environ.get("LCPROP_PROFILE_SOURCE_SHA")
    if override:
        return override
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _device_name(cp) -> str:
    raw = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _synchronize(cp) -> None:
    cp.cuda.Stream.null.synchronize()


def _save_reference(result, output_dir: Path, grid_size: int) -> dict:
    indices = [0, result.psi_final.shape[0] // 2, result.psi_final.shape[0] - 1]
    values = {
        "A_final": np.asarray(result.A_final),
        "psi_selected": np.asarray(result.psi_final)[indices],
        "source_selected": np.asarray(result.source_intensity_stack)[indices],
        "equilibrium_selected": np.asarray(result.equilibrium_residual_stack)[indices],
    }
    path = output_dir / f"reference_{grid_size}.npz"
    np.savez_compressed(path, **values)
    return {
        "path": path.name,
        **{f"{name}_sha256": _sha256_array(value) for name, value in values.items()},
    }


def _clean_case(cp, grid_size: int, output_dir: Path) -> dict:
    request = make_request(grid_size)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    _synchronize(cp)
    free_before, total_bytes = cp.cuda.runtime.memGetInfo()
    started = perf_counter()
    result = run_pr_transverse_static(request)
    _synchronize(cp)
    total_seconds = perf_counter() - started
    free_after, _ = cp.cuda.runtime.memGetInfo()
    product_started = perf_counter()
    run_data = pr_transverse_static_result_to_run_data(result)
    product_seconds = perf_counter() - product_started
    report = {
        "grid_size": grid_size,
        "status": result.status,
        "converged": bool(result.converged),
        "termination_reason": result.diagnostics["termination_reason"],
        "completed_coupled_iterations": result.completed_coupled_iterations,
        "outer_attempted": len(result.iteration_records),
        "outer_backtracks": sum(item.backtracks for item in result.iteration_records),
        "material_newton_iterations": sum(
            item.material_newton_iterations for item in result.iteration_records
        ),
        "material_pcg_iterations": sum(
            item.material_pcg_iterations for item in result.iteration_records
        ),
        "timing": {
            "synchronized_total_seconds": total_seconds,
            "workflow_total_seconds": result.timing["total_seconds"],
            "material_solve_seconds": result.timing["material_solve_seconds"],
            "optical_pass_seconds": result.timing["optical_pass_seconds"],
            "product_conversion_seconds": product_seconds,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "device_used_before_bytes": int(total_bytes - free_before),
            "device_used_after_bytes": int(total_bytes - free_after),
            "pool_used_after_bytes": int(pool.used_bytes()),
            "pool_reserved_after_bytes": int(pool.total_bytes()),
            "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "diagnostics": {
            key: result.diagnostics[key]
            for key in (
                "equilibrium_residual_rms",
                "equilibrium_residual_max",
                "td_rhs_residual_rms",
                "td_rhs_residual_max",
                "optical_power_relative_drift",
                "curl_rms",
                "curl_max",
                "gauss_rms",
                "gauss_max",
            )
        },
        "replay": result.replay_diagnostics,
        "products": {
            "field_count": len(run_data.fields),
            "curve_count": len(run_data.curves),
            "diagnostic_count": len(run_data.diagnostics),
        },
        "reference": _save_reference(result, output_dir, grid_size),
    }
    del result, run_data
    gc.collect()
    pool.free_all_blocks()
    _synchronize(cp)
    return report


@contextlib.contextmanager
def _detailed_instrumentation(cp):
    counters = {
        "optical_pass": 0,
        "optical_slices": 0,
        "optical_substeps": 0,
        "fft2": 0,
        "ifft2": 0,
        "phase_stack_build": 0,
        "phase_slice_build": 0,
        "midpoint_advance": 0,
        "prepared_advance": 0,
        "response_build": 0,
        "response_apply": 0,
        "linear_hop": 0,
        "intensity_build": 0,
        "scattering_apply": 0,
    }
    seconds = {key: 0.0 for key in counters}
    optical_depth = 0
    originals = {
        "optical_pass": workflow_module._optical_pass,
        "phase_stack": workflow_module._canonical_scattering_phase_stack,
        "phase_slice": pr_workflow_module._canonical_scattering_phase_for_slice,
        "midpoint": transverse_workflow_module.advance_pr_slice_with_midpoint_source,
        "scatter_apply": transverse_workflow_module._apply_canonical_scattering_after_slice,
        "prepared": pr_workflow_module.advance_prepared_response,
        "response_build": pr_workflow_module.half_step_response_from_E,
        "intensity": pr_workflow_module.pr_driving_intensity,
        "response_apply": splitstep_module.apply_response_screen_inplace,
        "linear_hop": splitstep_module.hop_linear_inplace,
        "fft2": cp.fft.fft2,
        "ifft2": cp.fft.ifft2,
    }

    def timed(name, function, *, optical_only=False):
        def wrapper(*args, **kwargs):
            if optical_only and optical_depth == 0:
                return function(*args, **kwargs)
            _synchronize(cp)
            started = perf_counter()
            result = function(*args, **kwargs)
            _synchronize(cp)
            counters[name] += 1
            seconds[name] += perf_counter() - started
            return result

        return wrapper

    def optical_pass(*args, **kwargs):
        nonlocal optical_depth
        _synchronize(cp)
        started = perf_counter()
        optical_depth += 1
        try:
            result = originals["optical_pass"](*args, **kwargs)
        finally:
            optical_depth -= 1
        _synchronize(cp)
        counters["optical_pass"] += 1
        seconds["optical_pass"] += perf_counter() - started
        request = kwargs["request"]
        grid = kwargs["grid"]
        counters["optical_slices"] += int(grid.Nz)
        counters["optical_substeps"] += int(grid.Nz) * int(
            request.solver.optical_substeps
        )
        return result

    workflow_module._optical_pass = optical_pass
    workflow_module._canonical_scattering_phase_stack = timed(
        "phase_stack_build", originals["phase_stack"]
    )
    pr_workflow_module._canonical_scattering_phase_for_slice = timed(
        "phase_slice_build", originals["phase_slice"]
    )
    transverse_workflow_module.advance_pr_slice_with_midpoint_source = timed(
        "midpoint_advance", originals["midpoint"], optical_only=True
    )
    transverse_workflow_module._apply_canonical_scattering_after_slice = timed(
        "scattering_apply", originals["scatter_apply"], optical_only=True
    )
    pr_workflow_module.advance_prepared_response = timed(
        "prepared_advance", originals["prepared"], optical_only=True
    )
    pr_workflow_module.half_step_response_from_E = timed(
        "response_build", originals["response_build"], optical_only=True
    )
    pr_workflow_module.pr_driving_intensity = timed(
        "intensity_build", originals["intensity"], optical_only=True
    )
    splitstep_module.apply_response_screen_inplace = timed(
        "response_apply", originals["response_apply"], optical_only=True
    )
    splitstep_module.hop_linear_inplace = timed(
        "linear_hop", originals["linear_hop"], optical_only=True
    )
    cp.fft.fft2 = timed("fft2", originals["fft2"], optical_only=True)
    cp.fft.ifft2 = timed("ifft2", originals["ifft2"], optical_only=True)
    try:
        yield counters, seconds
    finally:
        workflow_module._optical_pass = originals["optical_pass"]
        workflow_module._canonical_scattering_phase_stack = originals["phase_stack"]
        pr_workflow_module._canonical_scattering_phase_for_slice = originals["phase_slice"]
        transverse_workflow_module.advance_pr_slice_with_midpoint_source = originals[
            "midpoint"
        ]
        transverse_workflow_module._apply_canonical_scattering_after_slice = originals[
            "scatter_apply"
        ]
        pr_workflow_module.advance_prepared_response = originals["prepared"]
        pr_workflow_module.half_step_response_from_E = originals["response_build"]
        pr_workflow_module.pr_driving_intensity = originals["intensity"]
        splitstep_module.apply_response_screen_inplace = originals["response_apply"]
        splitstep_module.hop_linear_inplace = originals["linear_hop"]
        cp.fft.fft2 = originals["fft2"]
        cp.fft.ifft2 = originals["ifft2"]


def _detailed_case(cp, grid_size: int, output_dir: Path) -> dict:
    request = make_request(grid_size)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    _synchronize(cp)
    with _detailed_instrumentation(cp) as (counters, seconds):
        started = perf_counter()
        result = run_pr_transverse_static(request)
        _synchronize(cp)
        total_seconds = perf_counter() - started
    report = {
        "grid_size": grid_size,
        "status": result.status,
        "converged": bool(result.converged),
        "synchronized_total_seconds": total_seconds,
        "counters": counters,
        "inclusive_seconds": seconds,
        "memory": {
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
            "scattering_phase_cache_bytes": int(
                round(request.grid.z_length_um / request.grid.dz_um)
                * request.grid.Nx
                * request.grid.Ny
                * np.dtype(np.float32).itemsize
            ),
        },
        "reference": _save_reference(result, output_dir, grid_size),
    }
    del result
    gc.collect()
    pool.free_all_blocks()
    _synchronize(cp)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("clean", "detailed"), required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=(256, 512, 1024))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cupy as cp

    warm_started = perf_counter()
    cp.fft.ifft2(cp.fft.fft2(cp.zeros((16, 16), dtype=cp.float32))).real.sum()
    _synchronize(cp)
    report = {
        "schema": SCHEMA,
        "git_sha": _git_sha(),
        "mode": args.mode,
        "device": {
            "name": _device_name(cp),
            "id": int(cp.cuda.Device().id),
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "warmup_seconds": perf_counter() - warm_started,
        },
        "cases": [],
    }
    output = args.output_dir / f"{args.mode}.json"
    try:
        for size in args.sizes:
            case = (
                _clean_case(cp, size, args.output_dir)
                if args.mode == "clean"
                else _detailed_case(cp, size, args.output_dir)
            )
            report["cases"].append(case)
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
