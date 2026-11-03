#!/usr/bin/env python3
"""Bounded H200 timing and memory check for full-transverse PR Profile v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    run_pr_transverse_timedependent,
)


def _request(*, grid_size: int, steps: int) -> PRTransverseRunRequest:
    return PRTransverseRunRequest(
        grid=GridSpec(
            Nx=grid_size,
            Ny=grid_size,
            x_aperture_um=1000.0,
            y_aperture_um=1000.0,
            dz_um=50.0,
            z_length_um=4000.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    name="frozen first-fanning beam",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=200.0,
                    waist_y_um=200.0,
                    x0_um=83.2665844712203,
                    y0_um=0.0,
                    tilt_x_rad_per_um=-0.9909508003805507,
                    tilt_y_rad_per_um=0.0,
                    coherence_group="pr-fanning",
                ),
            )
        ),
        material=PRMaterialSpec(
            dark_intensity=0.01,
            uniform_background_intensity=0.0,
            applied_field=0.0,
            gain_length_product=10.0,
            refractive_index=2.4,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=0.02,
            optical_substeps=1,
        ),
        backend=BackendSpec("cupy", "float32", False),
        scattering=PRCanonicalScatteringSpec(
            epsilon=0.02,
            transverse_correlation_um=0.4,
            realization_seed=2137319267,
            canonical_dz_um=2.0,
            algorithm_version=PR_CANONICAL_SCATTERING_V2,
        ),
    )


def _selected_diagnostics(result) -> dict[str, object]:
    keys = (
        "carrier_relative_drift_max",
        "curl_rms",
        "curl_max",
        "gauss_rms",
        "gauss_max",
        "optical_power_relative_drift",
        "finite_material_state",
        "finite_optical_state",
    )
    return {key: result.diagnostics[key] for key in keys}


def _timed_run(cp, request):
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    started = perf_counter()
    result = run_pr_transverse_timedependent(request)
    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - started
    free_after, _ = cp.cuda.runtime.memGetInfo()
    return result, {
        "elapsed_seconds": elapsed,
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
        "device_total_bytes": int(total_memory),
        "device_free_before_bytes": int(free_before),
        "device_free_after_bytes": int(free_after),
        "device_free_delta_bytes": int(free_before - free_after),
    }


def _case(cp, grid_size: int) -> dict[str, object]:
    step_request = _request(grid_size=grid_size, steps=1)
    step_result, step_timing = _timed_run(cp, step_request)
    replay_result, replay_timing = _timed_run(
        cp, _request(grid_size=grid_size, steps=0)
    )
    marginal = step_timing["elapsed_seconds"] - replay_timing["elapsed_seconds"]
    tau_steps = int(round(10.0 / step_request.solver.dt_normalized))
    dx = step_request.grid.x_aperture_um / grid_size
    return {
        "grid_size": grid_size,
        "dx_um": dx,
        "k_nyquist_rad_per_um": math.pi / dx,
        "carrier_rad_per_um": step_request.beams.channels[0].tilt_x_rad_per_um,
        "carrier_resolved": abs(step_request.beams.channels[0].tilt_x_rad_per_um)
        < math.pi / dx,
        "first_broad_ring_rad_per_um": 2.63,
        "first_broad_ring_resolved": 2.63 < math.pi / dx,
        "request": asdict(step_request),
        "step": {
            **step_timing,
            "backend": step_result.backend_summary,
            "diagnostics": _selected_diagnostics(step_result),
        },
        "replay": {
            **replay_timing,
            "backend": replay_result.backend_summary,
            "diagnostics": _selected_diagnostics(replay_result),
        },
        "marginal_material_step_seconds": marginal,
        "tau_10_steps": tau_steps,
        "projected_tau_10_seconds": (
            replay_timing["elapsed_seconds"] + tau_steps * marginal
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=(256, 1024))
    args = parser.parse_args()

    import cupy as cp

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    payload = {
        "schema": "lcprop.pr_transverse_gpu_commissioning.v1",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_id": int(device.id),
        "device_name": str(device_name),
        "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cases": [_case(cp, grid_size) for grid_size in args.grids],
    }
    payload["host_max_rss_kib"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
