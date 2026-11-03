#!/usr/bin/env python3
"""Bounded GPU parity, timing, and memory check for the PR cyclic solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lcprop.pr.cyclic import solve_cyclic_tridiagonal_rows
from lcprop.pr.cyclic_gpu import solve_cyclic_tridiagonal_rows_gpu
from lcprop.pr import evolution


def _case(cp, *, name, shape, dtype, repeats):
    rng = np.random.default_rng(20260817 + shape[-2])
    intensity = (0.01 + 2.0 * rng.random(shape)).astype(dtype)
    q = (0.12 * intensity / 0.31**2).astype(dtype)
    lower = -q
    upper = -q
    diagonal = 1.0 + 2.0 * q
    rhs = rng.normal(scale=0.1, size=shape).astype(dtype)
    expected = solve_cyclic_tridiagonal_rows(
        lower, diagonal, upper, rhs, xp=np
    )
    gpu_arrays = tuple(
        cp.asarray(value) for value in (lower, diagonal, upper, rhs)
    )

    # Warm both paths before measuring. The old path is the currently trusted
    # CuPy implementation and deliberately retains its Python x loops.
    solve_cyclic_tridiagonal_rows(*gpu_arrays, xp=cp)
    solve_cyclic_tridiagonal_rows_gpu(*gpu_arrays)
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline_used = pool.used_bytes()
    baseline_total = pool.total_bytes()
    start = perf_counter()
    for _ in range(repeats):
        old_actual = solve_cyclic_tridiagonal_rows(*gpu_arrays, xp=cp)
    cp.cuda.Stream.null.synchronize()
    old_elapsed = (perf_counter() - start) / repeats
    old_pool_used = pool.used_bytes() - baseline_used
    old_pool_total = pool.total_bytes() - baseline_total
    del old_actual
    pool.free_all_blocks()

    baseline_used = pool.used_bytes()
    baseline_total = pool.total_bytes()
    start = perf_counter()
    for _ in range(repeats):
        actual = solve_cyclic_tridiagonal_rows_gpu(*gpu_arrays)
    cp.cuda.Stream.null.synchronize()
    new_elapsed = (perf_counter() - start) / repeats
    new_pool_used = pool.used_bytes() - baseline_used
    new_pool_total = pool.total_bytes() - baseline_total
    result = cp.asnumpy(actual)
    residual = (
        diagonal * result
        + lower * np.roll(result, 1, axis=-2)
        + upper * np.roll(result, -1, axis=-2)
        - rhs
    )
    difference = result - expected
    return {
        "name": name,
        "shape": list(shape),
        "dtype": np.dtype(dtype).name,
        "old_seconds_per_solve": old_elapsed,
        "new_seconds_per_solve": new_elapsed,
        "speedup": old_elapsed / new_elapsed,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel()) / np.linalg.norm(expected.ravel())
        ),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "residual_max": float(np.max(np.abs(residual))),
        "old_pool_used_delta_bytes": int(old_pool_used),
        "old_pool_total_delta_bytes": int(old_pool_total),
        "new_pool_used_delta_bytes": int(new_pool_used),
        "new_pool_total_delta_bytes": int(new_pool_total),
        "old_layout_materializations": 4,
        "new_layout_materializations": 0,
        "new_logical_work_volumes": 3,
        "new_custom_kernel_launches_per_solve": 1,
        "new_python_x_loop": False,
    }


def _semi_implicit_case(cp, *, shape, dtype, repeats):
    rng = np.random.default_rng(20260818)
    state = cp.asarray(rng.normal(scale=0.03, size=shape).astype(dtype))
    intensity = cp.asarray((0.1 + rng.random(shape)).astype(dtype))

    def source(_candidate):
        return intensity

    original_gpu_solver = evolution.solve_cyclic_tridiagonal_rows_gpu
    try:
        evolution.solve_cyclic_tridiagonal_rows_gpu = lambda a, b, c, d: (
            solve_cyclic_tridiagonal_rows(a, b, c, d, xp=cp)
        )
        old = evolution.semi_implicit_trapezoidal_step(
            state,
            source,
            dt_normalized=0.02,
            applied_field=0.4,
            background_intensity=0.1,
            dx_normalized=0.31,
            xp=cp,
        )
        cp.cuda.Stream.null.synchronize()
        start = perf_counter()
        for _ in range(repeats):
            old = evolution.semi_implicit_trapezoidal_step(
                state,
                source,
                dt_normalized=0.02,
                applied_field=0.4,
                background_intensity=0.1,
                dx_normalized=0.31,
                xp=cp,
            )
        cp.cuda.Stream.null.synchronize()
        old_elapsed = (perf_counter() - start) / repeats
    finally:
        evolution.solve_cyclic_tridiagonal_rows_gpu = original_gpu_solver

    new = evolution.semi_implicit_trapezoidal_step(
        state,
        source,
        dt_normalized=0.02,
        applied_field=0.4,
        background_intensity=0.1,
        dx_normalized=0.31,
        xp=cp,
    )
    cp.cuda.Stream.null.synchronize()
    start = perf_counter()
    for _ in range(repeats):
        new = evolution.semi_implicit_trapezoidal_step(
            state,
            source,
            dt_normalized=0.02,
            applied_field=0.4,
            background_intensity=0.1,
            dx_normalized=0.31,
            xp=cp,
        )
    cp.cuda.Stream.null.synchronize()
    new_elapsed = (perf_counter() - start) / repeats
    difference = cp.asnumpy(new - old)
    old_host = cp.asnumpy(old)
    return {
        "shape": list(shape),
        "dtype": np.dtype(dtype).name,
        "old_seconds_per_step": old_elapsed,
        "new_seconds_per_step": new_elapsed,
        "speedup": old_elapsed / new_elapsed,
        "relative_l2_difference": float(
            np.linalg.norm(difference.ravel()) / np.linalg.norm(old_host.ravel())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    import cupy as cp

    cases = []
    for dtype in (np.float32, np.float64):
        cases.append(
            _case(
                cp,
                name="small",
                shape=(32, 64, 64),
                dtype=dtype,
                repeats=args.repeats,
            )
        )
        cases.append(
            _case(
                cp,
                name="medium",
                shape=(120, 256, 128),
                dtype=dtype,
                repeats=args.repeats,
            )
        )
    payload = {
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_id": int(cp.cuda.Device().id),
        "device_name": cp.cuda.runtime.getDeviceProperties(
            cp.cuda.Device().id
        )["name"].decode(),
        "cupy_version": cp.__version__,
        "cases": cases,
        "semi_implicit_medium_float32": _semi_implicit_case(
            cp,
            shape=(120, 256, 128),
            dtype=np.float32,
            repeats=args.repeats,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
