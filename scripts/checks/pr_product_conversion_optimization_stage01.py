#!/usr/bin/env python3
"""Benchmark the bounded PR transverse product-conversion optimization."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import lcprop.pr.transverse.products as product_module
from lcprop.pr.transverse.specs import PR_FULL_TRANSVERSE_PROFILE_V1
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunResult
from lcprop.pr.transverse.transport import state_from_potential


SCHEMA = "lcprop.pr_product_conversion_optimization_stage01.v1"


def _synthetic_result(
    *, grid_size: int, material_slices: int, seed: int, scattering: bool
) -> PRTransverseStaticRunResult:
    rng = np.random.default_rng(seed)
    x = np.arange(grid_size, dtype=np.float32)[:, None]
    y = np.arange(grid_size, dtype=np.float32)[None, :]
    z = np.arange(material_slices, dtype=np.float32)[:, None, None]
    base = np.cos(2.0 * np.pi * (3.0 * x / grid_size + 2.0 * y / grid_size))
    modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * z / material_slices)
    potential = (0.015 * modulation * base[None, :, :]).astype(np.float32)
    if scattering:
        potential += rng.normal(0.0, 2.0e-4, potential.shape).astype(np.float32)
    phase = 2.0 * np.pi * (-0.13 * x / grid_size + 0.07 * y / grid_size)
    envelope = np.exp(
        -(
            (x - 0.45 * grid_size) ** 2
            + (y - 0.52 * grid_size) ** 2
        )
        / (0.18 * grid_size) ** 2
    )
    initial = (envelope * np.exp(1j * phase))[None].astype(np.complex64)
    final = (envelope * np.exp(1j * (phase + 0.1 * base)))[None].astype(
        np.complex64
    )
    source = (
        1.3 + 0.05 * modulation * np.cos(2.0 * np.pi * x / grid_size)[None]
    )
    source = np.broadcast_to(
        source, (material_slices, grid_size, grid_size)
    ).astype(np.float32, copy=True)
    zeros = np.zeros_like(potential)
    profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
        "workflow": "pr_transverse_static",
        "dx_normalized": 0.25,
        "dy_normalized": 0.25,
        "material": {
            "dark_intensity": 0.01,
            "uniform_background_intensity": 0.0,
            "refractive_index": 2.4,
        },
        "beam_request": {
            "channels": (
                {
                    "wavelength_um": 0.633,
                    "tilt_x_rad_per_um": -0.99,
                    "tilt_y_rad_per_um": 0.0,
                    "waist_x_um": 200.0,
                    "waist_y_um": 200.0,
                },
            ),
        },
        "dielectric": {"h_y": 1.0},
        "boundary": {"applied_field_x": 0.0},
        "projection": {
            "profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
            "g_x": 1.0,
            "g_y": 0.0,
        },
    }
    if scattering:
        profile["canonical_scattering"] = {"realization_seed": seed}
    return PRTransverseStaticRunResult(
        A_initial=initial,
        A_final=final,
        psi_initial=np.zeros_like(potential),
        psi_final=potential,
        source_intensity_stack=source,
        equilibrium_residual_stack=zeros,
        td_rhs_residual_stack=zeros.copy(),
        power_initial=float(np.sum(np.abs(initial) ** 2)),
        power_final=float(np.sum(np.abs(final) ** 2)),
        converged=True,
        completed_coupled_iterations=1,
        iteration_records=(),
        material_iteration_records=(),
        discrete_iteration_records=(),
        grid_summary={
            "Nx": grid_size,
            "Ny": grid_size,
            "Nz": material_slices,
            "dx_um": 1000.0 / grid_size,
            "dy_um": 1000.0 / grid_size,
            "dz_um": 50.0,
        },
        launch_summary={"coherence_groups": ("benchmark",)},
        backend_summary={"backend": "numpy", "precision": "float32"},
        resolved_profile=profile,
        replay_diagnostics={"complete_independent_replay": True},
        diagnostics={"equilibrium_residual_rms": 0.0},
        timing={"total_seconds": 0.0},
        status="converged",
    )


def _direct_state(potential, **keywords):
    return state_from_potential(potential, xp=np, **keywords)


@contextmanager
def _adapter_mode(*, optimized: bool, counters: dict[str, float]):
    original_state = product_module._state_from_potential_for_products
    original_spectrum = product_module.direction_cosine_spectrum
    original_intensity = product_module.total_intensity

    def timed_state(*args, **kwargs):
        started = perf_counter()
        value = (original_state if optimized else _direct_state)(*args, **kwargs)
        counters["state_reconstruction_seconds"] += perf_counter() - started
        return value

    def timed_spectrum(*args, **kwargs):
        started = perf_counter()
        value = original_spectrum(*args, **kwargs)
        counters["far_field_seconds"] += perf_counter() - started
        return value

    def timed_intensity(*args, **kwargs):
        started = perf_counter()
        value = original_intensity(*args, **kwargs)
        counters["intensity_seconds"] += perf_counter() - started
        return value

    product_module._state_from_potential_for_products = timed_state
    product_module.direction_cosine_spectrum = timed_spectrum
    product_module.total_intensity = timed_intensity
    try:
        yield
    finally:
        product_module._state_from_potential_for_products = original_state
        product_module.direction_cosine_spectrum = original_spectrum
        product_module.total_intensity = original_intensity


def _run_adapter(result, *, optimized: bool, repeats: int):
    samples = []
    counters = {
        "state_reconstruction_seconds": 0.0,
        "far_field_seconds": 0.0,
        "intensity_seconds": 0.0,
    }
    run_data = None
    with _adapter_mode(optimized=optimized, counters=counters):
        for _ in range(repeats):
            started = perf_counter()
            run_data = product_module.pr_transverse_static_result_to_run_data(result)
            samples.append(perf_counter() - started)
    return run_data, {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        **{key: value / repeats for key, value in counters.items()},
    }


def _equivalence(reference, candidate) -> dict[str, object]:
    field_keys_equal = tuple(reference.fields.keys()) == tuple(candidate.fields.keys())
    curve_keys_equal = tuple(reference.curves.keys()) == tuple(candidate.curves.keys())
    field_metadata_equal = field_keys_equal and all(
        (
            reference.fields[key].display_name,
            reference.fields[key].axes,
            reference.fields[key].kind,
            reference.fields[key].units,
            reference.fields[key].default_view,
            reference.fields[key].quantity,
            reference.fields[key].value_unit,
            reference.fields[key].colormap,
            reference.fields[key].source_volume_key,
        )
        == (
            candidate.fields[key].display_name,
            candidate.fields[key].axes,
            candidate.fields[key].kind,
            candidate.fields[key].units,
            candidate.fields[key].default_view,
            candidate.fields[key].quantity,
            candidate.fields[key].value_unit,
            candidate.fields[key].colormap,
            candidate.fields[key].source_volume_key,
        )
        for key in reference.fields.keys()
    )
    fields_bitwise = field_keys_equal and all(
        np.array_equal(reference.fields[key].data, candidate.fields[key].data)
        and reference.fields[key].data.dtype == candidate.fields[key].data.dtype
        for key in reference.fields.keys()
    )
    coordinates_bitwise = field_keys_equal and all(
        reference.fields[key].coordinates.keys()
        == candidate.fields[key].coordinates.keys()
        and all(
            np.array_equal(
                reference.fields[key].coordinates[axis],
                candidate.fields[key].coordinates[axis],
            )
            for axis in reference.fields[key].coordinates
        )
        for key in reference.fields.keys()
    )
    curves_bitwise = curve_keys_equal and all(
        np.array_equal(reference.curves[key].x, candidate.curves[key].x)
        and np.array_equal(reference.curves[key].y, candidate.curves[key].y)
        for key in reference.curves.keys()
    )
    return {
        "field_keys_equal": field_keys_equal,
        "field_metadata_equal": field_metadata_equal,
        "fields_bitwise_equal": fields_bitwise,
        "coordinates_bitwise_equal": coordinates_bitwise,
        "curve_keys_equal": curve_keys_equal,
        "curves_bitwise_equal": curves_bitwise,
        "diagnostic_keys_equal": tuple(reference.diagnostics.keys())
        == tuple(candidate.diagnostics.keys()),
        "diagnostic_values_equal": all(
            reference.diagnostics[key].values == candidate.diagnostics[key].values
            for key in reference.diagnostics.keys()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--slices", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = []
    for scattering in (False, True):
        result = _synthetic_result(
            grid_size=args.grid,
            material_slices=args.slices,
            seed=2137319267,
            scattering=scattering,
        )
        baseline, baseline_timing = _run_adapter(
            result, optimized=False, repeats=args.repeats
        )
        candidate, candidate_timing = _run_adapter(
            result, optimized=True, repeats=args.repeats
        )
        equivalence = _equivalence(baseline, candidate)
        speedup = baseline_timing["median_seconds"] / candidate_timing["median_seconds"]
        cases.append({
            "scattering": scattering,
            "baseline": baseline_timing,
            "candidate": candidate_timing,
            "speedup": speedup,
            "equivalence": equivalence,
            "device_to_host_transfers": 0,
            "device_to_host_bytes": 0,
            "field_count": len(candidate.fields),
            "curve_count": len(candidate.curves),
            "diagnostic_count": len(candidate.diagnostics),
        })
    payload = {
        "schema": SCHEMA,
        "grid": args.grid,
        "material_slices": args.slices,
        "dtype": "float32/complex64",
        "repeats": args.repeats,
        "chunk_target_bytes": product_module._STATE_RECONSTRUCTION_CHUNK_BYTES,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
