#!/usr/bin/env python3
"""Bounded H200 commissioning for production full-transverse nonlinear PR TD."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

import lcprop.persistence  # initialize persistence/transport registrations
from lcprop.core.backend import BackendSpec, synchronize
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    decode_pr_transverse_timedependent_request,
    encode_pr_transverse_timedependent_request,
)
from lcprop.pr.specs import PR_MATERIAL_ID, PRMaterialSpec
from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
from lcprop.pr.transverse.static import solve_pr_transverse_static_intensity
from lcprop.pr.transverse.timedependent_transport_codec import (
    decode_pr_transverse_timedependent_transport_request,
    encode_pr_transverse_timedependent_transport_request,
)
from lcprop.pr.transverse.transport import state_from_potential
import lcprop.pr.transverse.workflow as workflow_module
from lcprop.transport.defaults import default_transport_registry
from lcprop.transport.io import (
    read_request_package,
    read_result_package,
    write_request_package,
    write_result_package,
)
from lcprop.transport.result_policy import FAST_RESULT_POLICY


EXPECTED_SHA = "1a129dc3cb93f6c8a005a198f40d86a31792f223"
RTOL = 3.0e-10
ATOL = 3.0e-11
PRIMARY_SIZE = 256
PRIMARY_STEPS = 3
PRIMARY_DT = 0.05
SANITY_SIZE = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_json_value(value, *, path: str = "$"):
    """Return a strict JSON-native copy of commissioning evidence."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _normalize_json_value(value.value, path=path)
    if isinstance(value, np.generic):
        return _normalize_json_value(value.item(), path=path)
    module_root = type(value).__module__.split(".", 1)[0]
    if module_root == "cupy":
        cp = __import__("cupy")
        return _normalize_json_value(cp.asnumpy(value), path=path)
    if isinstance(value, np.ndarray):
        return _normalize_json_value(value.tolist(), path=path)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite floating-point evidence at {path}")
        return result
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON evidence key at {path} must be str, got "
                    f"{type(key).__name__}"
                )
            normalized[key] = _normalize_json_value(
                item, path=f"{path}.{key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"unsupported JSON evidence type at {path}: {type(value).__name__}"
    )


def _canonical_json(value) -> bytes:
    normalized = _normalize_json_value(value)
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value, *, canonical: bool = False):
    """Atomically retain normalized evidence and return its JSON-native value."""

    normalized = _normalize_json_value(value)
    if canonical:
        payload = _canonical_json(normalized)
    else:
        payload = (
            json.dumps(
                normalized,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return normalized


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _request(
    *,
    size: int,
    backend: str,
    steps: int = PRIMARY_STEPS,
    dt: float = PRIMARY_DT,
    gain: float = 0.025,
    initial_psi=None,
) -> PRTransverseRunRequest:
    aperture = 64.0
    spectral_bin = 2.0 * np.pi / aperture
    return PRTransverseRunRequest(
        grid=GridSpec(
            Nx=size,
            Ny=size,
            x_aperture_um=aperture,
            y_aperture_um=aperture,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    name="pump",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=22.0,
                    waist_y_um=19.0,
                    tilt_x_rad_per_um=3.0 * spectral_bin,
                    tilt_y_rad_per_um=2.0 * spectral_bin,
                    coherence_group="nonlinear-td-commissioning",
                ),
                BeamChannel(
                    name="signal",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=21.0,
                    waist_y_um=20.0,
                    tilt_x_rad_per_um=-2.0 * spectral_bin,
                    tilt_y_rad_per_um=-3.0 * spectral_bin,
                    phase_rad=0.37,
                    coherence_group="nonlinear-td-commissioning",
                ),
            ),
            coherence="coherent",
        ),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=gain,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=dt,
            optical_substeps=1,
        ),
        backend=BackendSpec(backend=backend, precision="float64", verbose=False),
        initial_psi=initial_psi,
    )


def _resolved_source(size: int, *, planes: int = 2) -> np.ndarray:
    x = np.arange(size)[:, None]
    y = np.arange(size)[None, :]
    plane = (
        1.0
        + 0.12 * np.sin(2.0 * np.pi * x / size)
        + 0.07 * np.cos(2.0 * np.pi * y / size)
        + 0.04 * np.cos(2.0 * np.pi * (2.0 * x + y) / size)
    )
    return np.stack(tuple(plane + 0.01 * index for index in range(planes)))


def _host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _comparison(actual, expected) -> dict[str, object]:
    actual_array = _host(actual)
    expected_array = _host(expected)
    difference = actual_array - expected_array
    denominator = float(np.linalg.norm(expected_array.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel()))
    if denominator:
        relative_l2 /= denominator
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    passed = bool(np.allclose(actual_array, expected_array, rtol=RTOL, atol=ATOL))
    if not passed:
        raise AssertionError(
            f"parity failed: rel_l2={relative_l2}, max_abs={maximum}"
        )
    return {
        "relative_l2": relative_l2,
        "maximum_absolute": maximum,
        "shape": list(actual_array.shape),
        "actual_dtype": str(actual_array.dtype),
        "reference_dtype": str(expected_array.dtype),
        "rtol": RTOL,
        "atol": ATOL,
        "passed": passed,
    }


class _Instrumentation:
    def __init__(self) -> None:
        self.optical_seconds = 0.0
        self.material_seconds = 0.0
        self.optical_calls = 0
        self.material_calls = 0
        self.final_source = None
        self._optical = workflow_module._optical_pass
        self._material = workflow_module.imex_euler_step

    def install(self) -> None:
        def optical(*args, **kwargs):
            xp = kwargs["grid"].xp
            synchronize(xp)
            started = perf_counter()
            value = self._optical(*args, **kwargs)
            synchronize(xp)
            self.optical_seconds += perf_counter() - started
            self.optical_calls += 1
            self.final_source = _host(value[1]).copy()
            return value

        def material(*args, **kwargs):
            xp = kwargs["xp"]
            synchronize(xp)
            started = perf_counter()
            value = self._material(*args, **kwargs)
            synchronize(xp)
            self.material_seconds += perf_counter() - started
            self.material_calls += 1
            return value

        workflow_module._optical_pass = optical
        workflow_module.imex_euler_step = material

    def restore(self) -> None:
        workflow_module._optical_pass = self._optical
        workflow_module.imex_euler_step = self._material


def _run(request: PRTransverseRunRequest):
    instrumentation = _Instrumentation()
    instrumentation.install()
    try:
        started = perf_counter()
        result = workflow_module.run_pr_transverse_timedependent(request)
        total = perf_counter() - started
    finally:
        instrumentation.restore()
    if instrumentation.final_source is None:
        raise AssertionError("production workflow did not expose a final source pass")
    return result, instrumentation.final_source, {
        "scientific_total_seconds": total,
        "optical_pass_seconds": instrumentation.optical_seconds,
        "nonlinear_material_seconds": instrumentation.material_seconds,
        "optical_pass_calls": instrumentation.optical_calls,
        "instrumented_material_calls": instrumentation.material_calls,
    }


def _state(result, *, backend: str):
    profile = result.resolved_profile
    xp = __import__("cupy") if backend == "cupy" else np
    state = state_from_potential(
        xp.asarray(result.psi_final),
        dx_normalized=float(profile["dx_normalized"]),
        dy_normalized=float(profile["dy_normalized"]),
        h_y=float(profile["dielectric"]["h_y"]),
        applied_field_x=float(profile["boundary"]["applied_field_x"]),
        xp=xp,
    )
    active = project_active_field(
        state.E_x,
        state.E_y,
        profile=PRTransverseProjectionProfile(**profile["projection"]),
        xp=xp,
    )
    return state, active


def _primary_case(size: int):
    numpy_result, numpy_source, numpy_timing = _run(
        _request(size=size, backend="numpy")
    )
    cupy_result, cupy_source, cupy_timing = _run(
        _request(size=size, backend="cupy")
    )
    numpy_state, numpy_active = _state(numpy_result, backend="numpy")
    cupy_state, cupy_active = _state(cupy_result, backend="cupy")
    comparisons = {
        "psi": _comparison(cupy_result.psi_final, numpy_result.psi_final),
        "E_x": _comparison(cupy_state.E_x, numpy_state.E_x),
        "E_y": _comparison(cupy_state.E_y, numpy_state.E_y),
        "E_active": _comparison(cupy_active, numpy_active),
        "final_optical_field": _comparison(
            cupy_result.A_final, numpy_result.A_final
        ),
        "final_source_intensity": _comparison(cupy_source, numpy_source),
    }
    diagnostic_names = (
        "carrier_integrals_per_z",
        "carrier_relative_drift_max",
        "optical_power_relative_drift",
        "potential_mean_max_abs",
        "nonlinear_td_rhs_rms",
        "nonlinear_td_rhs_max",
        "carrier_density_minimum",
    )
    diagnostic_comparisons = {
        name: _comparison(
            cupy_result.diagnostics[name], numpy_result.diagnostics[name]
        )
        for name in diagnostic_names
    }
    exact_names = (
        "finite_material_state",
        "finite_optical_state",
        "physical_state_valid",
        "integrator_policy",
        "complete_final_optical_replay",
        "material_response",
        "material_response_validation",
        "material_response_calls",
        "accepted_material_steps",
        "optical_passes_completed",
        "source_cadence",
        "nonlinear_material_iterations",
    )
    for name in exact_names:
        if cupy_result.diagnostics[name] != numpy_result.diagnostics[name]:
            raise AssertionError(f"diagnostic parity failed for {name}")
    for result in (numpy_result, cupy_result):
        if result.status != "completed" or result.completed_steps != PRIMARY_STEPS:
            raise AssertionError("production run did not complete three intervals")
        if result.diagnostics["material_response"] != PR_MATERIAL_RESPONSE_NONLINEAR:
            raise AssertionError("production did not dispatch nonlinear material")
        if result.diagnostics["material_response_validation"] != "validated":
            raise AssertionError("nonlinear production status is not Validated")
        if not result.diagnostics["physical_state_valid"]:
            raise AssertionError("nonlinear production state is not physical")
        if result.diagnostics["potential_mean_max_abs"] > 2.0e-13:
            raise AssertionError("potential gauge is not zero")
    backend = cupy_result.backend_summary
    if backend.get("backend") != "cupy" or not backend.get("is_gpu"):
        raise AssertionError(f"CuPy backend did not resolve on GPU: {backend}")
    return (
        {
            "numpy_timing": numpy_timing,
            "cupy_timing": cupy_timing,
            "comparisons": comparisons,
            "diagnostic_comparisons": diagnostic_comparisons,
            "exact_diagnostic_fields": list(exact_names),
            "numpy_diagnostics": numpy_result.diagnostics,
            "cupy_diagnostics": cupy_result.diagnostics,
            "backend": backend,
        },
        cupy_result,
    )


def _with_fixed_source(source, function):
    original = workflow_module._optical_pass

    def optical(A0, psi, **kwargs):
        return A0.copy(), kwargs["grid"].xp.asarray(
            source, dtype=kwargs["grid"].real_dtype
        )

    workflow_module._optical_pass = optical
    try:
        return function()
    finally:
        workflow_module._optical_pass = original


def _static_sanity() -> dict[str, object]:
    source = _resolved_source(SANITY_SIZE)
    request = _request(size=SANITY_SIZE, backend="cupy", steps=1, dt=0.01, gain=0.0)
    k0 = request.material.characteristic_wavenumber_per_um
    dx = k0 * request.grid.x_aperture_um / SANITY_SIZE
    dy = k0 * request.grid.y_aperture_um / SANITY_SIZE
    static = solve_pr_transverse_static_intensity(
        source, dx_normalized=dx, dy_normalized=dy
    )
    if not static.converged:
        raise AssertionError("bounded static oracle did not converge")
    evolved = _with_fixed_source(
        source,
        lambda: workflow_module.run_pr_transverse_timedependent(
            replace(request, initial_psi=static.psi)
        ),
    )
    difference = evolved.psi_final - static.psi
    relative_change = float(np.linalg.norm(difference) / np.linalg.norm(static.psi))
    if relative_change > 1.0e-4:
        raise AssertionError(f"static sanity change is too large: {relative_change}")
    return {
        "grid": [SANITY_SIZE, SANITY_SIZE],
        "dt_normalized": 0.01,
        "static_converged": True,
        "relative_psi_change_after_one_td_step": relative_change,
        "final_nonlinear_td_rhs_rms": evolved.diagnostics["nonlinear_td_rhs_rms"],
        "final_nonlinear_td_rhs_max": evolved.diagnostics["nonlinear_td_rhs_max"],
        "carrier_density_minimum": evolved.diagnostics["carrier_density_minimum"],
        "physical_state_valid": evolved.diagnostics["physical_state_valid"],
    }


def _timestep_sanity() -> dict[str, object]:
    source = _resolved_source(SANITY_SIZE)
    solutions = []
    steps = (10, 20, 40)
    for count in steps:
        request = _request(
            size=SANITY_SIZE,
            backend="cupy",
            steps=count,
            dt=0.2 / count,
            gain=0.0,
        )
        result = _with_fixed_source(
            source, lambda request=request: workflow_module.run_pr_transverse_timedependent(request)
        )
        solutions.append(result.psi_final)
    ratio = float(
        np.linalg.norm(solutions[0] - solutions[1])
        / np.linalg.norm(solutions[1] - solutions[2])
    )
    if not 1.7 < ratio < 2.3:
        raise AssertionError(f"first-order refinement sanity failed: {ratio}")
    return {
        "grid": [SANITY_SIZE, SANITY_SIZE],
        "final_time_normalized": 0.2,
        "step_counts": list(steps),
        "dt_normalized": [0.2 / count for count in steps],
        "successive_difference_ratio": ratio,
        "expected_order": 1,
        "accepted_ratio_interval": [1.7, 2.3],
    }


def _cancellation_sanity() -> dict[str, object]:
    source = _resolved_source(SANITY_SIZE)
    request = _request(size=SANITY_SIZE, backend="cupy", steps=3, gain=0.0)
    token = CancellationToken()
    original = workflow_module.imex_euler_step

    def cancel_after_candidate(*args, **kwargs):
        candidate = original(*args, **kwargs)
        token.cancel()
        token.cancel()
        return candidate

    def execute():
        workflow_module.imex_euler_step = cancel_after_candidate
        try:
            return workflow_module.run_pr_transverse_timedependent(
                request, cancellation_token=token
            )
        finally:
            workflow_module.imex_euler_step = original

    result = _with_fixed_source(source, execute)
    if result.status != "cancelled" or result.completed_steps != 0:
        raise AssertionError("discardable nonlinear candidate became authoritative")
    if not np.array_equal(result.psi_final, result.psi_initial):
        raise AssertionError("cancelled run did not preserve accepted material state")
    if not np.array_equal(result.A_final, result.A_initial):
        raise AssertionError("cancelled final replay did not preserve optical state")
    return {
        "grid": [SANITY_SIZE, SANITY_SIZE],
        "status": result.status,
        "completed_steps": result.completed_steps,
        "cancellation_stage": result.diagnostics["cancellation_observed_stage"],
        "candidate_discarded": True,
        "last_accepted_psi_preserved_bitwise": True,
        "final_optical_replay_preserved_bitwise": True,
        "repeated_cancel_harmless": True,
    }


def _fast_package(output: Path, result, request) -> dict[str, object]:
    package = output / "primary_fast"
    registry = default_transport_registry()
    run_id = os.environ.get("SLURM_JOB_ID", "local-preflight")
    write_request_package(
        package,
        registry=registry,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        request=request,
        run_id=run_id,
        provenance={"production_git_sha": EXPECTED_SHA},
        result_policy=FAST_RESULT_POLICY,
    )
    decoded_request = read_request_package(package, registry=registry)
    write_result_package(
        package,
        codec=decoded_request.codec,
        result=result,
        request_envelope=decoded_request.envelope,
        provenance={"production_git_sha": EXPECTED_SHA},
    )
    started = perf_counter()
    decoded = read_result_package(package, registry=registry)
    run_data = pr_transverse_result_to_run_data(decoded.result)
    product_seconds = perf_counter() - started
    if decoded.result.psi_initial is not None or decoded.result.psi_final is not None:
        raise AssertionError("Fast package retained material volumes")
    required = {"input_intensity", "output_intensity", "far_field_intensity"}
    if not required.issubset(set(run_data.fields)):
        raise AssertionError("Fast product conversion omitted optical products")
    summary = run_data.diagnostics["summary"].values
    transverse = run_data.diagnostics["transverse_pr"].values
    if summary["physics_profile"]["material_response"]["model"] != (
        PR_MATERIAL_RESPONSE_NONLINEAR
    ):
        raise AssertionError("Fast products lost nonlinear material provenance")
    if transverse["material_response_validation"] != "validated":
        raise AssertionError("Fast products lost nonlinear validation status")
    files = sorted(path for path in package.rglob("*") if path.is_file())
    return {
        "policy": decoded.result.retention_summary["policy"],
        "omitted_fields": decoded.result.retention_summary["omitted_fields"],
        "retained_optical_endpoints": ["A_initial", "A_final"],
        "product_fields": sorted(run_data.fields),
        "product_conversion_seconds": product_seconds,
        "geometry": {
            "x": int(run_data.geometry.x.size),
            "y": int(run_data.geometry.y.size),
            "z": int(run_data.geometry.z.size),
        },
        "package_bytes": sum(path.stat().st_size for path in files),
        "files": {str(path.relative_to(output)): _sha256(path) for path in files},
    }


def _serialization_smoke(request: PRTransverseRunRequest) -> dict[str, object]:
    experiment = encode_pr_transverse_timedependent_request(request)
    experiment_decoded = decode_pr_transverse_timedependent_request(experiment)
    portable = encode_pr_transverse_timedependent_transport_request(request)
    portable_decoded = decode_pr_transverse_timedependent_transport_request(
        portable.payload.metadata, portable.payload.arrays
    )
    if asdict(experiment_decoded) != asdict(request):
        raise AssertionError("experiment request round-trip changed the request")
    if asdict(portable_decoded) != asdict(request):
        raise AssertionError("transport request round-trip changed the request")
    return {
        "experiment_roundtrip": True,
        "transport_roundtrip": True,
        "material_response": request.material_response.model,
        "applied_field": request.material.applied_field,
    }


def _verify_source(source_root: Path) -> dict[str, object]:
    source_sha = _git("rev-parse", "HEAD", cwd=source_root)
    porcelain = _git("status", "--porcelain", cwd=source_root)
    if source_sha != EXPECTED_SHA or porcelain:
        raise RuntimeError(
            f"unclean or wrong production source: {source_sha!r}, {porcelain!r}"
        )
    return {
        "production_git_sha": source_sha,
        "production_git_status_porcelain": porcelain,
        "source_clean": True,
    }


def _scientific_evidence_object(
    *,
    primary,
    static,
    timestep,
    cancellation,
    fast,
    cupy_memory_pool,
) -> dict[str, object]:
    """Build the complete canonical scientific payload retained by a run."""

    return {
        "request": {
            "grid": [PRIMARY_SIZE, PRIMARY_SIZE],
            "Nz": 2,
            "accepted_intervals": PRIMARY_STEPS,
            "dt_normalized": PRIMARY_DT,
            "applied_field": 0.0,
            "gain_length_product": 0.025,
            "precision": "float64",
            "result_policy": "fast",
        },
        "primary": {
            "numpy_timing": primary["numpy_timing"],
            "cupy_timing": primary["cupy_timing"],
            "comparisons": primary["comparisons"],
            "diagnostic_comparisons": primary["diagnostic_comparisons"],
            "cupy_diagnostics": primary["cupy_diagnostics"],
        },
        "static_sanity": static,
        "timestep_sanity": timestep,
        "cancellation_sanity": cancellation,
        "fast_package": {
            key: value for key, value in fast.items() if key != "files"
        },
        "cupy_memory_pool": cupy_memory_pool,
    }


def _preflight(output: Path, source_root: Path) -> None:
    provenance = _verify_source(source_root)
    request = _request(size=SANITY_SIZE, backend="numpy", steps=1, gain=0.0)
    serialization = _serialization_smoke(request)
    result, source, timing = _run(request)
    if result.status != "completed" or not result.diagnostics["physical_state_valid"]:
        raise AssertionError("NumPy preflight did not complete physically")
    payload = {
        **provenance,
        "mode": "local_preflight",
        "request_serialization": serialization,
        "numpy_smoke": {
            "grid": [SANITY_SIZE, SANITY_SIZE],
            "completed_steps": result.completed_steps,
            "optical_passes_completed": result.diagnostics[
                "optical_passes_completed"
            ],
            "carrier_density_minimum": result.diagnostics[
                "carrier_density_minimum"
            ],
            "nonlinear_td_rhs_max": result.diagnostics["nonlinear_td_rhs_max"],
            "source_shape": list(source.shape),
            "timing": timing,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    normalized = _write_json(output / "local_preflight.json", payload)
    print(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False))


def _commission(output: Path, source_root: Path, launch_root: Path) -> None:
    provenance = _verify_source(source_root)
    import cupy as cp

    output.mkdir(parents=True, exist_ok=False)
    cp.cuda.Device().use()
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before_pool = {"used_bytes": pool.used_bytes(), "total_bytes": pool.total_bytes()}
    primary, cupy_result = _primary_case(PRIMARY_SIZE)
    _write_json(output / "primary_quantitative.json", primary)
    static = _static_sanity()
    _write_json(output / "static_sanity.json", static)
    timestep = _timestep_sanity()
    _write_json(output / "timestep_sanity.json", timestep)
    cancellation = _cancellation_sanity()
    _write_json(output / "cancellation_sanity.json", cancellation)
    fast = _fast_package(
        output,
        cupy_result,
        _request(size=PRIMARY_SIZE, backend="cupy"),
    )
    _write_json(output / "fast_package_summary.json", fast)
    cp.cuda.Stream.null.synchronize()
    after_pool = {"used_bytes": pool.used_bytes(), "total_bytes": pool.total_bytes()}
    cupy_memory_pool = {"before": before_pool, "after": after_pool}
    _write_json(output / "cupy_memory_pool.json", cupy_memory_pool)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = device["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    if "H200" not in str(device_name).upper():
        raise RuntimeError(f"allocated GPU is not an H200: {device_name}")

    scientific_rows = _scientific_evidence_object(
        primary=primary,
        static=static,
        timestep=timestep,
        cancellation=cancellation,
        fast=fast,
        cupy_memory_pool=cupy_memory_pool,
    )
    scientific_path = output / "scientific_rows.json"
    _write_json(scientific_path, scientific_rows, canonical=True)
    scientific_checksum = _sha256(scientific_path)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "gpu_name": device_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    metrics = {
        "schema_version": 1,
        **provenance,
        "backend": "cupy",
        "precision": "float64",
        "environment": environment,
        "primary": primary,
        "static_sanity": static,
        "timestep_sanity": timestep,
        "cancellation_sanity": cancellation,
        "fast_package": fast,
        "cupy_memory_pool": cupy_memory_pool,
        "canonical_scientific_data_sha256": scientific_checksum,
        "timing_semantics": (
            "Primary CuPy timing is the first scientific CuPy measurement after "
            "explicit CUDA/CuPy initialization; synchronized around instrumented "
            "optical and nonlinear material calls. It is not process-cold startup."
        ),
    }
    metrics_path = output / "commissioning_metrics.json"
    normalized_metrics = _write_json(metrics_path, metrics)
    launch_files = {
        name: _sha256(launch_root / name)
        for name in (
            "lcprop-1a129dc.bundle",
            "pr_full_transverse_nonlinear_td_h200_commissioning.py",
            "pr_full_transverse_nonlinear_td_h200.sbatch",
            "launch_manifest.json",
            "launch_checksums.sha256",
        )
    }
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        **provenance,
        "backend": "cupy",
        "precision": "float64",
        "harness_sha256": _sha256(Path(__file__)),
        "source_archive_sha256": launch_files["lcprop-1a129dc.bundle"],
        "canonical_scientific_data_sha256": scientific_checksum,
        "launch_input_checksums": launch_files,
        "artifact_checksums_at_harness_exit": {
            str(path.relative_to(output)): _sha256(path) for path in artifacts
        },
        "required_post_retrieval_artifacts": [
            "SCHEDULER_ACCOUNTING_FINAL.txt",
            "gpu_memory_baseline.csv",
            "gpu_process_memory_samples.csv",
            "LOCAL_RECONSTRUCTION.json",
            "REMOTE_CLEANUP_VERIFIED.txt",
        ],
    }
    _write_json(output / "evidence_manifest_in_job.json", manifest)
    print(json.dumps(normalized_metrics, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "commission"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--launch-root", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        _preflight(args.output, args.source_root)
        return
    if args.launch_root is None:
        parser.error("--launch-root is required for commissioning")
    _commission(args.output, args.source_root, args.launch_root)


if __name__ == "__main__":
    main()
