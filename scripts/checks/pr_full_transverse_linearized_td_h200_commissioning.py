#!/usr/bin/env python3
"""Bounded H200 commissioning for production full-transverse linearized PR TD."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from lcprop.core.backend import BackendSpec, synchronize
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.specs import PR_MATERIAL_ID, PRMaterialSpec
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.linearized_timedependent_reference import (
    solve_pr_biased_linearized_timedependent_reference,
)
from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
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


EXPECTED_SHA = "f865c5e30c5233c8f5b89e478f0bde68b28b2763"
RTOL = 3.0e-11
ATOL = 3.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _request(*, size: int, backend: str, bias: float, steps: int = 3,
             dt: float = 0.2, gain: float = 0.025) -> PRTransverseRunRequest:
    aperture = 64.0
    spectral_bin = 2.0 * np.pi / aperture
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="pump",
                wavelength_um=0.633,
                power_mW=1.0,
                waist_x_um=22.0,
                waist_y_um=19.0,
                tilt_x_rad_per_um=3.0 * spectral_bin,
                tilt_y_rad_per_um=2.0 * spectral_bin,
                coherence_group="td-commissioning",
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
                coherence_group="td-commissioning",
            ),
        ),
        coherence="coherent",
    )
    return PRTransverseRunRequest(
        grid=GridSpec(
            Nx=size,
            Ny=size,
            x_aperture_um=aperture,
            y_aperture_um=aperture,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=gain,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=bias,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps, dt_normalized=dt, optical_substeps=1
        ),
        backend=BackendSpec(backend=backend, precision="float64", verbose=False),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        ),
    )


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
        self._material = (
            workflow_module.solve_pr_biased_linearized_timedependent_reference
        )

    def install(self) -> None:
        def optical(*args, **kwargs):
            xp = kwargs["grid"].xp
            synchronize(xp)
            started = perf_counter()
            value = self._optical(*args, **kwargs)
            synchronize(xp)
            self.optical_seconds += perf_counter() - started
            self.optical_calls += 1
            self.final_source = np.asarray(
                value[1].get() if hasattr(value[1], "get") else value[1]
            ).copy()
            return value

        def material(*args, **kwargs):
            backend = kwargs["backend"]
            xp = __import__("cupy") if backend.backend == "cupy" else np
            synchronize(xp)
            started = perf_counter()
            value = self._material(*args, **kwargs)
            synchronize(xp)
            self.material_seconds += perf_counter() - started
            self.material_calls += 1
            return value

        workflow_module._optical_pass = optical
        workflow_module.solve_pr_biased_linearized_timedependent_reference = material

    def restore(self) -> None:
        workflow_module._optical_pass = self._optical
        workflow_module.solve_pr_biased_linearized_timedependent_reference = self._material


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
        "linearized_material_seconds": instrumentation.material_seconds,
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
        profile=PRTransverseProjectionProfile(**result.resolved_profile["projection"]),
        xp=xp,
    )
    return state, active


def _case(size: int, bias: float):
    numpy_result, numpy_source, numpy_timing = _run(
        _request(size=size, backend="numpy", bias=bias)
    )
    cupy_result, cupy_source, cupy_timing = _run(
        _request(size=size, backend="cupy", bias=bias)
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
    if cupy_result.completed_steps != numpy_result.completed_steps:
        raise AssertionError("accepted-step metadata differs")
    if cupy_result.diagnostics["material_response_calls"] != (
        numpy_result.diagnostics["material_response_calls"]
    ):
        raise AssertionError("material-response call counts differ")
    diagnostic_comparisons = {
        name: _comparison(
            cupy_result.diagnostics[name], numpy_result.diagnostics[name]
        )
        for name in (
            "carrier_integrals_per_z",
            "carrier_relative_drift_max",
            "optical_power_relative_drift",
            "potential_mean_max_abs",
            "linearized_rhs_rms",
            "linearized_rhs_max",
        )
    }
    exact_diagnostic_fields = (
        "finite_material_state",
        "finite_optical_state",
        "integrator_policy",
        "complete_final_optical_replay",
        "material_response",
        "material_response_validation",
        "material_response_calls",
        "accepted_material_steps",
        "source_cadence",
        "nonlinear_material_iterations",
    )
    for name in exact_diagnostic_fields:
        if cupy_result.diagnostics[name] != numpy_result.diagnostics[name]:
            raise AssertionError(f"diagnostic parity failed for {name}")
    for result in (numpy_result, cupy_result):
        if result.status != "completed" or result.completed_steps != 3:
            raise AssertionError("production run did not complete three intervals")
        if result.diagnostics["material_response"] != "linearized":
            raise AssertionError("wrong material response")
        if not result.diagnostics["finite_optical_state"]:
            raise AssertionError("non-finite optical state")
        if abs(float(result.diagnostics["potential_mean_max_abs"])) > 2.0e-13:
            raise AssertionError("potential gauge is not zero")
    backend = cupy_result.backend_summary
    if backend.get("backend") != "cupy" or not backend.get("is_gpu"):
        raise AssertionError(f"CuPy backend did not resolve on GPU: {backend}")
    return ({
        "bias": bias,
        "numpy_timing": numpy_timing,
        "cupy_timing": cupy_timing,
        "comparisons": comparisons,
        "diagnostic_comparisons": diagnostic_comparisons,
        "exact_diagnostic_fields": list(exact_diagnostic_fields),
        "numpy_diagnostics": numpy_result.diagnostics,
        "cupy_diagnostics": cupy_result.diagnostics,
        "backend": backend,
        "power": {
            "initial": cupy_result.power_initial,
            "final": cupy_result.power_final,
            "relative_drift": cupy_result.diagnostics[
                "optical_power_relative_drift"
            ],
        },
    }, numpy_result, cupy_result, cupy_source)


def _modal_sanity(source: np.ndarray, result, *, bias: float) -> dict[str, object]:
    profile = result.resolved_profile
    spec_values = dict(
        reference_intensity=1.5,
        dx_normalized=float(profile["dx_normalized"]),
        dy_normalized=float(profile["dy_normalized"]),
    )
    nx, ny = source.shape[-2:]
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    controlled_source = 1.5 + 0.08 * np.cos(
        2.0 * np.pi * (5.0 * x / nx + 5.0 * y / ny)
    )
    initial = np.zeros_like(controlled_source)
    plus = solve_pr_biased_linearized_timedependent_reference(
        controlled_source,
        time_normalized=0.2,
        spec=PRBiasedLinearizedReferenceSpec(applied_field=bias, **spec_values),
        initial_delta_psi=initial,
        backend=BackendSpec("cupy", "float64", False),
    ).delta_psi
    minus = solve_pr_biased_linearized_timedependent_reference(
        controlled_source,
        time_normalized=0.2,
        spec=PRBiasedLinearizedReferenceSpec(applied_field=-bias, **spec_values),
        initial_delta_psi=initial,
        backend=BackendSpec("cupy", "float64", False),
    ).delta_psi
    plus_hat = np.fft.fft2(np.asarray(plus.get() if hasattr(plus, "get") else plus))
    minus_hat = np.fft.fft2(np.asarray(minus.get() if hasattr(minus, "get") else minus))
    selected = (5, 5)
    reversal = _comparison(minus_hat[selected], np.conj(plus_hat[selected]))
    if not (plus_hat[selected].imag < 0.0 < minus_hat[selected].imag):
        raise AssertionError("bias reversal did not reverse modal phase direction")
    static = solve_pr_biased_linearized_reference(
        source[0],
        spec=PRBiasedLinearizedReferenceSpec(applied_field=bias, **spec_values),
        backend=BackendSpec("cupy", "float64", False),
    ).delta_psi
    long_td = solve_pr_biased_linearized_timedependent_reference(
        source[0],
        time_normalized=30.0,
        spec=PRBiasedLinearizedReferenceSpec(applied_field=bias, **spec_values),
        backend=BackendSpec("cupy", "float64", False),
    ).delta_psi
    static_host = np.asarray(static.get() if hasattr(static, "get") else static)
    long_host = np.asarray(long_td.get() if hasattr(long_td, "get") else long_td)
    static_comparison = _comparison(long_host, static_host)
    long_state = state_from_potential(
        long_td,
        dx_normalized=spec_values["dx_normalized"],
        dy_normalized=spec_values["dy_normalized"],
        applied_field_x=bias,
        xp=__import__("cupy"),
    )
    static_state = state_from_potential(
        static,
        dx_normalized=spec_values["dx_normalized"],
        dy_normalized=spec_values["dy_normalized"],
        applied_field_x=bias,
        xp=__import__("cupy"),
    )
    state_comparisons = {
        name: _comparison(
            getattr(long_state, name).get(), getattr(static_state, name).get()
        )
        for name in ("E_x", "E_y")
    }
    long_active = project_active_field(
        long_state.E_x,
        long_state.E_y,
        profile=PRTransverseProjectionProfile(),
        xp=__import__("cupy"),
    )
    static_active = project_active_field(
        static_state.E_x,
        static_state.E_y,
        profile=PRTransverseProjectionProfile(),
        xp=__import__("cupy"),
    )
    state_comparisons["E_active"] = _comparison(
        long_active.get(), static_active.get()
    )
    null_indices = ((0, 0), (0, source.shape[-1] // 2),
                    (source.shape[-2] // 2, 0),
                    (source.shape[-2] // 2, source.shape[-1] // 2))
    null_max = max(float(abs(plus_hat[index])) for index in null_indices)
    if null_max > 2.0e-12:
        raise AssertionError(f"null-mode contamination: {null_max}")
    return {
        "selected_fft_bin": list(selected),
        "positive_coefficient": [plus_hat[selected].real, plus_hat[selected].imag],
        "negative_coefficient": [minus_hat[selected].real, minus_hat[selected].imag],
        "bias_reversal": reversal,
        "static_limit": {"psi": static_comparison, **state_comparisons},
        "joint_null_mode_maximum": null_max,
    }


def _fast_package(output: Path, result, request) -> dict[str, object]:
    package = output / "case_b_fast"
    registry = default_transport_registry()
    run_id = os.environ.get("SLURM_JOB_ID", "local-smoke")
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
        raise AssertionError("Fast package retained a material volume")
    required = {"input_intensity", "output_intensity", "far_field_intensity"}
    if not required.issubset(set(run_data.fields)):
        raise AssertionError("Fast product conversion omitted an optical product")
    summary = run_data.diagnostics["summary"].values
    if summary["physics_profile"]["validation_status"] != "experimental":
        raise AssertionError("linearized TD product status is not Experimental")
    if run_data.geometry.x.shape != (request.grid.Nx,) or (
        run_data.geometry.y.shape != (request.grid.Ny,)
    ):
        raise AssertionError("Fast product geometry is inconsistent")
    decoded_diagnostics = json.dumps(
        decoded.result.diagnostics, sort_keys=True, default=_json_default
    )
    original_diagnostics = json.dumps(
        result.diagnostics, sort_keys=True, default=_json_default
    )
    if decoded_diagnostics != original_diagnostics:
        raise AssertionError("Fast transport did not preserve compact diagnostics")
    files = sorted(path for path in package.rglob("*") if path.is_file())
    return {
        "policy": decoded.result.retention_summary["policy"],
        "omitted_fields": decoded.result.retention_summary["omitted_fields"],
        "retained_optical_endpoints": ["A_initial", "A_final"],
        "product_fields": sorted(run_data.fields),
        "product_conversion_seconds": product_seconds,
        "package_bytes": sum(path.stat().st_size for path in files),
        "files": {str(path.relative_to(output)): _sha256(path) for path in files},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_sha = _git("rev-parse", "HEAD", cwd=args.source_root)
    source_porcelain = _git("status", "--porcelain", cwd=args.source_root)
    if source_sha != EXPECTED_SHA or source_porcelain:
        raise RuntimeError(
            f"unclean or wrong production source: {source_sha!r}, {source_porcelain!r}"
        )

    import cupy as cp

    cp.cuda.Device().use()
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before_pool = {"used_bytes": pool.used_bytes(), "total_bytes": pool.total_bytes()}
    case_a, _, _, _ = _case(args.size, 0.0)
    case_b, _, cupy_b, captured_source = _case(args.size, 0.35)
    modal = _modal_sanity(captured_source, cupy_b, bias=0.35)
    fast = _fast_package(
        args.output,
        cupy_b,
        _request(size=args.size, backend="cupy", bias=0.35),
    )
    cp.cuda.Stream.null.synchronize()
    after_pool = {"used_bytes": pool.used_bytes(), "total_bytes": pool.total_bytes()}

    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = device["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    if "H200" not in str(device_name).upper():
        raise RuntimeError(f"allocated GPU is not an H200: {device_name}")
    metrics = {
        "schema_version": 1,
        "production_git_sha": source_sha,
        "production_git_status_porcelain": source_porcelain,
        "source_clean": source_porcelain == "",
        "request": {
            "grid": [args.size, args.size],
            "Nz": 2,
            "accepted_intervals": 3,
            "dt_normalized": 0.2,
            "biases": [0.0, 0.35],
            "gain_length_product": 0.025,
            "reference_intensity": 1.5,
            "precision": "float64",
            "result_policy": "fast",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "gpu_name": device_name,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
        },
        "cases": {"unbiased": case_a, "biased": case_b},
        "modal_sanity": modal,
        "fast_package": fast,
        "cupy_memory_pool": {"before": before_pool, "after": after_pool},
        "timing_semantics": (
            "Per-backend measurements after explicit CUDA/CuPy initialization; "
            "synchronized around instrumented optical and material calls."
        ),
    }
    metrics_path = args.output / "commissioning_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    artifacts = sorted(path for path in args.output.rglob("*") if path.is_file())
    manifest = {
        "production_git_sha": source_sha,
        "production_git_status_porcelain": source_porcelain,
        "harness_sha256": _sha256(Path(__file__)),
        "artifacts": {
            str(path.relative_to(args.output)): _sha256(path) for path in artifacts
        },
    }
    (args.output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
