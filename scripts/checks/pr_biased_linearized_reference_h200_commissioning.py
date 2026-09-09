#!/usr/bin/env python3
"""Commission the isolated biased linearized PR reference on one H200."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter
import traceback

import numpy as np

from lcprop.core.backend import BackendSpec, get_backend
import lcprop.pr.transverse.linearized_reference as reference_module
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    biased_linearized_fourier_symbol,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.transport import potential_rhs


EXPECTED_SHA = "482acef47f677c9c23faffc4bc6aa6005ee06ee9"
FLOAT64_RTOL = 3.0e-13
FLOAT64_ATOL = 3.0e-13
FLOAT32_RTOL = 2.0e-5
FLOAT32_ATOL = 2.0e-6
WARM_REPEATS = 5


def backend_spec(name: str, precision: str):
    return BackendSpec(backend=name, precision=precision, verbose=False)


def host(value, cp):
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def comparison(candidate, expected, *, rtol: float, atol: float, cp):
    actual = host(candidate, cp)
    reference = host(expected, cp)
    difference = np.asarray(actual) - np.asarray(reference)
    reference_norm = float(np.linalg.norm(np.asarray(reference).ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    relative_l2 = difference_norm / reference_norm if reference_norm else difference_norm
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    passed = bool(np.allclose(actual, reference, rtol=rtol, atol=atol))
    if not passed:
        raise AssertionError(
            f"comparison failed: relative_l2={relative_l2}, max_abs={maximum}, "
            f"rtol={rtol}, atol={atol}"
        )
    return {
        "relative_l2": relative_l2,
        "max_absolute": maximum,
        "rtol": rtol,
        "atol": atol,
        "passed": passed,
    }


def multimode_fixture(dtype):
    nx, ny = 32, 35
    dx = 2.0 * math.pi / nx
    dy = 2.0 * math.pi / ny
    x = np.arange(nx)[:, None] * dx
    y = np.arange(ny)[None, :] * dy
    perturbation = (
        0.03
        + 0.02 * np.cos(2.0 * x)
        - 0.015 * np.sin(3.0 * y)
        + 0.01 * np.cos(2.0 * x - 3.0 * y)
    )
    intensity = (1.2 + perturbation).astype(dtype)
    spec = PRBiasedLinearizedReferenceSpec(1.2, 0.85, dx, dy, 1.3, 2.2)
    return intensity, spec


def compare_result(candidate, expected, *, rtol, atol, cp):
    fields = {}
    for name in (
        "response_kernel",
        "denominator",
        "delta_psi",
        "delta_E_x",
        "delta_E_y",
        "delta_P",
        "delta_mean_current",
        "mean_intensity_perturbation",
    ):
        fields[name] = comparison(
            getattr(candidate, name),
            getattr(expected, name),
            rtol=rtol,
            atol=atol,
            cp=cp,
        )
    return fields


def scientific_equivalence(cp):
    results = {}
    solved = {}
    for precision, real_dtype, complex_dtype, rtol, atol in (
        ("float64", np.float64, np.complex128, FLOAT64_RTOL, FLOAT64_ATOL),
        ("float32", np.float32, np.complex64, FLOAT32_RTOL, FLOAT32_ATOL),
    ):
        intensity, spec = multimode_fixture(real_dtype)
        numpy_result = solve_pr_biased_linearized_reference(
            intensity, spec=spec, backend=backend_spec("numpy", precision)
        )
        cupy_result = solve_pr_biased_linearized_reference(
            cp.asarray(intensity), spec=spec, backend=backend_spec("cupy", precision)
        )
        expected_real = np.dtype(real_dtype).name
        expected_complex = np.dtype(complex_dtype).name
        if cupy_result.delta_psi.dtype != getattr(cp, expected_real):
            raise AssertionError(f"unexpected CuPy real dtype: {cupy_result.delta_psi.dtype}")
        if cupy_result.response_kernel.dtype != getattr(cp, expected_complex):
            raise AssertionError(
                f"unexpected CuPy complex dtype: {cupy_result.response_kernel.dtype}"
            )
        mask_equal = np.array_equal(
            host(cupy_result.response_kernel == 0.0, cp),
            numpy_result.response_kernel == 0.0,
        )
        if not mask_equal:
            raise AssertionError("NumPy/CuPy zero-coefficient masks differ")
        gauge_mean = float(cp.mean(cupy_result.delta_psi).item())
        if abs(gauge_mean) > (2.0e-13 if precision == "float64" else 2.0e-6):
            raise AssertionError(f"nonzero reconstructed gauge mean: {gauge_mean}")
        provenance = {
            "requested_backend": cupy_result.requested_backend,
            "resolved_backend": cupy_result.resolved_backend,
            "real_dtype": cupy_result.real_dtype,
            "complex_dtype": cupy_result.complex_dtype,
            "is_gpu": cupy_result.is_gpu,
            "device_identity": cupy_result.device_identity,
            "model_id": cupy_result.model_id,
            "profile_id": cupy_result.profile_id,
            "applied_field": cupy_result.applied_field,
            "reference_intensity": cupy_result.reference_intensity,
        }
        if provenance["requested_backend"] != "cupy" or provenance["resolved_backend"] != "cupy":
            raise AssertionError(f"backend provenance mismatch: {provenance}")
        if provenance["real_dtype"] != expected_real or provenance["complex_dtype"] != expected_complex:
            raise AssertionError(f"dtype provenance mismatch: {provenance}")
        if not provenance["is_gpu"] or not str(provenance["device_identity"]).startswith("cuda:"):
            raise AssertionError(f"GPU provenance mismatch: {provenance}")
        results[precision] = {
            "fields": compare_result(
                cupy_result, numpy_result, rtol=rtol, atol=atol, cp=cp
            ),
            "mask_equal": mask_equal,
            "gauge_mean": gauge_mean,
            "numpy_dtypes": {
                "real": str(numpy_result.delta_psi.dtype),
                "complex": str(numpy_result.response_kernel.dtype),
            },
            "cupy_dtypes": {
                "real": str(cupy_result.delta_psi.dtype),
                "complex": str(cupy_result.response_kernel.dtype),
            },
            "provenance": provenance,
        }
        solved[precision] = (numpy_result, cupy_result)

    result32, _ = solved["float32"]
    result64, _ = solved["float64"]
    results["numpy_float32_vs_float64"] = compare_result(
        result32,
        result64,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
        cp=cp,
    )
    return results


def analytic_modes(cp):
    records = []
    cases = ((2, 0, 0.0), (0, 3, 0.0), (2, 0, 0.8), (2, 0, -0.8), (2, 3, 0.8))
    for precision, dtype, rtol, atol in (
        ("float64", np.float64, 3.0e-13, 8.0e-15),
        ("float32", np.float32, FLOAT32_RTOL, FLOAT32_ATOL),
    ):
        for mode_x, mode_y, applied_field in cases:
            nx, ny = 33, 35
            dx = 2.0 * math.pi / nx
            dy = 2.0 * math.pi / ny
            x = np.arange(nx)[:, None] * dx
            y = np.arange(ny)[None, :] * dy
            theta = mode_x * x + mode_y * y
            epsilon = 2.0e-4
            intensity = (1.3 + epsilon * np.cos(theta)).astype(dtype)
            spec = PRBiasedLinearizedReferenceSpec(1.3, applied_field, dx, dy, 1.4, 2.1)
            result = solve_pr_biased_linearized_reference(
                cp.asarray(intensity), spec=spec, backend=backend_spec("cupy", precision)
            )
            a_m = mode_x**2 + spec.m_y * mode_y**2
            a_h = mode_x**2 + spec.h_y * mode_y**2
            response = -(a_m + 1j * applied_field * mode_x) / (
                spec.reference_intensity
                * (a_m * (1.0 + a_h) + 1j * applied_field * mode_x * a_h)
            )
            expected_psi = epsilon * (
                response.real * np.cos(theta) - response.imag * np.sin(theta)
            )
            common = epsilon * (
                response.real * np.sin(theta) + response.imag * np.cos(theta)
            )
            expected_x = mode_x * common
            expected_y = mode_y * common
            records.append(
                {
                    "precision": precision,
                    "mode_x": mode_x,
                    "mode_y": mode_y,
                    "applied_field": applied_field,
                    "delta_psi": comparison(result.delta_psi, expected_psi, rtol=rtol, atol=atol, cp=cp),
                    "delta_E_x": comparison(result.delta_E_x, expected_x, rtol=rtol, atol=atol, cp=cp),
                    "delta_E_y": comparison(result.delta_E_y, expected_y, rtol=rtol, atol=atol, cp=cp),
                }
            )

    common_spec = dict(reference_intensity=1.2, dx_normalized=0.3, dy_normalized=0.4, m_y=1.6, h_y=2.2)
    plus = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=0.75, **common_spec),
        backend=backend_spec("cupy", "float64"),
    )
    minus = biased_linearized_fourier_symbol(
        (31, 29),
        spec=PRBiasedLinearizedReferenceSpec(applied_field=-0.75, **common_spec),
        backend=backend_spec("cupy", "float64"),
    )
    bias_reversal = bool(cp.array_equal(minus.response_kernel, plus.response_kernel.conj()).item())
    if not bias_reversal:
        raise AssertionError("CuPy bias reversal did not conjugate the symbol")
    unbiased_spec = PRBiasedLinearizedReferenceSpec(applied_field=0.0, **common_spec)
    unbiased = biased_linearized_fourier_symbol(
        (31, 29), spec=unbiased_spec, backend=backend_spec("cupy", "float64")
    )
    expected = cp.zeros_like(unbiased.response_kernel)
    expected[unbiased.resolved_mask] = -1.0 / (
        unbiased_spec.reference_intensity * (1.0 + unbiased.a_H[unbiased.resolved_mask])
    )
    unbiased_comparison = comparison(
        unbiased.response_kernel, expected, rtol=FLOAT64_RTOL, atol=FLOAT64_ATOL, cp=cp
    )
    return {
        "cases": records,
        "bias_reversal_exact": bias_reversal,
        "unbiased_limit": unbiased_comparison,
    }


def parity_and_batch(cp):
    parity = []
    spec = PRBiasedLinearizedReferenceSpec(1.1, 0.7, 0.2, 0.3, 1.4, 2.1)
    for shape in ((8, 10), (8, 9), (7, 10), (7, 9)):
        numpy_symbol = biased_linearized_fourier_symbol(
            shape, spec=spec, backend=backend_spec("numpy", "float64")
        )
        cupy_symbol = biased_linearized_fourier_symbol(
            shape, spec=spec, backend=backend_spec("cupy", "float64")
        )
        cupy_mask = host(cupy_symbol.resolved_mask, cp)
        mask_equal = bool(np.array_equal(cupy_mask, numpy_symbol.resolved_mask))
        zeros_equal = bool(
            np.array_equal(
                host(cupy_symbol.response_kernel == 0.0, cp),
                numpy_symbol.response_kernel == 0.0,
            )
        )
        expected_null = math.prod(2 if size % 2 == 0 else 1 for size in shape)
        actual_null = int(np.count_nonzero(~cupy_mask))
        if not mask_equal or not zeros_equal or actual_null != expected_null:
            raise AssertionError(f"null-mode parity mismatch for {shape}")
        ix = np.arange(shape[0])[:, None]
        iy = np.arange(shape[1])[None, :]
        intensity = (1.1 + 1.0e-3 * np.cos(2.0 * math.pi * (ix / shape[0] + iy / shape[1]))).astype(np.float64)
        result = solve_pr_biased_linearized_reference(
            cp.asarray(intensity), spec=spec, backend=backend_spec("cupy", "float64")
        )
        gauge_mean = float(cp.mean(result.delta_psi).item())
        if abs(gauge_mean) > 2.0e-13:
            raise AssertionError(f"gauge failure for {shape}: {gauge_mean}")
        parity.append(
            {
                "shape": list(shape),
                "expected_null_modes": expected_null,
                "actual_null_modes": actual_null,
                "mask_equal": mask_equal,
                "unresolved_zero_equal": zeros_equal,
                "gauge_mean": gauge_mean,
            }
        )

    base, batch_spec = multimode_fixture(np.float64)
    batch = np.stack((base, 1.2 - 0.6 * (base - 1.2))).astype(np.float64)
    numpy_batch = solve_pr_biased_linearized_reference(
        batch, spec=batch_spec, backend=backend_spec("numpy", "float64")
    )
    cupy_batch = solve_pr_biased_linearized_reference(
        cp.asarray(batch), spec=batch_spec, backend=backend_spec("cupy", "float64")
    )
    batch_fields = compare_result(
        cupy_batch,
        numpy_batch,
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
        cp=cp,
    )
    if cupy_batch.delta_psi.shape != batch.shape:
        raise AssertionError("batch shape was not preserved")
    return {
        "parity": parity,
        "batch": {
            "input_shape": list(batch.shape),
            "output_shape": list(cupy_batch.delta_psi.shape),
            "fft_axes": [-2, -1],
            "semantics": "independent frozen-intensity planes",
            "fields": batch_fields,
        },
    }


def operation_and_sync_audit(cp):
    intensity, spec = multimode_fixture(np.float64)
    counts = {"fft2": 0, "ifft2": 0}
    original_fft2 = cp.fft.fft2
    original_ifft2 = cp.fft.ifft2

    def counted_fft2(*args, **kwargs):
        counts["fft2"] += 1
        return original_fft2(*args, **kwargs)

    def counted_ifft2(*args, **kwargs):
        counts["ifft2"] += 1
        return original_ifft2(*args, **kwargs)

    cp.fft.fft2 = counted_fft2
    cp.fft.ifft2 = counted_ifft2
    try:
        result = solve_pr_biased_linearized_reference(
            cp.asarray(intensity), spec=spec, backend=backend_spec("cupy", "float64")
        )
        cp.cuda.Stream.null.synchronize()
    finally:
        cp.fft.fft2 = original_fft2
        cp.fft.ifft2 = original_ifft2
    if counts != {"fft2": 1, "ifft2": 4}:
        raise AssertionError(f"unexpected FFT counts: {counts}")
    if result.forward_fft_count != 1 or result.inverse_fft_count != 4:
        raise AssertionError("result operation provenance mismatch")

    original_asnumpy = reference_module.asnumpy
    sync_count = {"asnumpy_calls": 0}

    def counted_asnumpy(value):
        sync_count["asnumpy_calls"] += 1
        return original_asnumpy(value)

    reference_module.asnumpy = counted_asnumpy
    try:
        solve_pr_biased_linearized_reference(
            cp.asarray(intensity), spec=spec, backend=backend_spec("cupy", "float64")
        )
        cp.cuda.Stream.null.synchronize()
    finally:
        reference_module.asnumpy = original_asnumpy
    if sync_count["asnumpy_calls"] != 1:
        raise AssertionError(f"unexpected solver asnumpy count: {sync_count}")
    source = Path(reference_module.__file__).read_text()
    tokens = {
        "asnumpy": source.count("asnumpy("),
        ".item": source.count(".item("),
        ".get": source.count(".get("),
        "np.asarray": source.count("np.asarray("),
    }
    return {
        "instrumented_fft_calls": counts,
        "result_fft_provenance": {
            "forward": result.forward_fft_count,
            "inverse": result.inverse_fft_count,
        },
        "solver_validation_asnumpy_calls": sync_count["asnumpy_calls"],
        "source_token_audit": tokens,
        "iterative_solve": False,
        "timing_synchronizations_are_instrumentation": True,
    }


def clear_fft_cache(cp):
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass


def nvidia_process_memory():
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def benchmark(cp):
    records = []
    pool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    for precision, dtype in (("float64", cp.float64), ("float32", cp.float32)):
        for size in (256, 512, 1024):
            clear_fft_cache(cp)
            pool.free_all_blocks()
            pinned.free_all_blocks()
            x = cp.arange(size, dtype=dtype)[:, None]
            y = cp.arange(size, dtype=dtype)[None, :]
            intensity = (
                dtype(1.2)
                + dtype(0.02) * cp.cos(dtype(2.0 * math.pi / size) * (2 * x + 3 * y))
                + dtype(0.01) * cp.sin(dtype(2.0 * math.pi / size) * (5 * x - 2 * y))
            ).astype(dtype)
            del x, y
            cp.cuda.Stream.null.synchronize()
            free_before, total_device = cp.cuda.runtime.memGetInfo()
            pool_used_before = pool.used_bytes()
            pool_total_before = pool.total_bytes()
            spec = PRBiasedLinearizedReferenceSpec(
                1.2,
                0.85,
                2.0 * math.pi / size,
                2.0 * math.pi / size,
                1.3,
                2.2,
            )

            cp.cuda.Stream.null.synchronize()
            start = perf_counter()
            result = solve_pr_biased_linearized_reference(
                intensity, spec=spec, backend=backend_spec("cupy", precision)
            )
            cp.cuda.Stream.null.synchronize()
            first_seconds = perf_counter() - start
            warm = []
            for _ in range(WARM_REPEATS):
                cp.cuda.Stream.null.synchronize()
                start = perf_counter()
                result = solve_pr_biased_linearized_reference(
                    intensity, spec=spec, backend=backend_spec("cupy", precision)
                )
                cp.cuda.Stream.null.synchronize()
                warm.append(perf_counter() - start)
            free_after, _ = cp.cuda.runtime.memGetInfo()
            memory_sample = nvidia_process_memory()
            records.append(
                {
                    "precision": precision,
                    "grid": [size, size],
                    "first_call_seconds": first_seconds,
                    "warmed_seconds": warm,
                    "warmed_median_seconds": statistics.median(warm),
                    "repeats": WARM_REPEATS,
                    "timing_sync": "cupy.cuda.Stream.null.synchronize before and after",
                    "result_real_dtype": str(result.delta_psi.dtype),
                    "result_complex_dtype": str(result.response_kernel.dtype),
                    "pool_used_before_bytes": pool_used_before,
                    "pool_total_before_bytes": pool_total_before,
                    "pool_used_after_bytes": pool.used_bytes(),
                    "pool_total_after_bytes": pool.total_bytes(),
                    "pool_total_growth_bytes": pool.total_bytes() - pool_total_before,
                    "device_used_growth_bytes": free_before - free_after,
                    "device_total_bytes": total_device,
                    "nvidia_smi_process_memory": memory_sample,
                }
            )
            del result, intensity
            cp.cuda.Stream.null.synchronize()
    return records


def taylor_oracle(cp):
    nx = ny = 33
    spacing = 2.0 * math.pi / nx
    x = cp.arange(nx, dtype=cp.float64) * spacing
    X, Y = cp.meshgrid(x, x, indexing="ij")
    shape = cp.cos(2.0 * X + 3.0 * Y)
    spec = PRBiasedLinearizedReferenceSpec(1.3, 0.7, spacing, spacing, 1.4, 2.1)
    errors = []
    for epsilon in (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4):
        intensity = (spec.reference_intensity + epsilon * shape).astype(cp.float64)
        linearized = solve_pr_biased_linearized_reference(
            intensity, spec=spec, backend=backend_spec("cupy", "float64")
        )
        residual = potential_rhs(
            linearized.delta_psi,
            intensity,
            dx_normalized=spec.dx_normalized,
            dy_normalized=spec.dy_normalized,
            m_y=spec.m_y,
            h_y=spec.h_y,
            applied_field_x=spec.applied_field,
            xp=cp,
        )
        errors.append(float(cp.sqrt(cp.mean(residual * residual)).item()))
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    passed = all(1.98 < order < 2.02 for order in orders)
    if not passed:
        raise AssertionError(f"CuPy Taylor orders failed: {orders}")
    return {"status": "CuPy Taylor validated", "errors": errors, "orders": orders}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "classification": "Failed",
        "source_sha": args.source_sha,
        "expected_sha": EXPECTED_SHA,
        "tolerances": {
            "float64": {"rtol": FLOAT64_RTOL, "atol": FLOAT64_ATOL},
            "float32": {"rtol": FLOAT32_RTOL, "atol": FLOAT32_ATOL},
        },
    }
    try:
        if args.source_sha != EXPECTED_SHA:
            raise AssertionError(f"source SHA mismatch: {args.source_sha}")
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties.get("name")
        if isinstance(name, bytes):
            name = name.decode()
        backend = get_backend(backend_spec("cupy", "float64"))
        fft_support = {}
        for dtype in (cp.float32, cp.float64):
            transformed = cp.fft.fft2(cp.ones((8, 8), dtype=dtype))
            cp.cuda.Stream.null.synchronize()
            fft_support[str(dtype)] = str(transformed.dtype)
        metrics["environment"] = {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "device_id": device.id,
            "device_name": str(name),
            "backend_requested": "cupy",
            "backend_resolved": backend.name,
            "fft_support": fft_support,
        }
        if backend.name != "cupy" or not backend.is_gpu:
            raise AssertionError(f"backend failed to resolve to CuPy: {backend.summary()}")
        metrics["scientific_equivalence"] = scientific_equivalence(cp)
        metrics["analytic_modes"] = analytic_modes(cp)
        metrics["parity_and_batch"] = parity_and_batch(cp)
        metrics["operation_and_sync_audit"] = operation_and_sync_audit(cp)
        metrics["runtime_and_memory"] = benchmark(cp)
        metrics["taylor_oracle"] = taylor_oracle(cp)
        metrics["classification"] = "Commissioned"
        metrics["passed"] = True
    except Exception as error:
        metrics["passed"] = False
        metrics["error_type"] = type(error).__name__
        metrics["error"] = str(error)
        metrics["traceback"] = traceback.format_exc()
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        raise
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": metrics["classification"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
