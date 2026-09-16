"""Transparent planning estimates for PR runtime, memory, and result size.

The estimator is deliberately not a scheduler or convergence oracle.  It
combines a small versioned set of measured Product evidence with explicit
scaling formulas and returns ranges whose confidence records how directly the
selected request is covered by those measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import math
from typing import Any

from lcprop.core.grid import round_nz
from lcprop.pr.specs import PRRunRequest
from lcprop.pr.static_workflow import PRStaticRunRequest
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseRunRequest,
)
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunRequest
from lcprop.pr.visualization import FAST_MPR_TARGET_BYTES


CALIBRATION_RESOURCE = "runtime_estimator_calibration_v1.json"
MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class EstimateRange:
    """Inclusive planning range in one explicitly named unit."""

    low: float
    high: float
    unit: str

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.low)
            and math.isfinite(self.high)
            and 0.0 <= self.low <= self.high
        ):
            raise ValueError("estimate range must be finite, nonnegative, and ordered")


@dataclass(frozen=True)
class PRResourceEstimate:
    """Explainable planning result for one immutable PR request."""

    calibration_id: str
    model_cell: str
    grid_shape: tuple[int, int, int]
    precision: str
    material_intervals: int
    optical_passes: int
    optical_substeps: int
    work_summary: tuple[str, ...]
    local_runtime: EstimateRange | None
    h200_runtime: EstimateRange | None
    confidence: str
    dominant_cost: str
    peak_gpu_memory: EstimateRange
    peak_host_memory: EstimateRange
    fast_result_size: EstimateRange
    full_result_size: EstimateRange
    recommendation: str
    qualifications: tuple[str, ...]


@lru_cache(maxsize=1)
def calibration_data() -> dict[str, Any]:
    """Load and minimally validate the installed versioned calibration."""

    text = (
        resources.files("lcprop.pr.assets")
        .joinpath(CALIBRATION_RESOURCE)
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported PR runtime-estimator calibration schema")
    if not isinstance(value.get("calibration_id"), str):
        raise ValueError("PR runtime-estimator calibration lacks an identifier")
    cases = value.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("PR runtime-estimator calibration contains no cases")
    evidence_sources = value.get("evidence_sources")
    if not isinstance(evidence_sources, dict) or not evidence_sources:
        raise ValueError("PR runtime-estimator calibration lacks evidence checksums")
    for name, case in cases.items():
        if not isinstance(case, dict):
            raise ValueError(f"calibration case {name!r} must be an object")
        if "measurements" not in case and not isinstance(case.get("source"), str):
            raise ValueError(f"calibration case {name!r} lacks evidence provenance")
        for entry in case.get("measurements", {}).values():
            entries = entry if isinstance(entry, list) else [entry]
            for measurement in entries:
                if measurement.get("source") not in evidence_sources:
                    raise ValueError(
                        f"calibration case {name!r} uses unchecksummed evidence"
                    )
                if not isinstance(measurement.get("extract"), dict):
                    raise ValueError(
                        f"calibration case {name!r} lacks an extraction rule"
                    )
        if "source" in case and case["source"] not in evidence_sources:
            raise ValueError(f"calibration case {name!r} uses unchecksummed evidence")
    if not isinstance(value.get("derived_scaling"), dict):
        raise ValueError("PR runtime-estimator calibration lacks derived scaling")
    if not isinstance(value.get("planning_policy"), dict):
        raise ValueError("PR runtime-estimator calibration lacks planning policy")
    return value


def _measurement(reference: dict[str, Any], name: str) -> float:
    """Return one measured value while retaining provenance in the asset."""

    entry = reference["measurements"][name]
    if not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"calibration measurement {name!r} is malformed")
    return float(entry["value"])


def _measurement_range(reference: dict[str, Any], name: str) -> tuple[float, float]:
    entries = reference["measurements"][name]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError(f"calibration range {name!r} must have two measurements")
    values = tuple(float(entry["value"]) for entry in entries)
    return values[0], values[1]


def _planning_policy() -> dict[str, Any]:
    return calibration_data()["planning_policy"]


def _has_coherent_interference(request) -> bool:
    groups = request.beams.coherence_groups
    return len(set(groups)) < len(groups)


def _static_work_envelope(request) -> dict[str, int | bool]:
    """Return transparent direct and continuation-capable work envelopes."""

    iterations = int(request.solver.max_coupled_iterations)
    trials = int(request.solver.max_backtracks) + 1
    # Each stage has one initial optical evaluation, up to one evaluation per
    # line-search trial, and one final replay.  Invalid physical trials can
    # skip propagation, so this is a conservative planning envelope.
    direct_optical = 2 + iterations * trials
    continuation_capable = _has_coherent_interference(request)
    derived = calibration_data()["derived_scaling"]
    continuation_stages = int(derived["static_continuation_stage_count"])
    visibility_factors = tuple(
        int(value) for value in derived["static_continuation_visibility_factors"]
    )
    if len(visibility_factors) != continuation_stages:
        raise ValueError("static continuation calibration is internally inconsistent")
    stage_factor = 1 + sum(visibility_factors) if continuation_capable else 1
    stage_count = 1 + continuation_stages if continuation_capable else 1
    return {
        "continuation_capable": continuation_capable,
        "direct_material_iterations": iterations,
        "direct_optical_propagations": direct_optical,
        "planning_material_iterations": iterations * stage_count,
        "planning_optical_propagations": direct_optical * stage_factor,
        "planning_stage_count": stage_count,
    }


def _plane_fft_work(nx: int, ny: int, nz: int) -> float:
    points = float(nx * ny)
    return points * max(1.0, math.log2(points)) * float(nz)


def _grid(request) -> tuple[int, int, int]:
    grid = request.grid
    return int(grid.Nx), int(grid.Ny), round_nz(grid.z_length_um, grid.dz_um)


def _cell(request) -> tuple[str, bool, bool]:
    linearized = (
        request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
    )
    if isinstance(request, PRTransverseStaticRunRequest):
        return (
            "full_transverse_static_" + ("linearized" if linearized else "nonlinear"),
            True,
            True,
        )
    if isinstance(request, PRTransverseRunRequest):
        return (
            "full_transverse_td_" + ("linearized" if linearized else "nonlinear"),
            True,
            False,
        )
    if isinstance(request, PRStaticRunRequest):
        return (
            "reduced_x_static_" + ("linearized" if linearized else "nonlinear"),
            False,
            True,
        )
    if isinstance(request, PRRunRequest):
        return (
            "reduced_x_td_" + ("linearized" if linearized else "nonlinear"),
            False,
            False,
        )
    raise TypeError(f"unsupported PR estimator request: {type(request).__name__}")


def _run_counts(request, *, static: bool) -> tuple[int, int, int]:
    """Return material intervals, optical passes, and optical substeps."""

    if static:
        intervals = int(
            request.solver.max_coupled_iterations
            if isinstance(request, PRTransverseStaticRunRequest)
            else request.solver.max_coupled_passes
        )
    else:
        intervals = int(request.solver.Nt)
    return intervals, intervals + 1, int(request.solver.optical_substeps)


def _reference_scale(
    shape: tuple[int, int, int], reference: dict[str, Any]
) -> float:
    nx, ny, nz = shape
    rx, ry, rz = (int(value) for value in reference["grid"])
    return _plane_fft_work(nx, ny, nz) / _plane_fft_work(rx, ry, rz)


def _precision_factor(request) -> float:
    factor = float(calibration_data()["derived_scaling"]["float32_timing_factor"])
    return factor if request.backend.precision == "float32" else 1.0


def _component_projection(
    reference: dict[str, Any],
    *,
    shape: tuple[int, int, int],
    intervals: int,
    optical_passes: int,
    optical_substeps: int,
) -> tuple[float, float, float]:
    scale = _reference_scale(shape, reference)
    ref_intervals = max(1, int(reference["material_intervals"]))
    ref_passes = max(1, int(reference["optical_passes"]))
    material = _measurement(reference, "material_seconds") * scale * (
        intervals / ref_intervals
    )
    optical = _measurement(reference, "optical_seconds") * scale * (
        optical_passes * optical_substeps / ref_passes
    )
    residual = max(
        0.0,
        _measurement(reference, "scientific_total_seconds")
        - _measurement(reference, "material_seconds")
        - _measurement(reference, "optical_seconds"),
    )
    overhead_scale = max(1.0, scale ** 0.5)
    return material, optical, residual * overhead_scale


def _optical_projection(
    reference: dict[str, Any],
    *,
    shape: tuple[int, int, int],
    optical_propagations: int,
    optical_substeps: int,
) -> float:
    scale = _reference_scale(shape, reference)
    ref_passes = max(1, int(reference["optical_passes"]))
    return _measurement(reference, "optical_seconds") * scale * (
        optical_propagations * optical_substeps / ref_passes
    )


def _runtime_policy(name: str) -> tuple[float, float, float]:
    values = tuple(
        float(value)
        for value in _planning_policy()["runtime_range_factors"][name]
    )
    if len(values) == 2:
        return values[0], values[1], 0.0
    if len(values) == 3:
        return values
    raise ValueError(f"runtime planning policy {name!r} is malformed")


def _runtime_ranges(
    request,
    *,
    cell: str,
    shape: tuple[int, int, int],
    static: bool,
) -> tuple[EstimateRange | None, EstimateRange | None, str, str, tuple[str, ...]]:
    cases = calibration_data()["cases"]
    intervals, optical_passes, substeps = _run_counts(request, static=static)
    precision = _precision_factor(request)
    notes = [
        "Runtime ranges cover scientific computation, not queue, staging, "
        "or retrieval time."
    ]

    if cell.startswith("reduced_x_static_"):
        suffix = "linearized" if cell.endswith("linearized") else "nonlinear"
        ref = cases[f"reduced_static_numpy_{suffix}"]
        scale = _reference_scale(shape, ref) * (
            intervals / max(1, int(ref["coupled_passes"]))
        )
        measured_low, measured_high = _measurement_range(ref, "runtime_seconds")
        low_factor, high_factor, additive = _runtime_policy("reduced_static")
        local = EstimateRange(
            low_factor * measured_low * scale * precision,
            high_factor * measured_high * scale * precision + additive,
            "s",
        )
        notes.append(
            "No measured reduced-static H200 workflow calibration is retained."
        )
        return local, None, "Medium", "coupled optical/material passes", tuple(notes)

    if "linearized" in cell:
        cpu_ref = cases["full_transverse_linearized_td_numpy"]
        gpu_ref = cases["full_transverse_linearized_td_h200"]
        dominant = "fixed-count Fourier material and optical propagation"
    else:
        cpu_ref = cases["full_transverse_nonlinear_td_numpy"]
        gpu_ref = cases["full_transverse_nonlinear_td_h200"]
        dominant = "nonlinear full-transverse material solve"

    if cell.startswith("reduced_x_td_"):
        # Reduced material work is one-dimensional; optical propagation remains
        # two-dimensional.  This is intentionally a low-confidence projection.
        def reduced(ref):
            material, optical, overhead = _component_projection(
                ref,
                shape=shape,
                intervals=intervals,
                optical_passes=optical_passes,
                optical_substeps=substeps,
            )
            material_fraction = float(
                calibration_data()["derived_scaling"][
                    "reduced_td_material_fraction_of_full"
                ]
            )
            return material_fraction * material + optical + overhead

        local_center = reduced(cpu_ref) * precision
        h200_center = reduced(gpu_ref) * precision
        notes.append(
            "Reduced TD timing is derived from full-transverse component timings."
        )
        local_policy = _runtime_policy("reduced_td_local")
        h200_policy = _runtime_policy("reduced_td_h200")
        return (
            EstimateRange(
                local_policy[0] * local_center,
                local_policy[1] * local_center + local_policy[2],
                "s",
            ),
            EstimateRange(
                h200_policy[0] * h200_center,
                h200_policy[1] * h200_center + h200_policy[2],
                "s",
            ),
            "Low",
            "optical propagation",
            tuple(notes),
        )

    cpu_parts = _component_projection(
        cpu_ref,
        shape=shape,
        intervals=intervals,
        optical_passes=optical_passes,
        optical_substeps=substeps,
    )
    gpu_parts = _component_projection(
        gpu_ref,
        shape=shape,
        intervals=intervals,
        optical_passes=optical_passes,
        optical_substeps=substeps,
    )

    static_work = _static_work_envelope(request) if static else None

    if cell == "full_transverse_static_nonlinear":
        assert static_work is not None
        material_solver = request.solver.material_solver
        low_evaluations = 2.0
        high_evaluations = float(material_solver.max_newton_iterations) * (
            1.0 + math.sqrt(float(material_solver.max_pcg_iterations)) / 4.0
        )
        local_low = (low_evaluations * cpu_parts[0] + cpu_parts[1]) * precision
        stage_count = int(static_work["planning_stage_count"])
        optical_propagations = int(static_work["planning_optical_propagations"])
        local_high = (
            high_evaluations * cpu_parts[0] * stage_count
            + _optical_projection(
                cpu_ref,
                shape=shape,
                optical_propagations=optical_propagations,
                optical_substeps=substeps,
            )
            + cpu_parts[2] * stage_count
        ) * precision
        gpu_low = (low_evaluations * gpu_parts[0] + gpu_parts[1]) * precision
        gpu_high = (
            high_evaluations * gpu_parts[0] * stage_count
            + _optical_projection(
                gpu_ref,
                shape=shape,
                optical_propagations=optical_propagations,
                optical_substeps=substeps,
            )
            + gpu_parts[2] * stage_count
        ) * precision
        notes.append(
            "Nonlinear static bounds extrapolate commissioned TD kernels through "
            "the configured Newton/PCG limits; convergence strongly affects runtime."
        )
        if bool(static_work["continuation_capable"]):
            notes.append(
                "Shared coherence groups permit a direct failure followed by all "
                "five visibility-continuation stages; the upper planning envelope "
                "includes their material work, intermediate dual optical "
                "propagations, and configured line-search retries."
            )
        else:
            notes.append(
                "No coherent cross terms are present, so visibility continuation "
                "is inapplicable; the upper envelope still includes configured "
                "direct-stage line-search retries."
            )
        local_policy = _runtime_policy("full_static_nonlinear_local")
        h200_policy = _runtime_policy("full_static_nonlinear_h200")
        return (
            EstimateRange(
                local_policy[0] * local_low,
                local_policy[1] * local_high + local_policy[2],
                "s",
            ),
            EstimateRange(
                h200_policy[0] * gpu_low,
                h200_policy[1] * gpu_high + h200_policy[2],
                "s",
            ),
            "Low",
            dominant,
            tuple(notes),
        )

    if static:
        assert static_work is not None
        stage_count = int(static_work["planning_stage_count"])
        optical_propagations = int(static_work["planning_optical_propagations"])
        local_center = sum(cpu_parts) * precision
        gpu_center = sum(gpu_parts) * precision
        local_high_center = (
            cpu_parts[0] * stage_count
            + _optical_projection(
                cpu_ref,
                shape=shape,
                optical_propagations=optical_propagations,
                optical_substeps=substeps,
            )
            + cpu_parts[2] * stage_count
        ) * precision
        gpu_high_center = (
            gpu_parts[0] * stage_count
            + _optical_projection(
                gpu_ref,
                shape=shape,
                optical_propagations=optical_propagations,
                optical_substeps=substeps,
            )
            + gpu_parts[2] * stage_count
        ) * precision
        local_policy = _runtime_policy("full_static_linearized_local")
        h200_policy = _runtime_policy("full_static_linearized_h200")
        notes.append(
            "Static timing is projected from accepted TD component measurements; "
            + (
                "the upper envelope includes direct plus visibility-continuation "
                "stages and line-search optical retries."
                if bool(static_work["continuation_capable"])
                else "visibility continuation is inapplicable without coherent "
                "cross terms, while direct-stage line-search retries remain included."
            )
        )
        return (
            EstimateRange(
                local_policy[0] * local_center,
                local_policy[1] * local_high_center + local_policy[2],
                "s",
            ),
            EstimateRange(
                h200_policy[0] * gpu_center,
                h200_policy[1] * gpu_high_center + h200_policy[2],
                "s",
            ),
            "Low",
            dominant,
            tuple(notes),
        )

    local_center = sum(cpu_parts) * precision
    gpu_center = sum(gpu_parts) * precision
    confidence = "Medium" if not static else "Low"
    local_policy = _runtime_policy("full_td_local")
    h200_policy = _runtime_policy("full_td_h200")
    return (
        EstimateRange(
            local_policy[0] * local_center,
            local_policy[1] * local_center + local_policy[2],
            "s",
        ),
        EstimateRange(
            h200_policy[0] * gpu_center,
            h200_policy[1] * gpu_center + h200_policy[2],
            "s",
        ),
        confidence,
        dominant,
        tuple(notes),
    )


def _result_sizes(
    request, *, shape: tuple[int, int, int], static: bool, full_transverse: bool
) -> tuple[EstimateRange, EstimateRange]:
    nx, ny, nz = shape
    real_bytes = 4 if request.backend.precision == "float32" else 8
    complex_bytes = 2 * real_bytes
    channels = max(1, len(request.beams.channels))
    plane = nx * ny
    volume = nz * plane
    endpoints = 2 * channels * plane * complex_bytes
    optical_products = 3 * plane * real_bytes
    cuts = nz * (nx + ny) * real_bytes
    preview = min(
        FAST_MPR_TARGET_BYTES,
        nz * min(nx, 96) * min(ny, 96) * 4,
    )
    size_policy = _planning_policy()["result_size"]
    metadata_allowance = int(size_policy["metadata_allowance_bytes"])
    fast_raw = endpoints + optical_products + cuts + preview + metadata_allowance
    retained_volumes = 5 if (static and full_transverse) else 4 if static else 3
    full_raw = max(
        fast_raw - preview,
        endpoints
        + optical_products
        + retained_volumes * volume * real_bytes
        + metadata_allowance,
    )
    fast_factors = tuple(float(value) for value in size_policy["fast_range_factors"])
    full_factors = tuple(float(value) for value in size_policy["full_range_factors"])
    return (
        EstimateRange(fast_factors[0] * fast_raw, fast_factors[1] * fast_raw, "bytes"),
        EstimateRange(full_factors[0] * full_raw, full_factors[1] * full_raw, "bytes"),
    )


def _memory_ranges(
    request,
    *,
    cell: str,
    shape: tuple[int, int, int],
    full_result: EstimateRange,
    continuation_capable: bool,
) -> tuple[EstimateRange, EstimateRange]:
    nx, ny, nz = shape
    real_bytes = 4 if request.backend.precision == "float32" else 8
    plane_scale = (nx * ny * real_bytes) / (1024 * 1024 * 8)
    volume_bytes = nx * ny * nz * real_bytes

    if cell == "full_transverse_static_nonlinear":
        memory_ref = calibration_data()["cases"][
            "full_transverse_nonlinear_static_memory"
        ]
        ref_volume = 1024 * 1024 * 400 * 8
        volume_scale = volume_bytes / ref_volume
        projections = memory_ref["derived_projections"]
        gpu_projection = tuple(float(v) for v in projections["gpu_peak_gib"])
        host_key = (
            "host_continuation_overlap_gib"
            if continuation_capable
            else "host_direct_gib"
        )
        host_projection = tuple(float(v) for v in projections[host_key])
        memory_policy = _planning_policy()["memory"]
        full_floor = tuple(float(v) for v in memory_policy["full_gpu_floor_mib"])
        gpu = EstimateRange(
            max(
                full_floor[0] * MIB,
                gpu_projection[0] * GIB * plane_scale,
            ),
            max(
                full_floor[1] * MIB,
                gpu_projection[1] * GIB * plane_scale,
            ),
            "bytes",
        )
        host = EstimateRange(
            max(
                full_result.low,
                host_projection[0] * GIB * volume_scale,
            ),
            max(
                full_result.high,
                host_projection[1] * GIB * volume_scale,
            ),
            "bytes",
        )
        return gpu, host

    policy = _planning_policy()["memory"]
    full_floor = tuple(float(v) for v in policy["full_gpu_floor_mib"])
    reduced_floor = tuple(float(v) for v in policy["reduced_gpu_floor_mib"])
    full_plane = tuple(float(v) for v in policy["full_plane_multipliers"])
    full_volume = tuple(float(v) for v in policy["full_volume_multipliers"])
    reduced_plane = tuple(float(v) for v in policy["reduced_plane_multipliers"])
    host_working_multipliers = tuple(
        float(v) for v in policy["host_working_volume_multipliers"]
    )
    plane_bytes = nx * ny * real_bytes
    if cell.startswith("full_transverse"):
        gpu_low = full_floor[0] * MIB + full_plane[0] * plane_bytes + full_volume[0] * volume_bytes
        gpu_high = full_floor[1] * MIB + full_plane[1] * plane_bytes + full_volume[1] * volume_bytes
        host_working = 2 * volume_bytes
    else:
        gpu_low = reduced_floor[0] * MIB + reduced_plane[0] * plane_bytes
        gpu_high = reduced_floor[1] * MIB + reduced_plane[1] * plane_bytes + volume_bytes
        host_working = volume_bytes
    host = EstimateRange(
        full_result.low + host_working_multipliers[0] * host_working,
        float(policy["host_high_result_factor"]) * full_result.high
        + host_working_multipliers[1] * host_working,
        "bytes",
    )
    return EstimateRange(gpu_low, gpu_high, "bytes"), host


def estimate_pr_resources(request) -> PRResourceEstimate:
    """Estimate one request without executing scientific kernels."""

    request.grid.validate()
    request.backend.validate()
    request.solver.validate()
    request.material_response.validate()
    cell, full_transverse, static = _cell(request)
    shape = _grid(request)
    static_work = (
        _static_work_envelope(request)
        if isinstance(request, PRTransverseStaticRunRequest)
        else None
    )
    local, h200, confidence, dominant, notes = _runtime_ranges(
        request, cell=cell, shape=shape, static=static
    )
    fast_size, full_size = _result_sizes(
        request,
        shape=shape,
        static=static,
        full_transverse=full_transverse,
    )
    gpu_memory, host_memory = _memory_ranges(
        request,
        cell=cell,
        shape=shape,
        full_result=full_size,
        continuation_capable=bool(
            static_work is not None and static_work["continuation_capable"]
        ),
    )
    intervals, optical_passes, optical_substeps = _run_counts(
        request, static=static
    )

    if static_work is None:
        work_summary = (
            f"Configured work: {intervals} material intervals/passes; "
            f"{optical_passes} optical passes; optical substeps per pass: "
            f"{optical_substeps}",
        )
    else:
        work_summary = (
            "Configured direct stage: up to "
            f"{static_work['direct_material_iterations']} coupled material "
            "iterations; up to "
            f"{static_work['direct_optical_propagations']} underlying optical "
            "propagations including line-search trials and final replay; "
            f"{optical_substeps} optical substeps per propagation",
            (
                "Continuation-capable planning envelope: direct attempt plus five "
                "visibility stages; up to "
                f"{static_work['planning_material_iterations']} coupled material "
                "iterations and "
                f"{static_work['planning_optical_propagations']} underlying "
                "optical propagations. Activation is not predicted."
                if bool(static_work["continuation_capable"])
                else "Visibility continuation: inapplicable because the request "
                "has no shared coherence group."
            ),
        )

    recommendation_policy = _planning_policy()["recommendation_threshold_seconds"]
    local_high = local.high if local is not None else math.inf
    if (
        cell == "full_transverse_static_nonlinear"
        or local_high >= float(recommendation_policy["recommend_h200"])
    ):
        recommendation = "Slurm/H200 recommended"
    elif h200 is None:
        recommendation = "Local is reasonable; H200 timing is not calibrated"
    elif local_high >= float(recommendation_policy["consider_h200"]):
        recommendation = "Consider Slurm/H200"
    else:
        recommendation = "Local is reasonable"

    return PRResourceEstimate(
        calibration_id=str(calibration_data()["calibration_id"]),
        model_cell=cell,
        grid_shape=shape,
        precision=str(request.backend.precision),
        material_intervals=intervals,
        optical_passes=optical_passes,
        optical_substeps=optical_substeps,
        work_summary=work_summary,
        local_runtime=local,
        h200_runtime=h200,
        confidence=confidence,
        dominant_cost=dominant,
        peak_gpu_memory=gpu_memory,
        peak_host_memory=host_memory,
        fast_result_size=fast_size,
        full_result_size=full_size,
        recommendation=recommendation,
        qualifications=notes,
    )


def _format_seconds(value: float) -> str:
    if value < 1.0:
        return f"{value:.2g} s"
    if value < 90.0:
        return f"{value:.0f} s"
    if value < 90.0 * 60.0:
        minutes = value / 60.0
        return f"{minutes:.1f} min" if minutes < 10.0 else f"{minutes:.0f} min"
    hours = value / 3600.0
    return f"{hours:.1f} h" if hours < 10.0 else f"{hours:.0f} h"


def _format_bytes(value: float) -> str:
    if value >= GIB:
        gib = value / GIB
        return f"{gib:.1f} GiB" if gib < 10.0 else f"{gib:.0f} GiB"
    mib = value / MIB
    return f"{mib:.1f} MiB" if mib < 10.0 else f"{mib:.0f} MiB"


def _format_range(value: EstimateRange | None) -> str:
    if value is None:
        return "Insufficient calibration"
    formatter = _format_seconds if value.unit == "s" else _format_bytes
    return f"~{formatter(value.low)}–{formatter(value.high)}"


def format_pr_resource_estimate(estimate: PRResourceEstimate) -> str:
    """Return stable, concise GUI text without false numerical precision."""

    nx, ny, nz = estimate.grid_shape
    model_labels = {
        "reduced_x_static_linearized": "Reduced x-only static — linearized",
        "reduced_x_static_nonlinear": "Reduced x-only static — fully nonlinear",
        "reduced_x_td_linearized": "Reduced x-only TD — linearized",
        "reduced_x_td_nonlinear": "Reduced x-only TD — fully nonlinear",
        "full_transverse_static_linearized": "Full-transverse static — linearized",
        "full_transverse_static_nonlinear": (
            "Full-transverse static — fully nonlinear"
        ),
        "full_transverse_td_linearized": "Full-transverse TD — linearized",
        "full_transverse_td_nonlinear": "Full-transverse TD — fully nonlinear",
    }
    lines = [
        f"Model: {model_labels[estimate.model_cell]}",
        f"Grid: {nx} × {ny} × {nz}",
        f"Precision: {estimate.precision}",
        f"Local Mac / NumPy proxy: {_format_range(estimate.local_runtime)}",
        f"H200/CuPy: {_format_range(estimate.h200_runtime)}",
        f"Confidence: {estimate.confidence}",
        f"Dominant cost: {estimate.dominant_cost}",
        f"Estimated peak GPU memory: {_format_range(estimate.peak_gpu_memory)}",
        f"Estimated peak host memory: {_format_range(estimate.peak_host_memory)}",
        f"Estimated Fast result: {_format_range(estimate.fast_result_size)}",
        f"Estimated Full result: {_format_range(estimate.full_result_size)}",
        f"Recommendation: {estimate.recommendation}",
        f"Calibration: {estimate.calibration_id}",
    ]
    lines[3:3] = list(estimate.work_summary)
    lines.extend(f"Note: {note}" for note in estimate.qualifications)
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_RESOURCE",
    "EstimateRange",
    "PRResourceEstimate",
    "calibration_data",
    "estimate_pr_resources",
    "format_pr_resource_estimate",
]
