from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.runtime_estimator import (
    GIB,
    MIB,
    calibration_data,
    estimate_pr_resources,
    format_pr_resource_estimate,
)
from lcprop.pr.specs import (
    PRRunRequest,
    PRSolverOptions,
    PR_EXACT_MODAL_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.static_workflow import PRStaticRunRequest, PRStaticWorkflowOptions
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseMaterialResponseSpec,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
)


def _grid(nx=256, ny=256, nz=2):
    return GridSpec(
        Nx=nx,
        Ny=ny,
        x_aperture_um=64.0,
        y_aperture_um=64.0,
        dz_um=5.0,
        z_length_um=5.0 * nz,
    )


BEAMS = BeamStack(channels=(BeamChannel(
    wavelength_um=0.633,
    waist_x_um=20.0,
    waist_y_um=20.0,
    coherence_group="estimate",
),))
BACKEND = BackendSpec("numpy", "float64", False)


def _response(model):
    return PRTransverseMaterialResponseSpec(
        model=model,
        reference_intensity=(1.0 if model == PR_MATERIAL_RESPONSE_LINEARIZED else None),
    )


def _request(
    transport, evolution, response, *, nx=256, ny=256, nz=2, beams=BEAMS
):
    common = dict(
        grid=_grid(nx, ny, nz),
        beams=beams,
        backend=BACKEND,
        material_response=_response(response),
    )
    if transport == "reduced" and evolution == "static":
        return PRStaticRunRequest(
            **common,
            solver=PRStaticWorkflowOptions(max_coupled_passes=20),
        )
    if transport == "full" and evolution == "static":
        return PRTransverseStaticRunRequest(
            **common,
            solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=20),
        )
    if transport == "reduced":
        return PRRunRequest(
            **common,
            solver=PRSolverOptions(
                Nt=3,
                optical_substeps=1,
                integrator=(
                    PR_EXACT_MODAL_INTEGRATOR
                    if response == PR_MATERIAL_RESPONSE_LINEARIZED
                    else PR_SEMI_IMPLICIT_INTEGRATOR
                ),
            ),
        )
    return PRTransverseRunRequest(
        **common,
        solver=PRTransverseSolverOptions(Nt=3, optical_substeps=1),
    )


def _resolve_json_pointer(value, pointer):
    current = value
    for token in pointer.lstrip("/").split("/"):
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _resolve_measurement(root, measurement):
    source = root / measurement["source"]
    assert source.is_file(), source
    extraction = measurement["extract"]
    if extraction["kind"] == "json_pointer":
        content = json.loads(source.read_text(encoding="utf-8"))
        return _resolve_json_pointer(content, extraction["pointer"])
    if extraction["kind"] == "csv_column_max":
        with source.open(newline="", encoding="utf-8") as stream:
            return max(
                float(row[int(extraction["column_index"])])
                for row in csv.reader(stream)
                if row
            )
    raise AssertionError(f"unsupported extraction: {extraction}")


def test_every_measured_calibration_value_matches_its_exact_committed_source():
    data = calibration_data()
    assert data["schema_version"] == 1
    assert data["calibration_id"] == "pr-runtime-resource-v1"
    assert "queue" in data["timing_scope"]
    root = Path(__file__).resolve().parents[1]
    for source_name, expected_sha256 in data["evidence_sources"].items():
        source = root / source_name
        assert source.is_file(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256
    measured = 0
    for case in data["cases"].values():
        for entry in case.get("measurements", {}).values():
            entries = entry if isinstance(entry, list) else [entry]
            for measurement in entries:
                assert _resolve_measurement(root, measurement) == measurement["value"]
                measured += 1
    assert measured == 20

    projection = data["cases"]["full_transverse_nonlinear_static_memory"]
    source_text = (root / projection["source"]).read_text(encoding="utf-8")
    assert projection["source_excerpt"] in source_text
    assert set(data) >= {"cases", "derived_scaling", "planning_policy"}


@pytest.mark.parametrize("transport", ("reduced", "full"))
@pytest.mark.parametrize("evolution", ("static", "td"))
@pytest.mark.parametrize(
    "response", (PR_MATERIAL_RESPONSE_NONLINEAR, PR_MATERIAL_RESPONSE_LINEARIZED)
)
def test_all_eight_production_cells_have_bounded_resource_estimates(
    transport, evolution, response
):
    estimate = estimate_pr_resources(_request(transport, evolution, response))
    assert estimate.local_runtime is not None
    assert estimate.local_runtime.high >= estimate.local_runtime.low >= 0.0
    assert estimate.peak_gpu_memory.high >= estimate.peak_gpu_memory.low > 0.0
    assert estimate.peak_host_memory.high >= estimate.peak_host_memory.low > 0.0
    assert estimate.full_result_size.high > estimate.fast_result_size.low
    assert estimate.confidence in {"Low", "Medium", "High"}


def test_reduced_static_holdout_measurements_fall_within_estimated_ranges():
    grid = _grid(256, 256, 100)
    for response, measured in (
        (PR_MATERIAL_RESPONSE_NONLINEAR, 5.0429912919935305),
        (PR_MATERIAL_RESPONSE_LINEARIZED, 2.852051334019052),
    ):
        request = replace(_request("reduced", "static", response), grid=grid)
        runtime = estimate_pr_resources(request).local_runtime
        assert runtime is not None
        assert runtime.low <= measured <= runtime.high


def test_full_transverse_td_holdouts_and_component_scaling():
    linear = _request("full", "td", PR_MATERIAL_RESPONSE_LINEARIZED)
    nonlinear = _request("full", "td", PR_MATERIAL_RESPONSE_NONLINEAR)
    linear_estimate = estimate_pr_resources(linear)
    nonlinear_estimate = estimate_pr_resources(nonlinear)
    for estimate, local_measured, h200_measured in (
        (linear_estimate, 0.3038907089503482, 0.03357275703456253),
        (nonlinear_estimate, 0.34523770003579557, 0.42976761795580387),
    ):
        assert (
            estimate.local_runtime.low
            <= local_measured
            <= estimate.local_runtime.high
        )
        assert estimate.h200_runtime.low <= h200_measured <= estimate.h200_runtime.high

    doubled = replace(
        linear,
        solver=replace(linear.solver, optical_substeps=2),
    )
    doubled_estimate = estimate_pr_resources(doubled)
    assert doubled_estimate.local_runtime.low > linear_estimate.local_runtime.low
    assert doubled_estimate.h200_runtime.high > linear_estimate.h200_runtime.high

    cases = calibration_data()["cases"]
    for estimate, case_name in (
        (linear_estimate, "full_transverse_linearized_td_h200"),
        (nonlinear_estimate, "full_transverse_nonlinear_td_h200"),
    ):
        measurements = cases[case_name]["measurements"]
        peak = float(measurements["observed_peak_mib"]["value"]) * MIB
        package = float(measurements["fast_package_bytes"]["value"])
        assert estimate.peak_gpu_memory.low <= peak <= estimate.peak_gpu_memory.high
        assert estimate.fast_result_size.low <= package <= estimate.fast_result_size.high


def test_nonlinear_static_holdout_memory_projection_and_low_confidence():
    request = _request(
        "full", "static", PR_MATERIAL_RESPONSE_NONLINEAR,
        nx=1024, ny=1024, nz=400,
    )
    estimate = estimate_pr_resources(request)
    assert estimate.confidence == "Low"
    assert estimate.recommendation == "Slurm/H200 recommended"
    assert estimate.peak_gpu_memory.low == pytest.approx(1.5 * GIB)
    assert estimate.peak_gpu_memory.high == pytest.approx(4.0 * GIB)
    assert estimate.peak_host_memory.low == pytest.approx(25.0 * GIB)
    assert estimate.peak_host_memory.high == pytest.approx(45.0 * GIB)
    assert "Newton/PCG" in " ".join(estimate.qualifications)


@pytest.mark.parametrize(
    "response", (PR_MATERIAL_RESPONSE_NONLINEAR, PR_MATERIAL_RESPONSE_LINEARIZED)
)
def test_full_static_distinct_groups_use_direct_only_planning(response):
    beams = BeamStack(channels=(
        replace(BEAMS.channels[0], name="a", coherence_group="a"),
        replace(BEAMS.channels[0], name="b", coherence_group="b"),
    ))
    estimate = estimate_pr_resources(
        _request("full", "static", response, beams=beams)
    )
    text = format_pr_resource_estimate(estimate)
    assert "Configured direct stage" in text
    assert "up to 262 underlying optical propagations" in text
    assert "Visibility continuation: inapplicable" in text
    assert "Continuation-capable planning envelope" not in text


@pytest.mark.parametrize(
    "response", (PR_MATERIAL_RESPONSE_NONLINEAR, PR_MATERIAL_RESPONSE_LINEARIZED)
)
def test_full_static_shared_group_exposes_direct_and_continuation_scenarios(response):
    beams = BeamStack(channels=(
        replace(BEAMS.channels[0], name="a", coherence_group="shared"),
        replace(BEAMS.channels[0], name="b", coherence_group="shared"),
    ))
    direct = estimate_pr_resources(_request("full", "static", response))
    coherent = estimate_pr_resources(
        _request("full", "static", response, beams=beams)
    )
    text = format_pr_resource_estimate(coherent)
    assert "Configured direct stage" in text
    assert "Continuation-capable planning envelope" in text
    assert "up to 120 coupled material iterations" in text
    assert "2358 underlying optical propagations" in text
    assert "Activation is not predicted" in text
    assert coherent.local_runtime.high > direct.local_runtime.high
    assert coherent.h200_runtime.high > direct.h200_runtime.high


def test_nonlinear_static_host_memory_uses_coherence_appropriate_envelope():
    shared = BeamStack(channels=(
        replace(BEAMS.channels[0], name="a", coherence_group="shared"),
        replace(BEAMS.channels[0], name="b", coherence_group="shared"),
    ))
    direct_request = _request(
        "full", "static", PR_MATERIAL_RESPONSE_NONLINEAR,
        nx=1024, ny=1024, nz=400,
    )
    coherent_request = replace(direct_request, beams=shared)
    direct = estimate_pr_resources(direct_request)
    coherent = estimate_pr_resources(coherent_request)
    coherent32 = estimate_pr_resources(replace(
        coherent_request, backend=BackendSpec("numpy", "float32", False)
    ))
    assert direct.peak_host_memory.low == pytest.approx(25.0 * GIB)
    assert direct.peak_host_memory.high == pytest.approx(45.0 * GIB)
    assert coherent.peak_host_memory.low == pytest.approx(50.0 * GIB)
    assert coherent.peak_host_memory.high == pytest.approx(61.0 * GIB)
    assert coherent32.peak_host_memory.low == pytest.approx(25.0 * GIB)
    assert coherent32.peak_host_memory.high == pytest.approx(30.5 * GIB)


def test_estimator_does_not_mutate_the_scientific_request():
    shared = BeamStack(channels=(
        replace(BEAMS.channels[0], name="a", coherence_group="shared"),
        replace(BEAMS.channels[0], name="b", coherence_group="shared"),
    ))
    request = _request(
        "full", "static", PR_MATERIAL_RESPONSE_NONLINEAR, beams=shared
    )
    before = deepcopy(request)
    estimate_pr_resources(request)
    assert request == before


def test_precision_and_retention_formulas_are_monotone():
    request64 = _request("full", "td", PR_MATERIAL_RESPONSE_LINEARIZED, nz=100)
    request32 = replace(
        request64,
        backend=BackendSpec("numpy", "float32", False),
    )
    estimate64 = estimate_pr_resources(request64)
    estimate32 = estimate_pr_resources(request32)
    assert estimate32.peak_gpu_memory.high < estimate64.peak_gpu_memory.high
    assert estimate32.full_result_size.high < estimate64.full_result_size.high
    assert estimate64.fast_result_size.high < estimate64.full_result_size.high


def test_presentation_uses_ranges_and_explicit_qualifications():
    estimate = estimate_pr_resources(
        _request("full", "static", PR_MATERIAL_RESPONSE_NONLINEAR)
    )
    text = format_pr_resource_estimate(estimate)
    assert "Local Mac / NumPy proxy: ~" in text
    assert "H200/CuPy: ~" in text
    assert "Precision: float64" in text
    assert "Configured direct stage: up to 20 coupled material iterations" in text
    assert "up to 262 underlying optical propagations" in text
    assert "Visibility continuation: inapplicable" in text
    assert "Confidence: Low" in text
    assert "Dominant cost: nonlinear full-transverse material solve" in text
    assert "queue" in text.lower()
    assert "Recommendation: Slurm/H200 recommended" in text
