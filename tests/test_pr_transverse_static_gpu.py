from __future__ import annotations

from dataclasses import replace
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.scattering import PR_CANONICAL_SCATTERING_V2, PRCanonicalScatteringSpec
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import PRTransverseProjectionProfile
from lcprop.pr.transverse.static import (
    PRTransverseStaticMaterialSolverOptions,
    _jacobian_action,
    _pcg,
    _symbols,
    _criteria_met as _material_criteria_met,
    project_production_resolved_modes,
    production_steady_jvp,
    production_steady_residual,
    solve_pr_transverse_discrete_static_intensity,
    solve_pr_transverse_static_intensity,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport import state_from_potential


_DIAGNOSTIC_OUTPUT_ENV = "LCPROP_PR_STATIC_DIAGNOSTIC_JSON"
_STAGE15R_STATE_DIR_ENV = "LCPROP_PR_STATIC_STAGE15R_STATE_DIR"


def _diagnostic_payload(*, request, result, material_results):
    """Return compact JSON-safe convergence telemetry for a focused GPU run."""

    profile = result.resolved_profile
    state = state_from_potential(
        result.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    zero_flux_summaries = [
        {
            "coupled_iteration": coupled_iteration,
            **asdict(summary),
        }
        for coupled_iteration, material in enumerate(material_results, start=1)
        for summary in material.plane_summaries
    ]
    material_newton_records = [
        {
            "coupled_iteration": coupled_iteration,
            **asdict(record),
        }
        for coupled_iteration, material in enumerate(material_results, start=1)
        for record in material.iteration_records
    ]
    nonconverged = [
        summary for summary in zero_flux_summaries if not summary["converged"]
    ]
    worst_equilibrium = max(
        zero_flux_summaries,
        key=lambda summary: summary["equilibrium_max"],
        default=None,
    )
    worst_td = max(
        zero_flux_summaries,
        key=lambda summary: summary["td_rhs_max"],
        default=None,
    )
    return {
        "schema": "lcprop.pr_transverse_static_zero_flux_gpu_diagnostic.v2",
        "request": asdict(request),
        "workflow": {
            "status": result.status,
            "converged": bool(result.converged),
            "termination_reason": result.diagnostics["termination_reason"],
            "completed_coupled_iterations": result.completed_coupled_iterations,
            "attempted_outer_iterations": len(result.iteration_records),
            "accepted_outer_iterations": sum(
                int(record.accepted) for record in result.iteration_records
            ),
            "coupled_backtracks": sum(
                record.backtracks for record in result.iteration_records
            ),
            "final_replay": result.replay_diagnostics,
        },
        "final_residuals": {
            "equilibrium_rms": result.diagnostics["equilibrium_residual_rms"],
            "equilibrium_max": result.diagnostics["equilibrium_residual_max"],
            "production_td_rhs_rms": result.diagnostics["td_rhs_residual_rms"],
            "production_td_rhs_max": result.diagnostics["td_rhs_residual_max"],
        },
        "outer_history": [asdict(record) for record in result.iteration_records],
        "material": {
            "solve_calls": len(material_results),
            "newton_iterations_attempted": len(material_newton_records),
            "newton_iterations_accepted": sum(
                int(record["accepted"]) for record in material_newton_records
            ),
            "pcg_iterations": sum(
                record["pcg_iterations"] for record in material_newton_records
            ),
            "backtracks": sum(
                record["backtracks"] for record in material_newton_records
            ),
            "per_plane_nonconvergence_count": len(nonconverged),
            "termination_reason_counts": {
                reason: sum(
                    int(summary["status"] == reason)
                    for summary in zero_flux_summaries
                )
                for reason in sorted(
                    {summary["status"] for summary in zero_flux_summaries}
                )
            },
            "zero_flux_plane_summaries": zero_flux_summaries,
            "newton_history": material_newton_records,
            "experimental_discrete_corrector_invoked": False,
            "worst_equilibrium_plane": worst_equilibrium,
            "worst_td_rhs_diagnostic_plane": worst_td,
        },
        "sanity": {
            "psi_finite": bool(np.all(np.isfinite(result.psi_final))),
            "source_intensity_finite": bool(
                np.all(np.isfinite(result.source_intensity_stack))
            ),
            "equilibrium_residual_finite": bool(
                np.all(np.isfinite(result.equilibrium_residual_stack))
            ),
            "td_rhs_residual_finite": bool(
                np.all(np.isfinite(result.td_rhs_residual_stack))
            ),
            "transport_intensity_min": float(np.min(result.source_intensity_stack)),
            "transport_intensity_max": float(np.max(result.source_intensity_stack)),
            "carrier_density_min": float(np.min(state.carrier_density)),
            "psi_abs_max": float(np.max(np.abs(result.psi_final))),
            "curl_rms": result.diagnostics["curl_rms"],
            "curl_max": result.diagnostics["curl_max"],
            "gauss_rms": result.diagnostics["gauss_rms"],
            "gauss_max": result.diagnostics["gauss_max"],
            "optical_power_relative_drift": result.diagnostics[
                "optical_power_relative_drift"
            ],
        },
        "timing": result.timing,
        "backend": result.backend_summary,
    }


def _write_diagnostic_if_requested(*, request, result, material_results):
    output = os.environ.get(_DIAGNOSTIC_OUTPUT_ENV)
    if not output:
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _diagnostic_payload(
                request=request,
                result=result,
                material_results=material_results,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _cupy_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        cp.cuda.Device().compute_capability
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")
    return cp


def _material_options(precision: str) -> PRTransverseStaticMaterialSolverOptions:
    if precision == "float64":
        return PRTransverseStaticMaterialSolverOptions()
    return PRTransverseStaticMaterialSolverOptions(
        pcg_relative_tolerance=2.0e-5,
        pcg_absolute_tolerance=1.0e-7,
        equilibrium_rms_tolerance=4.0e-6,
        equilibrium_max_tolerance=2.0e-5,
        td_rhs_rms_tolerance=3.0e-6,
        td_rhs_max_tolerance=2.0e-5,
    )


def _constructed_equilibrium(n: int):
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    psi = 0.018 * np.cos(2.0 * x) + 0.011 * np.sin(3.0 * y)
    state = state_from_potential(
        psi, dx_normalized=spacing, dy_normalized=spacing
    )
    intensity = np.exp(-state.psi) / state.carrier_density
    return spacing, psi.astype(np.float64), intensity.astype(np.float64)


def _nyquist_rich_equilibrium(n: int):
    spacing = 2.0 * math.pi / n
    x = 2.0 * math.pi * np.arange(n)[:, None] / n
    y = 2.0 * math.pi * np.arange(n)[None, :] / n
    k = n // 2 - 1
    raw = (
        np.cos(k * x)
        + 0.8 * np.sin((k - 1) * y)
        + 0.5 * np.cos((k - 2) * x - (k - 1) * y)
    )
    state = state_from_potential(
        raw, dx_normalized=spacing, dy_normalized=spacing
    )
    psi = 0.35 * raw / np.max(np.abs(state.carrier_density - 1.0))
    state = state_from_potential(
        psi, dx_normalized=spacing, dy_normalized=spacing
    )
    intensity = np.exp(-state.psi) / state.carrier_density
    return spacing, psi, intensity


def test_float32_material_residual_policy_is_narrow_and_dtype_specific():
    float32_options = _material_options("float32")
    float64_options = _material_options("float64")

    assert float32_options.equilibrium_rms_tolerance == 4.0e-6
    assert float32_options.equilibrium_max_tolerance == 2.0e-5
    assert float32_options.td_rhs_rms_tolerance == 3.0e-6
    assert float32_options.td_rhs_max_tolerance == 2.0e-5
    assert float64_options == PRTransverseStaticMaterialSolverOptions()

    diagnosed_floor = (3.0514846e-6, 1.1878913e-5)
    assert _material_criteria_met(*diagnosed_floor, float32_options)
    assert not _material_criteria_met(
        np.nextafter(4.0e-6, math.inf),
        diagnosed_floor[1],
        float32_options,
    )
    assert not _material_criteria_met(
        diagnosed_floor[0],
        np.nextafter(2.0e-5, math.inf),
        float32_options,
    )


def test_cupy_pcg_reports_direct_true_linear_residual_norm():
    cp = _cupy_device()
    shape = (18, 16)
    denominator, null_mask = _symbols(
        shape, dx_normalized=0.37, dy_normalized=0.43, h_y=1.0, xp=cp
    )
    ii = cp.arange(shape[0], dtype=cp.float32)[:, None]
    jj = cp.arange(shape[1], dtype=cp.float32)[None, :]
    weight = cp.exp(0.08 * cp.cos(0.4 * ii) + 0.05 * cp.sin(0.3 * jj))
    weight /= cp.mean(weight)
    rhs = project_production_resolved_modes(
        cp.sin(0.2 * ii) + 0.7 * cp.cos(0.17 * jj),
        dx_normalized=0.37,
        dy_normalized=0.43,
        xp=cp,
    )
    options = _material_options("float32")
    (
        solution,
        iterations,
        converged,
        status,
        tolerance,
        reported_residual_norm,
    ) = _pcg(
        rhs,
        weight=weight,
        denominator=denominator,
        null_mask=null_mask,
        options=options,
        relative_tolerance=options.pcg_relative_tolerance,
        absolute_tolerance=options.pcg_absolute_tolerance,
        xp=cp,
    )
    direct = rhs - _jacobian_action(
        solution,
        weight=weight,
        denominator=denominator,
        null_mask=null_mask,
        xp=cp,
    )
    direct_norm = float(cp.asnumpy(cp.linalg.norm(direct.ravel())))
    assert converged
    assert status == "converged"
    assert iterations > 0
    assert reported_residual_norm <= tolerance
    assert reported_residual_norm == pytest.approx(direct_norm, rel=0, abs=0)


@pytest.mark.parametrize(
    ("plane", "ordinary_prefix", "maximum_total_pcg"),
    (
        (52, (8, 5, 5), 25),
        (58, (8, 5, 5), 26),
        (59, (9, 6, 5), 27),
        (60, (9, 6, 5), 28),
        (61, (9, 5, 5), 27),
    ),
)
def test_cupy_stage15r_saved_state_uses_near_gate_forcing_and_converges(
    plane, ordinary_prefix, maximum_total_pcg
):
    state_dir = os.environ.get(_STAGE15R_STATE_DIR_ENV)
    if not state_dir:
        pytest.skip("Stage-15R saved-state directory was not requested")
    source = Path(state_dir) / f"plane_{plane}.npz"
    if not source.is_file():
        pytest.fail(f"missing Stage-15R saved state: {source}")
    cp = _cupy_device()
    saved = np.load(source)
    material = PRMaterialSpec(
        dark_intensity=0.01,
        gain_length_product=3.0,
        refractive_index=2.4,
    )
    spacing = material.characteristic_wavenumber_per_um * 1000.0 / 1024.0
    result = solve_pr_transverse_static_intensity(
        cp.asarray(saved["frozen_transport_intensity"]),
        initial_psi=cp.asarray(saved["warm_start_psi"]),
        dx_normalized=spacing,
        dy_normalized=spacing,
        options=_material_options("float32"),
        xp=cp,
    )
    summary = result.plane_summaries[0]
    assert result.converged and summary.converged
    assert summary.equilibrium_rms <= 4.0e-6
    assert summary.equilibrium_max <= 2.0e-5
    assert summary.physical_state_valid
    assert summary.carrier_minimum > 0.0
    assert summary.newton_iterations == 4
    pcg_iterations = tuple(
        record.pcg_iterations for record in result.iteration_records
    )
    assert pcg_iterations[:3] == ordinary_prefix
    assert pcg_iterations[-1] >= 5
    assert summary.pcg_iterations <= maximum_total_pcg
    assert summary.backtracks == 0
    assert all(
        record.pcg_final_residual_norm <= record.pcg_effective_tolerance
        for record in result.iteration_records
    )
    assert [record.pcg_forcing_tightened for record in result.iteration_records] == [
        False,
        False,
        False,
        True,
    ]


def test_cupy_stage15r_plane52_float64_matches_numpy_reference():
    state_dir = os.environ.get(_STAGE15R_STATE_DIR_ENV)
    if not state_dir:
        pytest.skip("Stage-15R saved-state directory was not requested")
    source = Path(state_dir) / "plane_52.npz"
    if not source.is_file():
        pytest.fail(f"missing Stage-15R saved state: {source}")
    cp = _cupy_device()
    saved = np.load(source)
    material = PRMaterialSpec(
        dark_intensity=0.01,
        gain_length_product=3.0,
        refractive_index=2.4,
    )
    spacing = material.characteristic_wavenumber_per_um * 1000.0 / 1024.0
    intensity = np.asarray(saved["frozen_transport_intensity"], dtype=np.float64)
    initial = np.asarray(saved["warm_start_psi"], dtype=np.float64)
    expected = solve_pr_transverse_static_intensity(
        intensity,
        initial_psi=initial,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    actual = solve_pr_transverse_static_intensity(
        cp.asarray(intensity),
        initial_psi=cp.asarray(initial),
        dx_normalized=spacing,
        dy_normalized=spacing,
        xp=cp,
    )
    assert expected.converged and actual.converged
    assert expected.plane_summaries[0].backtracks == 0
    assert actual.plane_summaries[0].backtracks == 0
    np.testing.assert_allclose(
        cp.asnumpy(actual.psi), expected.psi, rtol=2.0e-12, atol=2.0e-12
    )


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("kind", ("uniform", "constructed"))
def test_cupy_frozen_static_matches_numpy_reference(precision, kind):
    cp = _cupy_device()
    if kind == "uniform":
        spacing = 0.4
        intensity = np.full((2, 17, 15), 0.31, dtype=np.float64)
        initial = np.zeros_like(intensity)
    else:
        spacing, psi, plane = _constructed_equilibrium(18)
        intensity = np.stack((plane, np.roll(plane, 3, axis=1)))
        initial = np.zeros_like(intensity)

    expected = solve_pr_transverse_static_intensity(
        intensity,
        initial_psi=initial,
        dx_normalized=spacing,
        dy_normalized=spacing,
    )
    dtype = np.float32 if precision == "float32" else np.float64
    actual = solve_pr_transverse_static_intensity(
        cp.asarray(intensity, dtype=dtype),
        initial_psi=cp.asarray(initial, dtype=dtype),
        dx_normalized=spacing,
        dy_normalized=spacing,
        options=_material_options(precision),
        xp=cp,
    )

    assert expected.converged and actual.converged
    tolerance = 3.0e-5 if precision == "float32" else 8.0e-12
    for expected_array, actual_array in (
        (expected.psi, actual.psi),
        (expected.equilibrium_residual, actual.equilibrium_residual),
        (expected.td_rhs_residual, actual.td_rhs_residual),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual_array), expected_array, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_cupy_production_discrete_jvp_matches_numpy(precision):
    cp = _cupy_device()
    rng = np.random.default_rng(82028)
    shape = (16, 14)
    dx, dy = 0.37, 0.43
    dtype = np.float32 if precision == "float32" else np.float64
    psi = rng.normal(scale=2.0e-3, size=shape).astype(dtype)
    intensity = (0.2 + rng.random(shape)).astype(dtype)
    vector = rng.normal(size=shape).astype(dtype)
    expected = production_steady_jvp(
        psi,
        intensity,
        vector,
        dx_normalized=dx,
        dy_normalized=dy,
    )
    actual = production_steady_jvp(
        cp.asarray(psi),
        cp.asarray(intensity),
        cp.asarray(vector),
        dx_normalized=dx,
        dy_normalized=dy,
        xp=cp,
    )
    tolerance = 2.0e-5 if precision == "float32" else 2.0e-11
    np.testing.assert_allclose(
        cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_cupy_discrete_corrector_matches_numpy_continuum_mismatch(precision):
    cp = _cupy_device()
    spacing, _, intensity = _nyquist_rich_equilibrium(16)
    expected = solve_pr_transverse_discrete_static_intensity(
        intensity, dx_normalized=spacing, dy_normalized=spacing
    )
    dtype = np.float32 if precision == "float32" else np.float64
    actual = solve_pr_transverse_discrete_static_intensity(
        cp.asarray(intensity, dtype=dtype),
        dx_normalized=spacing,
        dy_normalized=spacing,
        options=_material_options(precision),
        xp=cp,
    )
    assert expected.converged and actual.converged
    assert actual.plane_summaries[0].initial_residual_rms > 1.0e-6
    assert (
        actual.plane_summaries[0].final_residual_rms
        <= _material_options(precision).td_rhs_rms_tolerance
    )
    tolerance = 5.0e-5 if precision == "float32" else 2.0e-11
    np.testing.assert_allclose(
        cp.asnumpy(actual.psi), expected.psi, rtol=tolerance, atol=tolerance
    )


def _workflow_request(*, backend: str, precision: str, scattering: bool):
    material_options = _material_options(precision)
    if precision == "float32":
        workflow_options = PRTransverseStaticWorkflowOptions(
            material_solver=material_options,
            equilibrium_rms_tolerance=5.0e-6,
            equilibrium_max_tolerance=3.0e-5,
            td_rhs_rms_tolerance=5.0e-6,
            td_rhs_max_tolerance=3.0e-5,
            replay_rtol=3.0e-5,
            replay_atol=3.0e-6,
        )
    else:
        workflow_options = PRTransverseStaticWorkflowOptions(
            material_solver=material_options
        )
    return PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=24,
            Ny=24,
            x_aperture_um=40.0,
            y_aperture_um=40.0,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=10.0,
            tilt_x_rad_per_um=0.0,
            tilt_y_rad_per_um=0.0,
            coherence_group="transverse-static-gpu",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=1.0e-3,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=workflow_options,
        backend=BackendSpec(backend, precision, False),
        scattering=(
            PRCanonicalScatteringSpec(
                epsilon=1.0e-8,
                transverse_correlation_um=2.0,
                realization_seed=9182,
                canonical_dz_um=5.0,
                algorithm_version=PR_CANONICAL_SCATTERING_V2,
            )
            if scattering
            else None
        ),
    )


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("scattering", (False, True))
def test_cupy_coupled_static_matches_numpy_reference(
    precision, scattering, monkeypatch
):
    expected_request = _workflow_request(
        backend="numpy", precision="float64", scattering=scattering
    )
    expected = run_pr_transverse_static(expected_request)
    assert expected.converged

    _cupy_device()
    actual_request = _workflow_request(
        backend="cupy", precision=precision, scattering=scattering
    )
    material_results = []
    import lcprop.pr.transverse.static_workflow as static_workflow_module

    original_material_solve = static_workflow_module.solve_pr_transverse_static_intensity

    def capture_material_solve(*args, **kwargs):
        material_result = original_material_solve(*args, **kwargs)
        material_results.append(material_result)
        return material_result

    monkeypatch.setattr(
        static_workflow_module,
        "solve_pr_transverse_static_intensity",
        capture_material_solve,
    )
    actual = run_pr_transverse_static(actual_request)

    _write_diagnostic_if_requested(
        request=actual_request,
        result=actual,
        material_results=material_results,
    )

    assert actual.converged
    assert material_results
    assert all(
        summary.converged
        and summary.equilibrium_rms
        <= actual_request.solver.material_solver.equilibrium_rms_tolerance
        and summary.equilibrium_max
        <= actual_request.solver.material_solver.equilibrium_max_tolerance
        and summary.physical_state_valid
        for material_result in material_results
        for summary in material_result.plane_summaries
    )
    assert actual.backend_summary["backend"] == "cupy"
    assert actual.backend_summary["is_gpu"] is True
    tolerance = 5.0e-5 if precision == "float32" else 1.2e-11
    for expected_array, actual_array in (
        (expected.psi_final, actual.psi_final),
        (expected.A_final, actual.A_final),
        (expected.source_intensity_stack, actual.source_intensity_stack),
        (expected.equilibrium_residual_stack, actual.equilibrium_residual_stack),
        (expected.td_rhs_residual_stack, actual.td_rhs_residual_stack),
    ):
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=tolerance, atol=tolerance
        )

    profile = actual.resolved_profile
    state = state_from_potential(
        actual.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    active = project_active_field(
        state.E_x,
        state.E_y,
        profile=PRTransverseProjectionProfile(**profile["projection"]),
    )
    expected_state = state_from_potential(
        expected.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    expected_active = project_active_field(
        expected_state.E_x,
        expected_state.E_y,
        profile=PRTransverseProjectionProfile(**profile["projection"]),
    )
    for expected_array, actual_array in (
        (expected_state.carrier_density, state.carrier_density),
        (expected_state.E_x, state.E_x),
        (expected_state.E_y, state.E_y),
        (expected_active, active),
    ):
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=tolerance, atol=tolerance
        )
    assert actual.diagnostics["optical_power_relative_drift"] == pytest.approx(
        expected.diagnostics["optical_power_relative_drift"],
        rel=tolerance,
        abs=tolerance,
    )
    for key in ("curl_rms", "curl_max", "gauss_rms", "gauss_max"):
        assert actual.diagnostics[key] == pytest.approx(
            expected.diagnostics[key], rel=tolerance, abs=tolerance
        )


@pytest.mark.parametrize("scattering", (False, True))
def test_numpy_coupled_static_reference_fixture_converges(scattering):
    result = run_pr_transverse_static(_workflow_request(
        backend="numpy", precision="float64", scattering=scattering
    ))
    assert result.converged
    assert result.status == "converged"
    assert result.completed_coupled_iterations == 2
    assert result.replay_diagnostics["field_match"] is True
    assert result.replay_diagnostics["source_match"] is True


def test_static_diagnostic_payload_serializes_numpy_reference():
    request = _workflow_request(
        backend="numpy", precision="float64", scattering=True
    )
    result = run_pr_transverse_static(request)
    payload = _diagnostic_payload(
        request=request,
        result=result,
        material_results=(),
    )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert '"termination_reason": "residual_tolerance"' in encoded
    assert payload["workflow"]["converged"] is True


def test_cupy_zero_background_accepts_positive_optical_transport_intensity():
    _cupy_device()
    request = _workflow_request(
        backend="cupy", precision="float32", scattering=False
    )
    request = replace(
        request,
        material=replace(
            request.material,
            dark_intensity=0.0,
            uniform_background_intensity=0.0,
        ),
    )
    result = run_pr_transverse_static(request)
    assert np.min(result.source_intensity_stack) > 0.0
    assert result.status in {"converged", "not_converged"}
