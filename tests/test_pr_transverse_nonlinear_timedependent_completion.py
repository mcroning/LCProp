from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lcprop.pr.transverse.workflow as workflow_module
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.linearized_timedependent_reference import (
    linearized_timedependent_rhs,
)
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
from lcprop.pr.transverse.static import solve_pr_transverse_static_intensity
from lcprop.pr.transverse.transport import potential_rhs, state_from_potential
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent


def _request(
    *,
    steps: int = 1,
    dt: float = 0.05,
    precision: str = "float64",
    linearized: bool = False,
) -> PRTransverseRunRequest:
    request = PRTransverseRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=8,
            x_aperture_um=4.0,
            y_aperture_um=4.0,
            dz_um=5.0,
            z_length_um=5.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=6.0,
            waist_y_um=6.0,
            coherence_group="nonlinear-td-completion",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.0,
            uniform_background_intensity=0.0,
            applied_field=0.0,
            gain_length_product=0.0,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=1.0,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=dt,
            optical_substeps=1,
        ),
        backend=BackendSpec("numpy", precision, False),
    )
    if not linearized:
        return request
    return replace(
        request,
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=0.0,
        ),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.0,
        ),
    )


def _source(modulation: float = 0.05, dtype=np.float64) -> np.ndarray:
    x = np.arange(8)[:, None]
    y = np.arange(8)[None, :]
    plane = (
        1.0
        + modulation * np.sin(2.0 * np.pi * x / 8.0)
        + 0.6 * modulation * np.cos(2.0 * np.pi * y / 8.0)
    )
    return plane[None, ...].astype(dtype)


def _fixed_source(monkeypatch, source: np.ndarray) -> None:
    def optical(A0, psi, **kwargs):
        return A0.copy(), kwargs["grid"].xp.asarray(
            source, dtype=kwargs["grid"].real_dtype
        )

    monkeypatch.setattr(workflow_module, "_optical_pass", optical)


@pytest.mark.parametrize(
    ("precision", "dtype", "gauge_tolerance"),
    (("float64", np.float64, 1.0e-16), ("float32", np.float32, 2.0e-8)),
)
def test_nonlinear_default_dispatch_reports_real_diagnostics_and_dtype(
    monkeypatch, precision, dtype, gauge_tolerance
):
    source = _source(dtype=dtype)
    _fixed_source(monkeypatch, source)

    result = run_pr_transverse_timedependent(
        _request(steps=2, precision=precision)
    )
    profile = result.resolved_profile
    final_rhs = potential_rhs(
        result.psi_final[0],
        source[0],
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )

    assert result.resolved_profile["material_response"]["model"] == (
        PR_MATERIAL_RESPONSE_NONLINEAR
    )
    assert result.psi_final.dtype == np.dtype(dtype)
    assert result.A_final.dtype == np.dtype(
        np.complex64 if dtype is np.float32 else np.complex128
    )
    assert result.diagnostics["material_response_calls"] == 2
    assert result.diagnostics["accepted_material_steps"] == 2
    assert result.diagnostics["optical_passes_completed"] == 3
    assert result.diagnostics["physical_state_valid"] is True
    assert result.diagnostics["carrier_density_minimum"] > 0.0
    assert result.diagnostics["nonlinear_td_rhs_max"] == pytest.approx(
        float(np.max(np.abs(final_rhs))), rel=2.0e-6, abs=1.0e-12
    )
    assert abs(float(np.mean(result.psi_final))) < gauge_tolerance
    assert "linearized_rhs_max" not in result.diagnostics
    assert not any(
        "newton" in name.lower() or "pcg" in name.lower()
        for name in result.diagnostics
    )


def test_nonlinear_td_approaches_static_oracle_for_frozen_source(monkeypatch):
    source = _source()
    _fixed_source(monkeypatch, source)
    request = _request(steps=400, dt=0.05)
    profile_run = run_pr_transverse_timedependent(request)
    profile = profile_run.resolved_profile
    static = solve_pr_transverse_static_intensity(
        source,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    td_state = state_from_potential(
        profile_run.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    static_state = state_from_potential(
        static.psi,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )

    assert static.converged
    assert np.linalg.norm(profile_run.psi_final - static.psi) / np.linalg.norm(
        static.psi
    ) < 3.0e-7
    for td_field, static_field in (
        (td_state.E_x, static_state.E_x),
        (td_state.E_y, static_state.E_y),
    ):
        np.testing.assert_allclose(td_field, static_field, rtol=0.0, atol=8.0e-9)
    td_active = project_active_field(
        td_state.E_x, td_state.E_y, profile=request.projection
    )
    static_active = project_active_field(
        static_state.E_x, static_state.E_y, profile=request.projection
    )
    np.testing.assert_allclose(td_active, static_active, rtol=0.0, atol=8.0e-9)
    assert profile_run.diagnostics["nonlinear_td_rhs_max"] < 1.0e-12
    assert max(abs(summary.equilibrium_max) for summary in static.plane_summaries) < (
        1.0e-8
    )


def test_weak_tangent_limit_and_strong_dispatch_divergence(monkeypatch):
    relative_differences = []
    for modulation in (2.0e-4, 1.0e-4):
        _fixed_source(monkeypatch, _source(modulation))
        nonlinear = run_pr_transverse_timedependent(_request(dt=1.0e-8))
        linearized = run_pr_transverse_timedependent(
            _request(dt=1.0e-8, linearized=True)
        )
        relative_differences.append(
            np.linalg.norm(nonlinear.psi_final - linearized.psi_final)
            / np.linalg.norm(linearized.psi_final)
        )
        np.testing.assert_array_equal(nonlinear.A_final, linearized.A_final)

    # This production comparison includes the IMEX-versus-exact-modal temporal
    # truncation floor.  The independent RHS Taylor oracle covers O(epsilon^2).
    assert max(relative_differences) < 2.0e-8
    assert relative_differences[1] <= 1.01 * relative_differences[0]

    _fixed_source(monkeypatch, _source(0.6))
    nonlinear = run_pr_transverse_timedependent(_request(steps=10))
    linearized = run_pr_transverse_timedependent(
        _request(steps=10, linearized=True)
    )
    relative_strong_difference = np.linalg.norm(
        nonlinear.psi_final - linearized.psi_final
    ) / np.linalg.norm(linearized.psi_final)
    assert relative_strong_difference > 0.05
    assert nonlinear.diagnostics["physical_state_valid"] is True


def test_production_imex_has_first_order_timestep_refinement(monkeypatch):
    _fixed_source(monkeypatch, _source(0.3))
    solutions = []
    for dt in (0.02, 0.01, 0.005):
        solutions.append(
            run_pr_transverse_timedependent(
                _request(steps=round(0.5 / dt), dt=dt)
            ).psi_final
        )
    ratio = np.linalg.norm(solutions[0] - solutions[1]) / np.linalg.norm(
        solutions[1] - solutions[2]
    )
    assert 1.8 < ratio < 2.2


def test_cancellation_during_material_candidate_discards_candidate(
    monkeypatch,
):
    source = _source()
    _fixed_source(monkeypatch, source)
    token = CancellationToken()
    original = workflow_module.imex_euler_step

    def cancel_after_candidate(*args, **kwargs):
        candidate = original(*args, **kwargs)
        token.cancel()
        return candidate

    monkeypatch.setattr(workflow_module, "imex_euler_step", cancel_after_candidate)
    result = run_pr_transverse_timedependent(
        _request(steps=3), cancellation_token=token
    )

    assert result.status == "cancelled"
    assert result.completed_steps == 0
    assert result.diagnostics["cancellation_observed_stage"] == (
        "after_material_candidate"
    )
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    np.testing.assert_array_equal(result.A_final, result.A_initial)


def test_unphysical_nonlinear_candidate_is_never_accepted(monkeypatch):
    source = _source()
    _fixed_source(monkeypatch, source)

    def invalid_candidate(psi, *args, **kwargs):
        candidate = np.zeros_like(psi)
        candidate[:, 0, 0] = 100.0
        return candidate

    monkeypatch.setattr(workflow_module, "imex_euler_step", invalid_candidate)
    with pytest.raises(FloatingPointError, match="nonpositive carrier density"):
        run_pr_transverse_timedependent(_request())


def test_nonlinear_bias_remains_explicitly_unsupported():
    request = _request(steps=1)
    with pytest.raises(ValueError, match="applied_field=0"):
        run_pr_transverse_timedependent(
            replace(
                request,
                material=replace(request.material, applied_field=0.2),
            )
        )


def test_commissioned_linearized_rhs_contract_is_unchanged(monkeypatch):
    source = _source()
    _fixed_source(monkeypatch, source)
    request = _request(steps=2, linearized=True)
    result = run_pr_transverse_timedependent(request)
    profile = result.resolved_profile
    rhs = linearized_timedependent_rhs(
        result.psi_final[0],
        source[0],
        spec=workflow_module._linearized_reference_spec(
            request,
            dx_normalized=profile["dx_normalized"],
            dy_normalized=profile["dy_normalized"],
        ),
        backend=request.backend,
    )

    assert result.diagnostics["source_cadence"] == (
        "one_complete_optical_pass_per_material_interval"
    )
    assert result.diagnostics["material_response_calls"] == 2
    assert result.diagnostics["linearized_rhs_max"] == float(np.max(np.abs(rhs)))
    assert "nonlinear_td_rhs_max" not in result.diagnostics
