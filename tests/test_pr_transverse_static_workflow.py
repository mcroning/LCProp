from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PR_TRANSVERSE_STATIC_WORKFLOW,
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    pr_transverse_static_result_to_run_data,
    run_pr_transverse_static,
)
from lcprop.runners.local import LocalRunner


def _request(
    *,
    gain_length_product: float = 1.0e-3,
    max_coupled_iterations: int = 5,
    scattering=None,
):
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
            coherence_group="transverse-static",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=gain_length_product,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=max_coupled_iterations,
        ),
        backend=BackendSpec(
            backend="numpy", precision="float64", verbose=False
        ),
        scattering=scattering,
    )


def _two_beam_request(*, coherent: bool):
    groups = ("shared", "shared") if coherent else ("beam-a", "beam-b")
    request = _request(gain_length_product=1.0e-3)
    return replace(
        request,
        beams=BeamStack(channels=(
            replace(
                request.beams.channels[0],
                name="beam-a",
                x0_um=-3.0,
                tilt_x_rad_per_um=0.15,
                coherence_group=groups[0],
            ),
            replace(
                request.beams.channels[0],
                name="beam-b",
                x0_um=3.0,
                tilt_x_rad_per_um=-0.15,
                coherence_group=groups[1],
            ),
        )),
    )


def test_coupled_static_converges_with_refreshed_source_and_independent_replay():
    result = run_pr_transverse_static(_request())
    assert result.status == "converged"
    assert result.converged
    assert result.completed_coupled_iterations >= 1
    assert result.replay_diagnostics["complete_independent_replay"] is True
    assert result.replay_diagnostics["field_match"] is True
    assert result.replay_diagnostics["source_match"] is True
    assert result.diagnostics["equilibrium_residual_rms"] < 1e-8
    assert result.diagnostics["equilibrium_residual_max"] < 1e-7
    assert result.diagnostics["td_rhs_residual_rms"] < 1e-8
    assert result.diagnostics["td_rhs_residual_max"] < 1e-7
    assert result.diagnostics["discrete_corrector"][
        "authoritative_convergence"
    ] == "zero_flux_equilibrium_rms_and_max"
    assert result.diagnostics["discrete_corrector"]["invoked"] is False
    assert result.diagnostics["td_rhs_residual_role"] == "diagnostic_only"
    assert np.min(result.source_intensity_stack) >= 0.5
    assert result.timing["material_solve_seconds"] > 0.0
    assert result.timing["continuum_initializer_seconds"] > 0.0
    assert result.timing["discrete_corrector_seconds"] == 0.0
    assert result.timing["optical_pass_seconds"] > 0.0
    assert all(record.material_pcg_iterations >= 0 for record in result.iteration_records)
    assert result.material_iteration_records
    assert isinstance(result.discrete_iteration_records, tuple)
    assert result.discrete_iteration_records == ()
    assert all(
        record.newton_record.plane_index in (0, 1)
        for record in result.material_iteration_records
    )


def test_outer_convergence_is_authoritative_on_refreshed_zero_flux_residual():
    request = _request()
    request = replace(
        request,
        solver=replace(
            request.solver,
            td_rhs_rms_tolerance=1.0e-30,
            td_rhs_max_tolerance=1.0e-30,
        ),
    )
    result = run_pr_transverse_static(request)
    assert result.converged
    assert result.diagnostics["equilibrium_residual_rms"] <= (
        request.solver.equilibrium_rms_tolerance
    )
    assert result.diagnostics["equilibrium_residual_max"] <= (
        request.solver.equilibrium_max_tolerance
    )
    assert result.diagnostics["td_rhs_residual_rms"] > (
        request.solver.td_rhs_rms_tolerance
    )


def test_each_accepted_correction_uses_a_refreshed_midpoint_source(monkeypatch):
    import lcprop.pr.transverse.static_workflow as module

    original = module._optical_pass
    calls = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(np.asarray(result[1]).copy())
        return result

    monkeypatch.setattr(module, "_optical_pass", counted)
    result = run_pr_transverse_static(_request())
    accepted = sum(record.accepted for record in result.iteration_records)
    assert result.converged
    # Initial pass + every accepted trial + independent replay.
    assert len(calls) >= accepted + 2
    np.testing.assert_array_equal(calls[-1], result.source_intensity_stack)


def test_canonical_scattering_is_deterministic_and_phase_only():
    scattering = PRCanonicalScatteringSpec(
        epsilon=1.0e-8,
        transverse_correlation_um=2.0,
        realization_seed=9182,
        canonical_dz_um=5.0,
        algorithm_version=PR_CANONICAL_SCATTERING_V2,
    )
    request = _request(scattering=scattering)
    first = run_pr_transverse_static(request)
    second = run_pr_transverse_static(request)
    assert first.converged and second.converged
    np.testing.assert_array_equal(first.psi_final, second.psi_final)
    np.testing.assert_array_equal(first.A_final, second.A_final)
    assert abs(first.diagnostics["optical_power_relative_drift"]) < 2e-14
    assert "canonical_scattering" in first.diagnostics


def test_cancellation_before_work_retains_last_complete_accepted_state():
    token = CancellationToken()
    token.cancel()
    result = run_pr_transverse_static(_request(), cancellation_token=token)
    assert result.status == "cancelled"
    assert not result.converged
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.replay_diagnostics["field_match"] is True
    assert result.replay_diagnostics["source_match"] is True


def test_cancellation_during_material_trial_discards_partial_candidate(monkeypatch):
    import lcprop.pr.transverse.static_workflow as module

    token = CancellationToken()
    original = module.solve_pr_transverse_static_intensity

    def cancelling_solve(*args, **kwargs):
        result = original(*args, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(
        module, "solve_pr_transverse_static_intensity", cancelling_solve
    )
    result = run_pr_transverse_static(_request(), cancellation_token=token)
    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert not any(record.accepted for record in result.iteration_records)


def test_canonical_workflow_never_invokes_experimental_discrete_corrector(
    monkeypatch,
):
    import lcprop.pr.transverse.static as static_module

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical workflow invoked discrete corrector")

    monkeypatch.setattr(
        static_module, "solve_pr_transverse_discrete_static_intensity", forbidden
    )
    result = run_pr_transverse_static(_request())
    assert result.converged
    assert result.discrete_iteration_records == ()


def test_nonconvergence_is_distinct_from_success():
    result = run_pr_transverse_static(
        _request(gain_length_product=0.2, max_coupled_iterations=1)
    )
    assert result.status == "not_converged"
    assert not result.converged
    assert result.diagnostics["termination_reason"] != "residual_tolerance"
    assert result.replay_diagnostics["complete_independent_replay"] is True


def test_static_products_and_material_neutral_operation_reconstruct_state():
    request = _request()
    direct = run_pr_transverse_static(request)
    run_data = pr_transverse_static_result_to_run_data(direct)
    assert run_data.workflow == PR_TRANSVERSE_STATIC_WORKFLOW
    for key in (
        "psi",
        "P",
        "E_x",
        "E_y",
        "E_active",
        "transport_intensity",
        "equilibrium_residual",
        "td_rhs_residual",
    ):
        assert key in run_data.fields
    assert PR_TRANSVERSE_STATIC_OPERATION.workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW
    runner = LocalRunner(operations=(PR_TRANSVERSE_STATIC_OPERATION,))
    composed = runner.run_registered("pr", PR_TRANSVERSE_STATIC_WORKFLOW, request)
    assert composed.result.converged
    assert composed.run_data.workflow == PR_TRANSVERSE_STATIC_WORKFLOW


def test_static_workflow_rejects_auto_and_accepts_optical_only_intensity():
    with pytest.raises(ValueError, match="explicit"):
        run_pr_transverse_static(replace(
            _request(),
            backend=BackendSpec(backend="auto", precision="float64", verbose=False),
        ))
    with pytest.raises(ValueError, match="requires precision='float64'"):
        run_pr_transverse_static(replace(
            _request(),
            backend=BackendSpec(
                backend="numpy", precision="float32", verbose=False
            ),
        ))
    zero_background = replace(
        _request(),
        material=replace(
            _request().material,
            dark_intensity=0.0,
            uniform_background_intensity=0.0,
        ),
    )
    result = run_pr_transverse_static(zero_background)
    assert np.min(result.source_intensity_stack) > 0.0
    assert result.status in {"converged", "not_converged"}


def test_visibility_endpoints_are_exact_coherent_and_incoherent_sources(
    monkeypatch,
):
    import lcprop.pr.transverse.static_workflow as module

    request = _two_beam_request(coherent=True)
    A = np.ones((2, 3, 4), dtype=np.complex128)
    psi = np.zeros((1, 3, 4), dtype=np.float64)
    coherent_source = np.full((1, 3, 4), 7.0)
    incoherent_source = np.full((1, 3, 4), 3.0)

    def fake_pass(A0, state, *, request, **kwargs):
        source = (
            coherent_source
            if len(set(request.beams.coherence_groups)) == 1
            else incoherent_source
        )
        return A0.copy(), source.copy()

    monkeypatch.setattr(module, "_optical_pass", fake_pass)
    common = {
        "request": request,
        "grid": object(),
        "kernel": object(),
        "peak_reference": 1.0,
        "wavelength_um": 0.633,
        "dx_normalized": 1.0,
        "dy_normalized": 1.0,
    }
    _, zero = module._optical_pass_at_visibility(
        A, psi, visibility=0.0, **common
    )
    _, one = module._optical_pass_at_visibility(
        A, psi, visibility=1.0, **common
    )
    _, quarter = module._optical_pass_at_visibility(
        A, psi, visibility=0.25, **common
    )
    np.testing.assert_array_equal(zero, incoherent_source)
    np.testing.assert_array_equal(one, coherent_source)
    np.testing.assert_array_equal(quarter, np.full_like(incoherent_source, 4.0))


def test_direct_success_bypasses_visibility_continuation(monkeypatch):
    import lcprop.pr.transverse.static_workflow as module

    def forbidden(*args, **kwargs):
        raise AssertionError("successful direct solve invoked continuation")

    monkeypatch.setattr(
        module, "_run_pr_transverse_static_visibility_continuation", forbidden
    )
    result = run_pr_transverse_static(_two_beam_request(coherent=True))
    assert result.converged
    assert "continuation_used" not in result.diagnostics


def test_coherent_line_search_failure_uses_fixed_stage_schedule_and_warm_starts(
    monkeypatch,
):
    import lcprop.pr.transverse.static_workflow as module

    request = _two_beam_request(coherent=True)
    baseline = module._run_pr_transverse_static_at_visibility(
        request, visibility=1.0
    )
    failed_diagnostics = dict(baseline.diagnostics)
    failed_diagnostics["termination_reason"] = "coupled_line_search_failed"
    failed = replace(
        baseline,
        converged=False,
        status="not_converged",
        diagnostics=failed_diagnostics,
    )
    calls = []

    def staged(stage_request, *, visibility, **kwargs):
        calls.append((visibility, stage_request.initial_psi))
        if len(calls) == 1:
            return failed
        stage_psi = np.full_like(baseline.psi_final, visibility + len(calls))
        return replace(baseline, psi_final=stage_psi)

    monkeypatch.setattr(module, "_run_pr_transverse_static_at_visibility", staged)
    result = run_pr_transverse_static(request)
    assert [call[0] for call in calls] == [
        1.0,
        *module.PR_COHERENT_VISIBILITY_CONTINUATION_SCHEDULE,
    ]
    assert calls[1][1] is request.initial_psi
    for index in range(2, len(calls)):
        np.testing.assert_array_equal(
            calls[index][1],
            np.full_like(baseline.psi_final, calls[index - 1][0] + index),
        )
    assert result.converged
    assert result.diagnostics["continuation_used"] is True
    assert result.diagnostics["visibility_schedule"] == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    assert result.diagnostics["final_visibility"] == 1.0
    assert result.diagnostics["direct_attempt_status"] == "not_converged"


def test_failed_intermediate_visibility_never_becomes_reported_solution(
    monkeypatch,
):
    import lcprop.pr.transverse.static_workflow as module

    request = _two_beam_request(coherent=True)
    converged = module._run_pr_transverse_static_at_visibility(
        request, visibility=1.0
    )
    failed_diagnostics = dict(converged.diagnostics)
    failed_diagnostics["termination_reason"] = "coupled_line_search_failed"
    direct_failure = replace(
        converged,
        converged=False,
        status="not_converged",
        diagnostics=failed_diagnostics,
    )

    def staged(stage_request, *, visibility, **kwargs):
        if visibility < 0.5:
            return converged
        return direct_failure

    monkeypatch.setattr(module, "_run_pr_transverse_static_at_visibility", staged)
    result = module._run_pr_transverse_static_visibility_continuation(
        request, direct_result=direct_failure
    )
    assert result.status == "not_converged"
    assert result.diagnostics["final_visibility"] is None
    assert result.diagnostics["continuation_failure_visibility"] == 0.5
    assert result.diagnostics["returned_state_source"] == (
        "direct_full_visibility_failure"
    )
    np.testing.assert_array_equal(result.psi_final, direct_failure.psi_final)


def test_cancellation_during_continuation_returns_last_accepted_stage_state(
    monkeypatch,
):
    import lcprop.pr.transverse.static_workflow as module

    request = _two_beam_request(coherent=True)
    converged = module._run_pr_transverse_static_at_visibility(
        request, visibility=1.0
    )
    direct_diagnostics = dict(converged.diagnostics)
    direct_diagnostics["termination_reason"] = "coupled_line_search_failed"
    direct_failure = replace(
        converged,
        converged=False,
        status="not_converged",
        diagnostics=direct_diagnostics,
    )
    accepted_psi = np.full_like(converged.psi_final, 3.0)
    accepted_stage = replace(converged, psi_final=accepted_psi)
    cancelled_psi = np.full_like(converged.psi_final, 4.0)
    cancelled_diagnostics = dict(converged.diagnostics)
    cancelled_diagnostics.update({
        "termination_reason": "cancelled_at_accepted_boundary",
        "cancelled": True,
    })
    cancelled_stage = replace(
        converged,
        psi_final=cancelled_psi,
        converged=False,
        status="cancelled",
        diagnostics=cancelled_diagnostics,
    )

    def staged(stage_request, *, visibility, **kwargs):
        if visibility == 0.0:
            return accepted_stage
        assert visibility == 0.25
        np.testing.assert_array_equal(stage_request.initial_psi, accepted_psi)
        return cancelled_stage

    monkeypatch.setattr(module, "_run_pr_transverse_static_at_visibility", staged)
    result = module._run_pr_transverse_static_visibility_continuation(
        request, direct_result=direct_failure
    )
    assert result.status == "cancelled"
    assert not result.converged
    np.testing.assert_array_equal(result.psi_final, cancelled_psi)
    assert result.diagnostics["final_visibility"] is None
    assert result.diagnostics["continuation_cancelled"] is True
    assert result.diagnostics["continuation_cancellation_visibility"] == 0.25
    assert result.diagnostics["continuation_failure_visibility"] is None
    assert result.diagnostics["returned_state_source"] == (
        "cancelled_continuation_stage"
    )


def test_forced_visibility_schedule_matches_easy_direct_solution():
    import lcprop.pr.transverse.static_workflow as module

    request = _request()
    direct = run_pr_transverse_static(request)
    continued = module._run_pr_transverse_static_visibility_continuation(
        request, direct_result=direct
    )
    assert direct.converged and continued.converged
    overlap = np.vdot(continued.A_final.ravel(), direct.A_final.ravel())
    phase = np.exp(-1j * np.angle(overlap))
    np.testing.assert_allclose(phase * continued.A_final, direct.A_final, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(continued.psi_final, direct.psi_final, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        continued.source_intensity_stack,
        direct.source_intensity_stack,
        rtol=1e-10,
        atol=1e-11,
    )
    assert abs(continued.power_final - direct.power_final) < 1e-12


def test_incoherent_failure_does_not_invoke_visibility_continuation(monkeypatch):
    import lcprop.pr.transverse.static_workflow as module

    request = _two_beam_request(coherent=False)
    baseline = module._run_pr_transverse_static_at_visibility(
        request, visibility=1.0
    )
    diagnostics = dict(baseline.diagnostics)
    diagnostics["termination_reason"] = "coupled_line_search_failed"
    failed = replace(
        baseline,
        converged=False,
        status="not_converged",
        diagnostics=diagnostics,
    )
    monkeypatch.setattr(
        module,
        "_run_pr_transverse_static_at_visibility",
        lambda *args, **kwargs: failed,
    )
    result = run_pr_transverse_static(request)
    assert result is failed
    assert "continuation_used" not in result.diagnostics
