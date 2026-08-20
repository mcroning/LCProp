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
    assert np.min(result.source_intensity_stack) >= 0.5
    assert result.timing["material_solve_seconds"] > 0.0
    assert result.timing["optical_pass_seconds"] > 0.0
    assert all(record.material_pcg_iterations >= 0 for record in result.iteration_records)
    assert result.material_iteration_records
    assert all(
        record.newton_record.plane_index in (0, 1)
        for record in result.material_iteration_records
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

    monkeypatch.setattr(module, "solve_pr_transverse_static_intensity", cancelling_solve)
    result = run_pr_transverse_static(_request(), cancellation_token=token)
    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert not any(record.accepted for record in result.iteration_records)


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


def test_static_workflow_rejects_backend_expansion_and_accepts_optical_only_intensity():
    with pytest.raises(ValueError, match="backend='numpy'"):
        run_pr_transverse_static(replace(
            _request(),
            backend=BackendSpec(backend="cupy", precision="float64", verbose=False),
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
