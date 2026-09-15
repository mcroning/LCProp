from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lcprop.persistence  # initialize the codec registry before direct codec use
import lcprop.pr.transverse.static_workflow as static_workflow_module
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
    decode_pr_transverse_static_request,
    encode_pr_transverse_static_request,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.products import pr_transverse_static_result_to_run_data
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
from lcprop.pr.transverse.transport_codec import (
    decode_pr_transverse_static_transport_request,
    decode_pr_transverse_static_transport_result,
    encode_pr_transverse_static_transport_request,
    encode_pr_transverse_static_transport_result,
)
from lcprop.transport.result_policy import FAST_RESULT_POLICY


def _request(
    *,
    linearized: bool,
    waist_um: float = 1000.0,
    gain_length_product: float = 1.0e-5,
    applied_field: float = 0.0,
    precision: str = "float64",
) -> PRTransverseStaticRunRequest:
    request = PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=12,
            Ny=10,
            x_aperture_um=40.0,
            y_aperture_um=36.0,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=waist_um,
            waist_y_um=waist_um,
            coherence_group="linearized-production",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=gain_length_product,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=12),
        backend=BackendSpec(
            backend="numpy", precision=precision, verbose=False
        ),
    )
    if not linearized:
        return request
    profile_id = PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
    return replace(
        request,
        transport=PRTransverseTransportProfile(),
        dielectric=PRTransverseDielectricProfile(),
        boundary=PRTransverseBoundaryProfile(
            profile_id=profile_id,
            applied_field_x=applied_field,
        ),
        projection=PRTransverseProjectionProfile(),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        ),
    )


def test_old_positional_static_request_signature_remains_compatible():
    expected = _request(linearized=False)
    request = PRTransverseStaticRunRequest(
        expected.grid,
        expected.beams,
        expected.material,
        expected.transport,
        expected.dielectric,
        expected.boundary,
        expected.projection,
        expected.solver,
        expected.backend,
        expected.launch_elements,
        expected.initial_A,
        expected.initial_psi,
        expected.scattering,
    )

    for field_name in (
        "grid",
        "beams",
        "material",
        "transport",
        "dielectric",
        "boundary",
        "projection",
        "solver",
        "backend",
        "launch_elements",
        "initial_A",
        "initial_psi",
        "scattering",
    ):
        assert getattr(request, field_name) is getattr(expected, field_name)
    assert request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
    assert request.material_response.reference_intensity is None


@pytest.mark.parametrize(
    ("precision", "real_dtype", "complex_dtype", "atol"),
    (
        ("float64", np.float64, np.complex128, 3.0e-13),
        ("float32", np.float32, np.complex64, 2.0e-6),
    ),
)
def test_production_frozen_material_path_reuses_commissioned_operator(
    precision,
    real_dtype,
    complex_dtype,
    atol,
):
    request = _request(
        linearized=True,
        gain_length_product=0.0,
        applied_field=0.35,
        precision=precision,
    )
    result = run_pr_transverse_static(request)
    profile = result.resolved_profile
    reference = solve_pr_biased_linearized_reference(
        np.asarray(result.source_intensity_stack, dtype=real_dtype),
        spec=PRBiasedLinearizedReferenceSpec(
            reference_intensity=1.5,
            applied_field=0.35,
            dx_normalized=profile["dx_normalized"],
            dy_normalized=profile["dy_normalized"],
            m_y=1.0,
            h_y=1.0,
        ),
        backend=request.backend,
    )

    assert result.converged
    assert result.psi_final.dtype == np.dtype(real_dtype)
    assert result.A_final.dtype == np.dtype(complex_dtype)
    np.testing.assert_allclose(
        result.psi_final, reference.delta_psi, rtol=0.0, atol=atol
    )
    state = state_from_potential(
        result.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
        applied_field_x=0.35,
    )
    np.testing.assert_allclose(
        state.E_x, 0.35 + reference.delta_E_x, rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        state.E_y, reference.delta_E_y, rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        state.carrier_density, 1.0 + reference.delta_P, rtol=0.0, atol=atol
    )


def test_self_consistent_linearized_static_products_and_provenance():
    result = run_pr_transverse_static(
        _request(linearized=True, applied_field=0.2)
    )
    run_data = pr_transverse_static_result_to_run_data(result)

    assert result.converged
    assert result.replay_diagnostics["field_match"]
    assert result.replay_diagnostics["source_match"]
    assert abs(result.diagnostics["optical_power_relative_drift"]) < 1.0e-12
    assert result.diagnostics["material_response"] == "linearized"
    assert result.diagnostics["material_response_validation"] == "experimental"
    assert "discrete_corrector" not in result.diagnostics
    assert result.td_rhs_residual_stack is None
    assert "td_rhs_residual" not in run_data.fields
    assert run_data.workflow == "pr_transverse_static"
    assert result.resolved_profile["transport_model"] == "full_transverse"
    assert result.resolved_profile["electrical_ensemble"] == (
        "fixed_harmonic_mean_field"
    )
    assert result.resolved_profile["reference_intensity"] == 1.5
    assert result.resolved_profile["applied_field"] == 0.2
    assert result.resolved_profile["precision"] == "float64"


def test_linearized_cupy_seam_keeps_operator_calls_plane_local(monkeypatch):
    observed_shapes = []
    original = static_workflow_module.solve_pr_biased_linearized_reference

    def observed(intensity, **kwargs):
        observed_shapes.append(np.shape(intensity))
        return original(intensity, **kwargs)

    monkeypatch.setattr(
        static_workflow_module,
        "solve_pr_biased_linearized_reference",
        observed,
    )
    result = run_pr_transverse_static(
        _request(linearized=True, gain_length_product=0.0)
    )
    assert observed_shapes
    assert all(len(shape) == 2 for shape in observed_shapes)
    assert len(observed_shapes) == result.diagnostics["material_response_calls"]


def test_linearized_and_nonlinear_agree_perturbatively_but_diverge_strongly():
    weak_nonlinear = run_pr_transverse_static(_request(linearized=False))
    weak_linearized = run_pr_transverse_static(_request(linearized=True))
    assert weak_nonlinear.converged and weak_linearized.converged
    weak_relative_potential = np.linalg.norm(
        weak_linearized.psi_final - weak_nonlinear.psi_final
    ) / np.linalg.norm(weak_nonlinear.psi_final)
    weak_relative_field = np.linalg.norm(
        weak_linearized.A_final - weak_nonlinear.A_final
    ) / np.linalg.norm(weak_nonlinear.A_final)
    assert weak_relative_potential < 5.0e-4
    assert weak_relative_field < 1.0e-11

    strong_nonlinear = run_pr_transverse_static(
        _request(linearized=False, waist_um=30.0)
    )
    strong_linearized = run_pr_transverse_static(
        _request(linearized=True, waist_um=30.0)
    )
    assert strong_nonlinear.converged and strong_linearized.converged
    strong_absolute_potential = float(np.max(np.abs(
        strong_linearized.psi_final - strong_nonlinear.psi_final
    )))
    assert strong_absolute_potential > 0.02
    assert strong_absolute_potential > 100.0 * float(np.max(np.abs(
        weak_linearized.psi_final - weak_nonlinear.psi_final
    )))


def test_linearized_request_persistence_and_transport_are_explicit_and_additive():
    request = _request(linearized=True, applied_field=-0.25)
    encoded = encode_pr_transverse_static_request(request)
    assert encoded["schema_version"] == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
    assert encoded["material_response"] == {
        "model": "linearized",
        "reference_intensity": 1.5,
    }
    assert decode_pr_transverse_static_request(encoded) == request

    legacy = encode_pr_transverse_static_request(_request(linearized=False))
    legacy["schema_version"] = 2
    legacy.pop("material_response")
    legacy.pop("optical_boundary")
    decoded_legacy = decode_pr_transverse_static_request(legacy)
    assert decoded_legacy.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
    assert decoded_legacy.material_response.reference_intensity is None

    portable = encode_pr_transverse_static_transport_request(request)
    decoded = decode_pr_transverse_static_transport_request(
        portable.payload.metadata, portable.payload.arrays
    )
    assert decoded == request

    result = run_pr_transverse_static(request)
    full = encode_pr_transverse_static_transport_result(result)
    decoded_result = decode_pr_transverse_static_transport_result(
        full.payload.metadata, full.payload.arrays
    )
    assert decoded_result.resolved_profile["material_response"] == {
        "model": "linearized",
        "reference_intensity": 1.5,
    }
    assert decoded_result.td_rhs_residual_stack is None

    fast = encode_pr_transverse_static_transport_result(
        result, result_policy=FAST_RESULT_POLICY
    )
    fast_result = decode_pr_transverse_static_transport_result(
        fast.payload.metadata, fast.payload.arrays
    )
    fast_data = pr_transverse_static_result_to_run_data(fast_result)
    full_data = pr_transverse_static_result_to_run_data(decoded_result)
    assert fast_result.retention_summary["policy"] == FAST_RESULT_POLICY
    assert {"input_intensity", "output_intensity"}.issubset(fast_data.fields)
    assert "psi" not in fast_data.fields
    np.testing.assert_array_equal(
        fast_result.longitudinal_intensity_xz,
        full_data.fields["optical_intensity_xz"].data,
    )
    np.testing.assert_array_equal(
        fast_result.longitudinal_intensity_yz,
        full_data.fields["optical_intensity_yz"].data,
    )
    assert fast_data.longitudinal_enabled


def test_linearized_request_requires_explicit_reference_and_single_bias_owner():
    request = _request(linearized=True)
    with pytest.raises(ValueError, match="reference_intensity"):
        run_pr_transverse_static(replace(
            request,
            material_response=PRTransverseMaterialResponseSpec(
                model=PR_MATERIAL_RESPONSE_LINEARIZED
            ),
        ))
    with pytest.raises(ValueError, match="reduced x-only material setting"):
        run_pr_transverse_static(replace(
            request,
            material=replace(request.material, applied_field=0.2),
        ))
    with pytest.raises(ValueError, match="profiles must match"):
        run_pr_transverse_static(replace(
            request,
            boundary=PRTransverseBoundaryProfile(),
        ))


def test_linearized_progress_and_cancellation_use_outer_boundaries():
    progress = []
    result = run_pr_transverse_static(
        _request(linearized=True, applied_field=0.2),
        progress_callback=progress.append,
    )
    assert result.converged
    assert progress
    assert progress[-1].coordinate_name == "coupled_iteration"
    assert progress[-1].diagnostics["material_response"] == "linearized"
    assert "outer iteration accepted" in progress[-1].message

    token = CancellationToken()
    token.cancel()
    cancelled = run_pr_transverse_static(
        _request(linearized=True), cancellation_token=token
    )
    assert cancelled.status == "cancelled"
    assert cancelled.diagnostics["termination_reason"] == (
        "cancelled_at_accepted_boundary"
    )


def test_periodic_biased_profile_is_not_silently_enabled_for_td():
    static_request = _request(linearized=False)
    td_request = PRTransverseRunRequest(
        grid=static_request.grid,
        beams=static_request.beams,
        material=static_request.material,
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=0.2,
        ),
        backend=static_request.backend,
    )
    with pytest.raises(ValueError, match="does not support"):
        run_pr_transverse_timedependent(td_request)
