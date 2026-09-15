from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import lcprop.persistence  # initialize experiment/transport registrations
import lcprop.pr.workflow as workflow_module
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
    decode_pr_timedependent_request,
    encode_pr_timedependent_request,
)
from lcprop.pr.gui.run_cost import classify_pr_run_cost
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.persistence import (
    PR_CHECKPOINT_SCHEMA_VERSION,
    PR_CHECKPOINT_SUPPORTED_SCHEMA_VERSIONS,
    load_pr_checkpoint,
    save_pr_checkpoint,
)
from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.reduced_linearized import PRReducedLinearizedSpec
from lcprop.pr.reduced_linearized_timedependent import (
    solve_pr_reduced_linearized_timedependent,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_EULER_INTEGRATOR,
    PR_EXACT_MODAL_INTEGRATOR,
)
from lcprop.pr.timedependent_transport_codec import (
    PR_TIMEDEPENDENT_TRANSPORT_CODEC,
    decode_pr_timedependent_transport_request,
    decode_pr_timedependent_transport_result,
    encode_pr_timedependent_transport_request,
    encode_pr_timedependent_transport_result,
)
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.workflow import continue_pr_timedependent, run_pr_timedependent
from lcprop.transport.result_policy import FAST_RESULT_POLICY
from lcprop.transport.defaults import default_transport_operations


def _request(
    *,
    linearized: bool = True,
    precision: str = "float64",
    steps: int = 2,
    bias: float = 0.5,
) -> PRRunRequest:
    request = PRRunRequest(
        grid=GridSpec(
            Nx=12,
            Ny=6,
            x_aperture_um=24.0,
            y_aperture_um=12.0,
            dz_um=4.0,
            z_length_um=8.0,
        ),
        beams=BeamStack(
            channels=(BeamChannel(
                wavelength_um=0.633,
                power_mW=1.0,
                waist_x_um=8.0,
                waist_y_um=7.0,
                coherence_group="reduced-linearized-td",
            ),)
        ),
        material=PRMaterialSpec(
            dark_intensity=0.25,
            uniform_background_intensity=0.15,
            applied_field=bias,
            gain_length_product=0.01,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRSolverOptions(
            Nt=steps,
            dt_normalized=0.2 if linearized else 0.001,
            optical_substeps=1,
            integrator=(
                PR_EXACT_MODAL_INTEGRATOR if linearized else PR_EULER_INTEGRATOR
            ),
        ),
        backend=BackendSpec(backend="numpy", precision=precision, verbose=False),
    )
    if not linearized:
        return request
    return replace(
        request,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        ),
    )


def _source(request: PRRunRequest, dtype=np.float64):
    nx, ny = request.grid.Nx, request.grid.Ny
    nz = round(request.grid.z_length_um / request.grid.dz_um)
    x = np.arange(nx)[:, None] * 2.0 * np.pi / nx
    y = np.arange(ny)[None, :] * 2.0 * np.pi / ny
    plane = 1.5 + 0.08 * np.cos(2.0 * x) + 0.03 * np.sin(3.0 * x + y)
    return np.stack([plane + 0.01 * index for index in range(nz)]).astype(dtype)


def _fixed_source(monkeypatch, source, calls=None):
    def optical(A0, E, **kwargs):
        if calls is not None:
            calls.append(np.asarray(E).copy())
        result = (
            A0.copy(),
            kwargs["grid"].xp.asarray(source, dtype=kwargs["grid"].real_dtype),
        )
        return result

    monkeypatch.setattr(workflow_module, "_optical_pass", optical)


def _assert_result_physics_equal(actual, expected):
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.completed_steps == expected.completed_steps
    assert actual.time_normalized == expected.time_normalized


def test_old_positional_request_signature_defaults_to_nonlinear_without_shifting():
    expected = _request(linearized=False)
    positional = PRRunRequest(
        expected.grid,
        expected.beams,
        expected.material,
        expected.solver,
        expected.backend,
        expected.launch_elements,
        expected.initial_A,
        expected.initial_E,
        expected.scattering,
        expected.optical_boundary,
    )
    for name in (
        "grid", "beams", "material", "solver", "backend", "initial_A",
        "initial_E", "scattering", "launch_elements", "optical_boundary",
    ):
        assert getattr(positional, name) is getattr(expected, name)
    assert positional.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR


def test_explicit_I0_and_exact_integrator_pairing_are_required():
    nonlinear = _request(linearized=False)
    with pytest.raises(ValueError, match="reference_intensity"):
        run_pr_timedependent(
            replace(
                nonlinear,
                solver=replace(
                    nonlinear.solver, integrator=PR_EXACT_MODAL_INTEGRATOR
                ),
                material_response=PRTransverseMaterialResponseSpec(
                    model=PR_MATERIAL_RESPONSE_LINEARIZED
                ),
            )
        )
    with pytest.raises(ValueError, match="requires integrator='exact_modal'"):
        request = _request()
        run_pr_timedependent(
            replace(
                request,
                solver=replace(request.solver, integrator=PR_EULER_INTEGRATOR),
            )
        )
    with pytest.raises(ValueError, match="nonlinear TD does not support"):
        run_pr_timedependent(
            replace(
                nonlinear,
                solver=replace(
                    nonlinear.solver, integrator=PR_EXACT_MODAL_INTEGRATOR
                ),
            )
        )


@pytest.mark.parametrize(
    ("precision", "dtype", "atol"),
    [("float64", np.float64, 3e-14), ("float32", np.float32, 4e-6)],
)
def test_production_dispatch_matches_reference_and_reports_exact_metadata(
    monkeypatch, precision, dtype, atol
):
    request = _request(precision=precision, steps=1)
    source = _source(request, dtype)
    _fixed_source(monkeypatch, source)
    result = run_pr_timedependent(request)
    spec = PRReducedLinearizedSpec(
        reference_intensity=1.5,
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity,
        dx_normalized=(
            request.material.characteristic_wavenumber_per_um
            * request.grid.x_aperture_um / request.grid.Nx
        ),
    )
    expected = solve_pr_reduced_linearized_timedependent(
        np.full(source.shape, spec.equilibrium_field, dtype=dtype),
        source,
        dt_normalized=request.solver.dt_normalized,
        spec=spec,
        backend=request.backend,
    )
    assert result.E_final.dtype == dtype
    np.testing.assert_allclose(result.E_final, expected.E, rtol=0.0, atol=atol)
    assert result.diagnostics["integrator"] == PR_EXACT_MODAL_INTEGRATOR
    assert result.diagnostics["integrator_policy"] == (
        "exact_frozen_source_modal_update"
    )
    assert result.diagnostics["source_cadence"] == (
        "one_complete_optical_pass_per_material_interval"
    )
    assert result.material_response_summary == {
        "model": "linearized",
        "reference_intensity": 1.5,
        "software_evidence": "locally_validated",
        "integrator": "exact_modal",
    }


def test_optical_source_cadence_is_one_pass_per_interval_plus_final_replay(monkeypatch):
    request = _request(steps=3)
    calls = []
    _fixed_source(monkeypatch, _source(request), calls=calls)
    result = run_pr_timedependent(request)
    assert result.completed_steps == 3
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0], result.E_initial)
    np.testing.assert_array_equal(calls[-1], result.E_final)


def test_cancellation_discards_incomplete_candidate_and_resume_is_exact(monkeypatch):
    request = _request(steps=3)
    source = _source(request)
    pre_cancelled_token = CancellationToken()
    pre_cancelled_token.cancel()
    pre_cancelled_token.cancel()
    pre_cancelled = run_pr_timedependent(
        request, cancellation_token=pre_cancelled_token
    )
    assert pre_cancelled.completed_steps == 0
    np.testing.assert_array_equal(pre_cancelled.E_final, pre_cancelled.E_initial)

    token = CancellationToken()
    original_advance = workflow_module.advance_pr_slice_with_midpoint_source
    traversed_slices = []

    def cancel_during_traversal(*args, **kwargs):
        result = original_advance(*args, **kwargs)
        traversed_slices.append(1)
        token.cancel()
        return result

    monkeypatch.setattr(
        workflow_module,
        "advance_pr_slice_with_midpoint_source",
        cancel_during_traversal,
    )
    cancelled = run_pr_timedependent(request, cancellation_token=token)
    assert cancelled.status == "cancelled"
    assert cancelled.completed_steps == 0
    assert traversed_slices == [1]
    assert cancelled.diagnostics["cancellation_observed_stage"] == (
        "material_source_optical_z_march"
    )
    np.testing.assert_array_equal(cancelled.E_final, cancelled.E_initial)

    monkeypatch.setattr(
        workflow_module,
        "advance_pr_slice_with_midpoint_source",
        original_advance,
    )
    _fixed_source(monkeypatch, source)
    progress_token = CancellationToken()
    partial = run_pr_timedependent(
        request,
        cancellation_token=progress_token,
        progress_callback=lambda update: progress_token.cancel(),
    )
    assert partial.completed_steps == 1
    resumed = continue_pr_timedependent(request, partial.checkpoint, 2)
    uninterrupted = run_pr_timedependent(request)
    _assert_result_physics_equal(resumed, uninterrupted)


def test_experiment_transport_fast_full_products_and_legacy_migration(monkeypatch):
    request = _request(steps=1)
    _fixed_source(monkeypatch, _source(request))
    experiment = encode_pr_timedependent_request(request)
    assert experiment["schema_version"] == (
        PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
    ) == 6
    assert decode_pr_timedependent_request(experiment) == request

    legacy_request = _request(linearized=False, steps=1)
    legacy = encode_pr_timedependent_request(legacy_request)
    legacy["schema_version"] = 4
    legacy.pop("material_response")
    legacy.pop("scattering")
    assert (
        decode_pr_timedependent_request(legacy).material_response.model
        == "nonlinear"
    )

    encoded_request = encode_pr_timedependent_transport_request(request)
    decoded_request = decode_pr_timedependent_transport_request(
        encoded_request.payload.metadata, encoded_request.payload.arrays
    )
    assert decoded_request == request
    old_metadata = dict(
        encode_pr_timedependent_transport_request(legacy_request).payload.metadata
    )
    old_metadata.pop("material_response")
    assert decode_pr_timedependent_transport_request(
        old_metadata, {}
    ).material_response.model == "nonlinear"
    assert PR_TIMEDEPENDENT_TRANSPORT_CODEC.request_codec_version == 3
    assert (
        PR_TIMEDEPENDENT_TRANSPORT_CODEC.compatible_request_codec_versions
        == (1, 2)
    )

    result = run_pr_timedependent(request)
    full_encoded = encode_pr_timedependent_transport_result(result)
    full = decode_pr_timedependent_transport_result(
        full_encoded.payload.metadata, full_encoded.payload.arrays
    )
    np.testing.assert_array_equal(full.E_final, result.E_final)
    assert full.material_response_summary == result.material_response_summary
    full_products = pr_result_to_run_data(full)
    assert full_products.diagnostics["summary"].values[
        "material_response"
    ]["model"] == "linearized"

    fast_encoded = encode_pr_timedependent_transport_result(
        result, FAST_RESULT_POLICY
    )
    fast = decode_pr_timedependent_transport_result(
        fast_encoded.payload.metadata, fast_encoded.payload.arrays
    )
    assert fast.E_final is None
    assert fast.checkpoint is None
    assert fast.longitudinal_intensity_xz is not None
    assert fast.longitudinal_intensity_yz is not None
    assert fast.material_response_summary == result.material_response_summary
    fast_products = pr_result_to_run_data(fast)
    assert fast_products.longitudinal_enabled is True
    assert fast_products.diagnostics["summary"].values[
        "material_response"
    ]["integrator"] == "exact_modal"


def test_linearized_checkpoint_roundtrip_and_old_checkpoint_default(
    monkeypatch, tmp_path
):
    request = _request(steps=1)
    _fixed_source(monkeypatch, _source(request))
    result = run_pr_timedependent(request)
    save_pr_checkpoint(result.checkpoint, tmp_path / "linearized")
    assert PR_CHECKPOINT_SCHEMA_VERSION == 4
    assert PR_CHECKPOINT_SUPPORTED_SCHEMA_VERSIONS == (1, 2, 3, 4)
    loaded = load_pr_checkpoint(tmp_path / "linearized")
    assert loaded.request.material_response == request.material_response
    assert loaded.request.solver.integrator == PR_EXACT_MODAL_INTEGRATOR
    np.testing.assert_array_equal(loaded.E_current, result.E_final)

    nonlinear = run_pr_timedependent(_request(linearized=False, steps=1))
    old_path = tmp_path / "old"
    save_pr_checkpoint(nonlinear.checkpoint, old_path)
    request_path = old_path / "request.json"
    request_document = json.loads(request_path.read_text(encoding="utf-8"))
    request_document["schema_version"] = 3
    request_document["request"].pop("material_response")
    request_path.write_text(json.dumps(request_document), encoding="utf-8")
    provenance_path = old_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["schema_version"] = 3
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    migrated = load_pr_checkpoint(old_path)
    assert migrated.request.material_response.model == "nonlinear"


def test_nonlinear_default_produces_the_preexisting_result_exactly():
    request = _request(linearized=False, steps=1)
    explicit = replace(request, material_response=PRTransverseMaterialResponseSpec())
    implicit_result = run_pr_timedependent(request)
    explicit_result = run_pr_timedependent(explicit)
    _assert_result_physics_equal(implicit_result, explicit_result)


def test_operation_registration_runner_and_fixed_count_cost_metadata(monkeypatch):
    request = _request(steps=4)
    assert PR_TIMEDEPENDENT_OPERATION.key in {
        operation.key for operation in default_transport_operations()
    }
    _fixed_source(monkeypatch, _source(request))
    result = PR_TIMEDEPENDENT_OPERATION.run(request)
    assert result.completed_steps == 4
    assert result.material_response_summary["integrator"] == "exact_modal"
    cost = classify_pr_run_cost(request, execution_target="local")
    assert "Linearized time dependent" in cost.model_label
    assert "exact one-dimensional modal" in cost.rationale


def test_real_optical_material_path_runs_headlessly_without_special_dispatch():
    result = PR_TIMEDEPENDENT_OPERATION.run(_request(steps=1))
    assert result.status == "completed"
    assert result.completed_steps == 1
    assert np.isfinite(result.E_final).all()
    assert np.isfinite(result.A_final).all()
    assert result.source_intensity_stack.shape == result.E_final.shape


def test_cupy_production_seam_remains_conditional():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CuPy GPU is unavailable")
    request = replace(
        _request(precision="float32", steps=1),
        backend=BackendSpec(backend="cupy", precision="float32", verbose=False),
    )
    result = run_pr_timedependent(request)
    assert result.E_final.dtype == np.float32
