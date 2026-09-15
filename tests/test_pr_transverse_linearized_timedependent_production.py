from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lcprop.persistence  # initialize transport registrations
import lcprop.pr.transverse.workflow as workflow_module
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    decode_pr_transverse_timedependent_request,
    encode_pr_transverse_timedependent_request,
)
from lcprop.pr.gui.run_cost import classify_pr_run_cost
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.linearized_timedependent_reference import (
    linearized_timedependent_rhs,
    solve_pr_biased_linearized_timedependent_reference,
)
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
from lcprop.pr.transverse.timedependent_transport_codec import (
    decode_pr_transverse_timedependent_transport_request,
    decode_pr_transverse_timedependent_transport_result,
    encode_pr_transverse_timedependent_transport_request,
    encode_pr_transverse_timedependent_transport_result,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
from lcprop.transport.result_policy import FAST_RESULT_POLICY


def _request(
    *,
    linearized: bool = True,
    precision: str = "float64",
    steps: int = 1,
    dt: float = 0.2,
    bias: float = 0.35,
    initial_psi=None,
) -> PRTransverseRunRequest:
    request = PRTransverseRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=8,
            x_aperture_um=16.0,
            y_aperture_um=16.0,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=7.0,
            waist_y_um=6.0,
            coherence_group="linearized-td-production",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=0.0,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=dt,
            optical_substeps=1,
        ),
        backend=BackendSpec(
            backend="numpy", precision=precision, verbose=False
        ),
        initial_psi=initial_psi,
    )
    if not linearized:
        return request
    return replace(
        request,
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=bias,
        ),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        ),
    )


def _fixed_source(monkeypatch, source):
    def optical(A0, psi, **kwargs):
        return A0.copy(), kwargs["grid"].xp.asarray(
            source, dtype=kwargs["grid"].real_dtype
        )

    monkeypatch.setattr(workflow_module, "_optical_pass", optical)


def _resolved_source(shape=(2, 8, 8), dtype=np.float64):
    _, nx, ny = shape
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    plane = 1.5 + 0.08 * np.sin(2.0 * np.pi * x / nx)
    plane = plane + 0.04 * np.cos(4.0 * np.pi * y / ny)
    return np.stack([plane + 0.01 * index for index in range(shape[0])]).astype(
        dtype
    )


def _assert_nested_equal(actual, expected):
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_old_positional_td_request_signature_and_default_are_unchanged():
    expected = _request(linearized=False)
    request = PRTransverseRunRequest(
        expected.grid,
        expected.beams,
        expected.material,
        expected.transport,
        expected.dielectric,
        expected.boundary,
        expected.projection,
        expected.solver,
        expected.backend,
        expected.initial_A,
        expected.initial_psi,
        expected.scattering,
        expected.launch_elements,
    )

    for name in (
        "grid", "beams", "material", "transport", "dielectric", "boundary",
        "projection", "solver", "backend", "initial_A", "initial_psi",
        "scattering", "launch_elements",
    ):
        assert getattr(request, name) is getattr(expected, name)
    assert request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
    assert request.material_response.reference_intensity is None


def test_linearized_td_requires_explicit_reference_and_matching_bias_profile():
    nonlinear = _request(linearized=False)
    missing_reference = replace(
        nonlinear,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED
        ),
    )
    with pytest.raises(ValueError, match="reference_intensity"):
        run_pr_transverse_timedependent(missing_reference)
    with pytest.raises(ValueError, match="must match"):
        run_pr_transverse_timedependent(
            replace(
                _request(),
                boundary=PRTransverseBoundaryProfile(),
            )
        )


@pytest.mark.parametrize(
    ("precision", "dtype", "atol"),
    (("float64", np.float64, 2.0e-14), ("float32", np.float32, 2.0e-6)),
)
def test_production_update_equals_exact_reference_and_preserves_dtype(
    monkeypatch, precision, dtype, atol
):
    source = _resolved_source(dtype=dtype)
    rng = np.random.default_rng(42)
    initial = rng.normal(scale=0.03, size=source.shape).astype(dtype)
    request = _request(precision=precision, initial_psi=initial)
    _fixed_source(monkeypatch, source)

    result = run_pr_transverse_timedependent(request)
    profile = result.resolved_profile
    expected = solve_pr_biased_linearized_timedependent_reference(
        source,
        time_normalized=request.solver.dt_normalized,
        spec=PRBiasedLinearizedReferenceSpec(
            reference_intensity=1.5,
            applied_field=0.35,
            dx_normalized=profile["dx_normalized"],
            dy_normalized=profile["dy_normalized"],
        ),
        initial_delta_psi=initial - initial.mean(axis=(-2, -1), keepdims=True),
        backend=request.backend,
    )

    assert result.psi_final.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result.psi_final, expected.delta_psi, atol=atol, rtol=0)
    assert result.diagnostics["integrator_policy"] == (
        "exact_frozen_source_linearized_modal_update"
    )
    assert result.diagnostics["material_response_calls"] == source.shape[0]


def test_zero_and_equilibrium_initial_conditions_reach_expected_states(monkeypatch):
    source = _resolved_source()
    request = _request(dt=0.0 + 0.3)
    _fixed_source(monkeypatch, source)
    first = run_pr_transverse_timedependent(request)
    profile = first.resolved_profile
    spec = PRBiasedLinearizedReferenceSpec(
        reference_intensity=1.5,
        applied_field=0.35,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
    )
    equilibrium = solve_pr_biased_linearized_reference(
        source, spec=spec, backend=request.backend
    ).delta_psi
    equilibrium_run = run_pr_transverse_timedependent(
        replace(request, initial_psi=equilibrium)
    )
    long_run = run_pr_transverse_timedependent(
        replace(request, solver=replace(request.solver, dt_normalized=30.0))
    )

    assert np.linalg.norm(first.psi_final) > 0.0
    np.testing.assert_allclose(
        equilibrium_run.psi_final, equilibrium, atol=2.0e-14, rtol=0
    )
    np.testing.assert_allclose(long_run.psi_final, equilibrium, atol=2.0e-12, rtol=0)
    td_state = state_from_potential(
        long_run.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
        applied_field_x=0.35,
    )
    static_state = state_from_potential(
        equilibrium,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
        applied_field_x=0.35,
    )
    np.testing.assert_allclose(td_state.E_x, static_state.E_x, atol=2e-12, rtol=0)
    np.testing.assert_allclose(td_state.E_y, static_state.E_y, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        project_active_field(td_state.E_x, td_state.E_y, profile=request.projection),
        project_active_field(
            static_state.E_x, static_state.E_y, profile=request.projection
        ),
        atol=2e-12,
        rtol=0,
    )
    residual = linearized_timedependent_rhs(
        long_run.psi_final, source, spec=spec, backend=request.backend
    )
    assert np.max(np.abs(residual)) < 2e-12


def test_bias_reversal_and_null_modes_pass_through_production(monkeypatch):
    x = np.arange(8)[:, None]
    y = np.arange(8)[None, :]
    plane = (
        1.5
        + 0.08 * np.cos(2.0 * np.pi * x / 8)
        + 0.04 * np.cos(4.0 * np.pi * y / 8)
    )
    source = np.stack((plane, plane + 0.01))
    source[:, 4, :] += 0.07 * (-1.0) ** np.arange(8)[None, :]
    _fixed_source(monkeypatch, source)
    positive = run_pr_transverse_timedependent(_request(bias=0.4))
    negative = run_pr_transverse_timedependent(_request(bias=-0.4))
    positive_hat = np.fft.fft2(positive.psi_final, axes=(-2, -1))
    negative_hat = np.fft.fft2(negative.psi_final, axes=(-2, -1))

    np.testing.assert_allclose(
        negative_hat, np.conj(positive_hat), atol=2.0e-13, rtol=0
    )
    assert np.imag(positive_hat[0, 1, 0]) < 0.0
    assert np.imag(negative_hat[0, 1, 0]) > 0.0
    assert np.max(np.abs(positive_hat[:, 4, 4])) < 1.0e-14
    assert np.max(np.abs(positive_hat[:, 0, 0])) < 1.0e-14


def test_linearized_calls_are_plane_local_and_cancellation_discards_candidate(
    monkeypatch,
):
    source = _resolved_source()
    _fixed_source(monkeypatch, source)
    token = CancellationToken()
    observed = []
    original = workflow_module.solve_pr_biased_linearized_timedependent_reference

    def cancelling(value, **kwargs):
        observed.append(value.shape)
        result = original(value, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(
        workflow_module,
        "solve_pr_biased_linearized_timedependent_reference",
        cancelling,
    )
    result = run_pr_transverse_timedependent(
        _request(steps=2), cancellation_token=token
    )

    assert observed == [(8, 8)]
    assert result.status == "cancelled"
    assert result.completed_steps == 0
    assert result.diagnostics["cancellation_observed_stage"] == (
        "linearized_material_plane"
    )
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)


def test_pre_cancelled_linearized_run_stops_before_source_traversal():
    token = CancellationToken()
    token.cancel()
    result = run_pr_transverse_timedependent(
        _request(steps=2), cancellation_token=token
    )

    assert result.status == "cancelled"
    assert result.completed_steps == 0
    assert result.diagnostics["cancellation_observed_stage"] == (
        "material_step_boundary"
    )
    assert result.diagnostics["material_response_calls"] == 0


def test_progress_and_products_identify_experimental_linearized_td(monkeypatch):
    source = _resolved_source()
    _fixed_source(monkeypatch, source)
    progress = []
    result = run_pr_transverse_timedependent(
        _request(steps=2), progress_callback=progress.append
    )
    run_data = pr_transverse_result_to_run_data(result)

    assert [item.completed_units for item in progress] == [1, 2]
    assert all(
        item.diagnostics["material_update"] == "exact_frozen_source_modal"
        for item in progress
    )
    assert result.diagnostics["material_response"] == "linearized"
    assert result.diagnostics["material_response_validation"] == "experimental"
    assert result.diagnostics["accepted_material_steps"] == 2
    assert result.diagnostics["material_time_normalized"] == pytest.approx(0.4)
    assert np.isfinite(result.diagnostics["linearized_rhs_rms"])
    assert np.isfinite(result.diagnostics["linearized_rhs_max"])
    assert result.resolved_profile["material_response"] == {
        "model": "linearized", "reference_intensity": 1.5
    }
    assert run_data.diagnostics["summary"].values["physics_profile"][
        "validation_status"
    ] == "experimental"
    assert "E_x" in run_data.fields


def test_full_optical_material_propagation_with_nonzero_gain():
    request = _request(steps=2)
    request = replace(
        request,
        material=replace(request.material, gain_length_product=0.03),
    )
    result = run_pr_transverse_timedependent(request)
    run_data = pr_transverse_result_to_run_data(result)

    assert result.status == "completed"
    assert result.completed_steps == 2
    assert result.diagnostics["complete_final_optical_replay"] is True
    assert result.diagnostics["finite_optical_state"] is True
    assert np.linalg.norm(result.A_final - result.A_initial) > 0.0
    assert np.all(np.isfinite(run_data.fields["E_active"].data))


@pytest.mark.parametrize(("nx", "ny"), ((7, 9), (8, 10)))
def test_odd_and_even_production_grids_match_direct_reference(
    monkeypatch, nx, ny
):
    request = _request()
    request = replace(
        request,
        grid=replace(
            request.grid,
            Nx=nx,
            Ny=ny,
            x_aperture_um=2.0 * nx,
            y_aperture_um=2.0 * ny,
        ),
    )
    source = _resolved_source((2, nx, ny))
    _fixed_source(monkeypatch, source)
    result = run_pr_transverse_timedependent(request)
    profile = result.resolved_profile
    direct = solve_pr_biased_linearized_timedependent_reference(
        source,
        time_normalized=request.solver.dt_normalized,
        spec=PRBiasedLinearizedReferenceSpec(
            reference_intensity=1.5,
            applied_field=0.35,
            dx_normalized=profile["dx_normalized"],
            dy_normalized=profile["dy_normalized"],
        ),
        backend=request.backend,
    )
    np.testing.assert_allclose(result.psi_final, direct.delta_psi, atol=2e-14, rtol=0)


def test_experiment_and_transport_round_trip_with_legacy_default():
    request = _request()
    experiment = encode_pr_transverse_timedependent_request(request)
    assert experiment["material_response"] == {
        "model": "linearized", "reference_intensity": 1.5
    }
    assert decode_pr_transverse_timedependent_request(experiment) == request
    legacy_experiment = dict(encode_pr_transverse_timedependent_request(
        _request(linearized=False)
    ))
    legacy_experiment["schema_version"] = 2
    legacy_experiment.pop("material_response")
    legacy_experiment.pop("optical_boundary")
    assert decode_pr_transverse_timedependent_request(
        legacy_experiment
    ).material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR

    encoded = encode_pr_transverse_timedependent_transport_request(request)
    decoded = decode_pr_transverse_timedependent_transport_request(
        encoded.payload.metadata, encoded.payload.arrays
    )
    assert decoded == request
    legacy_metadata = dict(
        encode_pr_transverse_timedependent_transport_request(
            _request(linearized=False)
        ).payload.metadata
    )
    legacy_metadata.pop("material_response")
    decoded_legacy = decode_pr_transverse_timedependent_transport_request(
        legacy_metadata, {}
    )
    assert decoded_legacy.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR

    result = run_pr_transverse_timedependent(request)
    encoded_result = encode_pr_transverse_timedependent_transport_result(result)
    decoded_result = decode_pr_transverse_timedependent_transport_result(
        encoded_result.payload.metadata, encoded_result.payload.arrays
    )
    assert decoded_result.resolved_profile["material_response"] == {
        "model": "linearized", "reference_intensity": 1.5
    }
    encoded_fast = encode_pr_transverse_timedependent_transport_result(
        result, FAST_RESULT_POLICY
    )
    decoded_fast = decode_pr_transverse_timedependent_transport_result(
        encoded_fast.payload.metadata, encoded_fast.payload.arrays
    )
    np.testing.assert_array_equal(decoded_fast.A_initial, decoded_result.A_initial)
    np.testing.assert_array_equal(decoded_fast.A_final, decoded_result.A_final)
    _assert_nested_equal(decoded_fast.diagnostics, decoded_result.diagnostics)
    _assert_nested_equal(decoded_fast.resolved_profile, decoded_result.resolved_profile)
    assert decoded_fast.psi_initial is None
    assert decoded_fast.psi_final is None
    np.testing.assert_array_equal(
        decoded_fast.longitudinal_intensity_xz,
        result.longitudinal_intensity_xz,
    )
    np.testing.assert_array_equal(
        decoded_fast.longitudinal_intensity_yz,
        result.longitudinal_intensity_yz,
    )
    fast_run_data = pr_transverse_result_to_run_data(decoded_fast)
    assert "input_intensity" in fast_run_data.fields
    assert "output_intensity" in fast_run_data.fields
    assert "psi" not in fast_run_data.fields
    assert fast_run_data.longitudinal_enabled


def test_linearized_cost_is_no_higher_than_nonlinear_cost():
    nonlinear = _request(linearized=False, steps=10)
    linearized = _request(steps=10)
    nonlinear_cost = classify_pr_run_cost(nonlinear, execution_target="local")
    linearized_cost = classify_pr_run_cost(linearized, execution_target="local")

    assert linearized_cost.work_score < nonlinear_cost.work_score
    assert "Linearized" in linearized_cost.model_label


def test_weak_linearized_and_nonlinear_updates_agree(monkeypatch):
    source = _resolved_source()
    source = 1.5 + 1.0e-5 * (source - source.mean(axis=(-2, -1), keepdims=True))
    _fixed_source(monkeypatch, source)
    linearized = run_pr_transverse_timedependent(_request(dt=1.0e-5, bias=0.0))
    nonlinear = run_pr_transverse_timedependent(
        _request(linearized=False, dt=1.0e-5)
    )

    difference = nonlinear.psi_final - linearized.psi_final
    assert np.linalg.norm(difference) / np.linalg.norm(linearized.psi_final) < 7e-5
    assert np.max(np.abs(difference)) < 8e-16
    np.testing.assert_array_equal(nonlinear.A_final, linearized.A_final)


def test_weak_full_optical_runs_agree_with_nonzero_gain(monkeypatch):
    linearized_request = _request(dt=1e-4, bias=0.0)
    channel = linearized_request.beams.channels[0]
    linearized_request = replace(
        linearized_request,
        beams=replace(
            linearized_request.beams,
            channels=(replace(
                channel, waist_x_um=1.0e5, waist_y_um=1.0e5
            ),),
        ),
        material=replace(
            linearized_request.material, gain_length_product=1.0e-3
        ),
    )
    nonlinear_request = replace(
        linearized_request,
        boundary=PRTransverseBoundaryProfile(),
        material_response=PRTransverseMaterialResponseSpec(),
    )
    original = workflow_module._optical_pass
    observed_sources = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        observed_sources.append(np.asarray(result[1]).copy())
        return result

    monkeypatch.setattr(workflow_module, "_optical_pass", observed)
    linearized = run_pr_transverse_timedependent(linearized_request)
    linearized_sources = tuple(observed_sources)
    observed_sources.clear()
    nonlinear = run_pr_transverse_timedependent(nonlinear_request)
    nonlinear_sources = tuple(observed_sources)

    potential_difference = nonlinear.psi_final - linearized.psi_final
    assert np.linalg.norm(potential_difference) / np.linalg.norm(
        linearized.psi_final
    ) < 5e-4
    assert np.max(np.abs(potential_difference)) < 7e-16
    assert np.linalg.norm(nonlinear.A_final - linearized.A_final) / np.linalg.norm(
        linearized.A_final
    ) < 2e-18
    for nonlinear_source, linearized_source in zip(
        nonlinear_sources, linearized_sources
    ):
        np.testing.assert_array_equal(nonlinear_source, linearized_source)


def test_cupy_float32_seam_is_conditional():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")
    request = replace(
        _request(precision="float32"),
        backend=BackendSpec(backend="cupy", precision="float32", verbose=False),
    )
    result = run_pr_transverse_timedependent(request)
    assert result.psi_final.dtype == np.dtype(np.float32)
