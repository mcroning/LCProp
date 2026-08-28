from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.operations import PR_TRANSVERSE_TIMEDEPENDENT_OPERATION
from lcprop.pr.transverse.specs import (
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.timedependent_transport_codec import (
    decode_pr_transverse_timedependent_transport_request,
    decode_pr_transverse_timedependent_transport_result,
    encode_pr_transverse_timedependent_transport_request,
    encode_pr_transverse_timedependent_transport_result,
)
from lcprop.transport.defaults import (
    default_transport_operations,
    default_transport_registry,
)
from lcprop.transport.envelopes import TransportCodecError
from lcprop.transport.io import (
    read_request_package,
    read_result_package,
    runner_result_from_package,
    write_request_package,
    write_result_package,
)


def _request(
    *, precision: str = "float64", steps: int = 2
) -> PRTransverseRunRequest:
    return PRTransverseRunRequest(
        grid=GridSpec(
            Nx=10,
            Ny=8,
            x_aperture_um=30.0,
            y_aperture_um=24.0,
            dz_um=2.0,
            z_length_um=4.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    name="signal",
                    wavelength_um=0.633,
                    power_mW=0.8,
                    waist_x_um=8.0,
                    waist_y_um=7.0,
                    tilt_x_rad_per_um=0.08,
                    tilt_y_rad_per_um=-0.04,
                    phase_rad=0.2,
                    coherence_group="laser",
                ),
                BeamChannel(
                    name="pump",
                    wavelength_um=0.633,
                    power_mW=1.2,
                    waist_x_um=9.0,
                    waist_y_um=8.0,
                    tilt_x_rad_per_um=-0.06,
                    tilt_y_rad_per_um=0.03,
                    phase_rad=-0.1,
                    coherence_group="laser",
                ),
            ),
            coherence="coherent",
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=0.02,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        transport=PRTransverseTransportProfile(m_y=1.0),
        dielectric=PRTransverseDielectricProfile(h_y=1.0),
        boundary=PRTransverseBoundaryProfile(applied_field_x=0.0),
        projection=PRTransverseProjectionProfile(g_x=1.0, g_y=0.0),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=1.0e-4,
            optical_substeps=2,
        ),
        backend=BackendSpec(
            backend="numpy", precision=precision, verbose=False
        ),
        scattering=PRCanonicalScatteringSpec(
            epsilon=1e-6,
            transverse_correlation_um=3.0,
            realization_seed=1234,
            canonical_dz_um=2.0,
        ),
    )


def _screened_request() -> PRTransverseRunRequest:
    screen = IntensityRasterScreen(
        source=RasterSource.from_array(
            np.asarray([[0.2, 0.8], [1.0, 0.4]], dtype=float)
        ),
        placement=ScreenPlacement(
            center_x_um=1.0,
            center_y_um=-1.0,
            width_um=12.0,
            height_um=10.0,
            outside_intensity_transmission=0.5,
        ),
    )
    return replace(
        _request(),
        launch_elements=(
            ChannelLaunchElements(channel_index=1, elements=(screen,)),
        ),
    )


def _roundtrip_request(request: PRTransverseRunRequest):
    encoded = encode_pr_transverse_timedependent_transport_request(request)
    decoded = decode_pr_transverse_timedependent_transport_request(
        encoded.payload.metadata, encoded.payload.arrays
    )
    return encoded, decoded


def _roundtrip_result(result):
    encoded = encode_pr_transverse_timedependent_transport_result(result)
    decoded = decode_pr_transverse_timedependent_transport_result(
        encoded.payload.metadata, encoded.payload.arrays
    )
    return encoded, decoded


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)) and isinstance(
        expected, (tuple, list)
    ):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_result_equal(actual, expected) -> None:
    for name in ("A_initial", "A_final", "psi_initial", "psi_final"):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name)
        )
    for name in (
        "power_initial",
        "power_final",
        "completed_steps",
        "requested_steps",
        "time_normalized",
        "status",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    for name in (
        "grid_summary",
        "launch_summary",
        "backend_summary",
        "resolved_profile",
    ):
        _assert_nested_equal(getattr(actual, name), getattr(expected, name))
    _assert_nested_equal(actual.diagnostics, expected.diagnostics)


def test_transverse_td_codec_and_operation_are_registered_by_default():
    key = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.key
    assert key in {codec.key for codec in default_transport_registry().codecs}
    assert key in {operation.key for operation in default_transport_operations()}


def test_transverse_td_fast_projection_retains_optics_and_far_field():
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_transverse_timedependent_transport_result(result, "fast")
    decoded = decode_pr_transverse_timedependent_transport_result(
        encoded.payload.metadata, encoded.payload.arrays
    )
    assert encoded.result_policy == "fast"
    np.testing.assert_array_equal(decoded.A_initial, result.A_initial)
    np.testing.assert_array_equal(decoded.A_final, result.A_final)
    assert decoded.psi_initial is None
    assert decoded.psi_final is None
    products = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.to_run_data(decoded)
    assert "input_intensity" in products.fields
    assert "output_intensity" in products.fields
    assert "far_field_intensity" in products.fields
    assert products.longitudinal_enabled is False


@pytest.mark.parametrize("factory", (_request, _screened_request))
def test_transverse_td_local_and_fake_remote_execution_are_exact(factory):
    request = factory()
    encoded_request, transported_request = _roundtrip_request(request)
    assert transported_request == request
    assert encoded_request.scientific_backend_requested == "numpy"
    local = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(request)
    remote_equivalent = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(
        transported_request
    )
    encoded_result, transported = _roundtrip_result(remote_equivalent)
    _assert_result_equal(transported, remote_equivalent)
    _assert_result_equal(transported, local)
    assert encoded_result.scientific_status == "completed"
    assert encoded_result.converged is None
    assert encoded_result.cancelled is False
    assert (
        encoded_result.termination_reason
        == "requested_material_steps_completed"
    )
    original_products = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.to_run_data(local)
    transported_products = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.to_run_data(
        transported
    )
    assert transported_products.geometry.units == original_products.geometry.units
    for coordinate in ("x", "y", "z"):
        np.testing.assert_array_equal(
            getattr(transported_products.geometry, coordinate),
            getattr(original_products.geometry, coordinate),
        )
    assert tuple(transported_products.fields) == tuple(original_products.fields)
    for key in original_products.fields:
        np.testing.assert_array_equal(
            transported_products.fields[key].data,
            original_products.fields[key].data,
        )
    assert tuple(transported_products.diagnostics) == tuple(
        original_products.diagnostics
    )


def test_screened_transport_preserves_source_and_exact_once_launch():
    request = _screened_request()
    _, decoded = _roundtrip_request(request)
    source = request.launch_elements[0].elements[0].source
    decoded_source = decoded.launch_elements[0].elements[0].source
    assert decoded_source.sha256 == source.sha256
    np.testing.assert_array_equal(decoded_source.grayscale, source.grayscale)
    assert decoded.launch_elements == request.launch_elements
    screened = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(decoded)
    direct = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(request)
    np.testing.assert_array_equal(screened.A_initial, direct.A_initial)
    plain = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(
        replace(request, launch_elements=())
    )
    np.testing.assert_array_equal(screened.A_initial[0], plain.A_initial[0])
    assert not np.array_equal(screened.A_initial[1], plain.A_initial[1])


def test_explicit_initial_fields_round_trip_and_conflict_is_preserved():
    request = _request(steps=0)
    shape_A = (len(request.beams.channels), request.grid.Nx, request.grid.Ny)
    shape_psi = (2, request.grid.Nx, request.grid.Ny)
    prepared = replace(
        request,
        initial_A=np.ones(shape_A, dtype=np.complex128),
        initial_psi=np.zeros(shape_psi, dtype=np.float64),
    )
    _, decoded = _roundtrip_request(prepared)
    np.testing.assert_array_equal(decoded.initial_A, prepared.initial_A)
    np.testing.assert_array_equal(decoded.initial_psi, prepared.initial_psi)
    with pytest.raises(ValueError, match="already-prepared runtime launch"):
        encode_pr_transverse_timedependent_transport_request(
            replace(prepared, launch_elements=_screened_request().launch_elements)
        )


def test_transverse_td_float32_and_backend_provenance_are_preserved():
    request = _request(precision="float32")
    _, decoded_request = _roundtrip_request(request)
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(decoded_request)
    encoded, decoded = _roundtrip_result(result)
    assert decoded_request.backend == request.backend
    assert encoded.scientific_backend_resolved == "numpy"
    assert encoded.device_summary["is_gpu"] is False
    assert decoded.A_final.dtype == np.complex64
    assert decoded.psi_final.dtype == np.float32
    assert decoded.backend_summary["real_dtype"] == "float32"

    cupy_style = replace(
        result,
        backend_summary={
            **result.backend_summary,
            "backend": "cupy",
            "is_gpu": True,
        },
    )
    encoded_cupy, decoded_cupy = _roundtrip_result(cupy_style)
    assert encoded_cupy.scientific_backend_resolved == "cupy"
    assert encoded_cupy.device_summary["is_gpu"] is True
    assert decoded_cupy.backend_summary["backend"] == "cupy"


@pytest.mark.parametrize("after_accepted_step", (False, True))
def test_transverse_td_cancellation_transport_preserves_accepted_boundary(
    after_accepted_step,
):
    request = _request(steps=3)
    token = CancellationToken()
    callback = None
    if after_accepted_step:

        def cancel_after_first(progress):
            if progress.completed_units == 1:
                token.cancel()

        callback = cancel_after_first
    else:
        token.cancel()
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(
        request,
        cancellation_token=token,
        progress_callback=callback,
    )
    assert result.status == "cancelled"
    assert result.completed_steps == int(after_accepted_step)
    encoded, decoded = _roundtrip_result(result)
    assert encoded.cancelled is True
    assert encoded.converged is None
    assert (
        encoded.termination_reason
        == "cancelled_at_accepted_material_boundary"
    )
    assert decoded.status == "cancelled"
    _assert_result_equal(decoded, result)


def test_transverse_td_result_rejects_shapes_status_and_missing_fields():
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_transverse_timedependent_transport_result(result)
    arrays = dict(encoded.payload.arrays)
    arrays["result__psi_final"] = arrays["result__psi_final"][:, :-1, :]
    with pytest.raises(TransportCodecError, match="psi_final shape"):
        decode_pr_transverse_timedependent_transport_result(
            encoded.payload.metadata, arrays
        )

    metadata = dict(encoded.payload.metadata)
    metadata["status"] = "converged"
    with pytest.raises(TransportCodecError, match="status is invalid"):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded.payload.arrays
        )

    metadata = dict(encoded.payload.metadata)
    del metadata["A_final"]
    with pytest.raises(TransportCodecError, match="missing A_final"):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded.payload.arrays
        )


def test_transverse_td_result_rejects_incoherent_completion_and_cancellation():
    completed = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(_request(steps=2))
    encoded_completed = encode_pr_transverse_timedependent_transport_result(
        completed
    )
    with pytest.raises(
        TransportCodecError, match="completed.*did not reach requested steps"
    ):
        encode_pr_transverse_timedependent_transport_result(
            replace(completed, completed_steps=1)
        )
    metadata = dict(encoded_completed.payload.metadata)
    metadata["completed_steps"] = 1
    with pytest.raises(
        TransportCodecError, match="completed.*did not reach requested steps"
    ):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded_completed.payload.arrays
        )

    completed_with_cancellation = replace(
        completed,
        diagnostics={
            **completed.diagnostics,
            "cancellation_observed_stage": "material_step_boundary",
        },
    )
    with pytest.raises(
        TransportCodecError, match="completed.*cancellation provenance"
    ):
        encode_pr_transverse_timedependent_transport_result(
            completed_with_cancellation
        )
    metadata = dict(encoded_completed.payload.metadata)
    metadata["diagnostics"] = {
        **metadata["diagnostics"],
        "cancellation_observed_stage": "material_step_boundary",
    }
    with pytest.raises(
        TransportCodecError, match="completed.*cancellation provenance"
    ):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded_completed.payload.arrays
        )

    token = CancellationToken()
    token.cancel()
    cancelled = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(
        _request(steps=2), cancellation_token=token
    )
    encoded_cancelled = encode_pr_transverse_timedependent_transport_result(
        cancelled
    )
    with pytest.raises(
        TransportCodecError, match="cancelled.*reached all requested steps"
    ):
        encode_pr_transverse_timedependent_transport_result(
            replace(cancelled, completed_steps=cancelled.requested_steps)
        )
    metadata = dict(encoded_cancelled.payload.metadata)
    metadata["completed_steps"] = cancelled.requested_steps
    with pytest.raises(
        TransportCodecError, match="cancelled.*reached all requested steps"
    ):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded_cancelled.payload.arrays
        )

    cancelled_without_stage = replace(
        cancelled,
        diagnostics={
            **cancelled.diagnostics,
            "cancellation_observed_stage": None,
        },
    )
    with pytest.raises(
        TransportCodecError, match="lacks valid cancellation provenance"
    ):
        encode_pr_transverse_timedependent_transport_result(
            cancelled_without_stage
        )
    metadata = dict(encoded_cancelled.payload.metadata)
    metadata["diagnostics"] = {
        **metadata["diagnostics"],
        "cancellation_observed_stage": None,
    }
    with pytest.raises(
        TransportCodecError, match="lacks valid cancellation provenance"
    ):
        decode_pr_transverse_timedependent_transport_result(
            metadata, encoded_cancelled.payload.arrays
        )


def test_transverse_td_request_rejects_malformed_payload():
    encoded = encode_pr_transverse_timedependent_transport_request(_request())
    metadata = dict(encoded.payload.metadata)
    del metadata["projection"]
    with pytest.raises(TransportCodecError, match="missing projection"):
        decode_pr_transverse_timedependent_transport_request(
            metadata, encoded.payload.arrays
        )


def test_transverse_td_artifact_roundtrip_regenerates_products(tmp_path):
    registry = default_transport_registry()
    request = _screened_request()
    run_dir = tmp_path / "transverse-td"
    write_request_package(
        run_dir,
        registry=registry,
        material_id="pr",
        workflow_id="pr_transverse_timedependent",
        request=request,
        run_id="transverse-td-run",
        execution_target="local",
    )
    decoded_request = read_request_package(run_dir, registry=registry)
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(decoded_request.request)
    write_result_package(
        run_dir,
        codec=registry.codec("pr", "pr_transverse_timedependent"),
        result=result,
        request_envelope=decoded_request.envelope,
    )
    decoded_result = read_result_package(run_dir, registry=registry)
    _assert_result_equal(decoded_result.result, result)
    regenerated = runner_result_from_package(
        run_dir,
        registry=registry,
        operations=default_transport_operations(),
    )
    assert regenerated.kind == "pr_transverse_timedependent"
    assert regenerated.run_data.workflow == "pr_transverse_timedependent"
    assert tuple(regenerated.run_data.fields) == tuple(
        PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.to_run_data(result).fields
    )


def test_transverse_td_result_size_is_canonical_state_plus_diagnostics():
    result = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_transverse_timedependent_transport_result(result)
    arrays = encoded.payload.arrays
    state_keys = {
        "result__A_initial",
        "result__A_final",
        "result__psi_initial",
        "result__psi_final",
    }
    assert state_keys.issubset(arrays)
    assert set(arrays) - state_keys == {
        "result__diagnostics__carrier_integrals_per_z"
    }
    optical_bytes = arrays["result__A_initial"].nbytes + arrays[
        "result__A_final"
    ].nbytes
    material_bytes = arrays["result__psi_initial"].nbytes + arrays[
        "result__psi_final"
    ].nbytes
    assert optical_bytes > 0
    assert material_bytes > 0
    diagnostic_bytes = arrays[
        "result__diagnostics__carrier_integrals_per_z"
    ].nbytes
    assert sum(value.nbytes for value in arrays.values()) == (
        optical_bytes + material_bytes + diagnostic_bytes
    )
