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
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.timedependent_transport_codec import (
    decode_pr_timedependent_transport_request,
    decode_pr_timedependent_transport_result,
    encode_pr_timedependent_transport_request,
    encode_pr_timedependent_transport_result,
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


def _request(*, precision: str = "float64", steps: int = 2) -> PRRunRequest:
    return PRRunRequest(
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
            applied_field=0.4,
            gain_length_product=0.02,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(
            Nt=steps,
            dt_normalized=0.001,
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


def _screened_request() -> PRRunRequest:
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
            ChannelLaunchElements(channel_index=0, elements=(screen,)),
        ),
    )


def _roundtrip_request(request: PRRunRequest) -> PRRunRequest:
    encoded = encode_pr_timedependent_transport_request(request)
    return decode_pr_timedependent_transport_request(
        encoded.payload.metadata, encoded.payload.arrays
    )


def _roundtrip_result(result):
    encoded = encode_pr_timedependent_transport_result(result)
    decoded = decode_pr_timedependent_transport_result(
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
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
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
        "grid_summary",
        "launch_summary",
    ):
        assert getattr(actual, name) == getattr(expected, name)
    _assert_nested_equal(actual.diagnostics, expected.diagnostics)
    assert actual.checkpoint.request == expected.checkpoint.request
    for name in ("A0", "E_initial", "E_current"):
        np.testing.assert_array_equal(
            getattr(actual.checkpoint, name), getattr(expected.checkpoint, name)
        )


def test_reduced_td_codec_is_registered_as_an_ordinary_operation():
    key = PR_TIMEDEPENDENT_OPERATION.key
    assert key in {codec.key for codec in default_transport_registry().codecs}
    assert key in {operation.key for operation in default_transport_operations()}


@pytest.mark.parametrize("factory", (_request, _screened_request))
def test_reduced_td_request_and_result_match_fake_remote_execution(factory):
    request = factory()
    transported_request = _roundtrip_request(request)
    assert transported_request == request
    local = PR_TIMEDEPENDENT_OPERATION.run(request)
    remote_equivalent = PR_TIMEDEPENDENT_OPERATION.run(transported_request)
    encoded, transported = _roundtrip_result(remote_equivalent)
    _assert_result_equal(transported, remote_equivalent)
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
        np.testing.assert_array_equal(
            getattr(transported, name), getattr(local, name)
        )
    assert encoded.scientific_status == "completed"
    assert encoded.converged is None
    assert encoded.termination_reason == "requested_material_steps_completed"
    for name in (
        "backend",
        "conservative_dt_limit",
        "paper_equation_15_dt_limit",
        "legacy_prprop3d_dt_limit",
        "cancellation_observed_stage",
        "cancellation_observed_wall_time",
        "integrator",
        "canonical_scattering",
    ):
        assert name in transported.diagnostics
    original_products = PR_TIMEDEPENDENT_OPERATION.to_run_data(local)
    transported_products = PR_TIMEDEPENDENT_OPERATION.to_run_data(transported)
    assert tuple(transported_products.fields) == tuple(original_products.fields)
    for key in original_products.fields:
        np.testing.assert_array_equal(
            transported_products.fields[key].data,
            original_products.fields[key].data,
        )


def test_reduced_td_transport_preserves_float32_and_backend_provenance():
    request = _request(precision="float32")
    decoded_request = _roundtrip_request(request)
    result = PR_TIMEDEPENDENT_OPERATION.run(decoded_request)
    encoded, decoded = _roundtrip_result(result)
    assert decoded_request.backend == request.backend
    assert encoded.scientific_backend_resolved == "numpy"
    assert decoded.A_final.dtype == np.complex64
    assert decoded.E_final.dtype == np.float32
    assert decoded.source_intensity_stack.dtype == np.float32
    assert decoded.diagnostics["backend"]["real_dtype"] == "float32"


def test_reduced_td_fast_projection_keeps_optics_and_omits_full_volumes():
    result = PR_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_timedependent_transport_result(result, "fast")
    decoded = decode_pr_timedependent_transport_result(
        encoded.payload.metadata, encoded.payload.arrays
    )
    assert encoded.result_policy == "fast"
    np.testing.assert_array_equal(decoded.A_initial, result.A_initial)
    np.testing.assert_array_equal(decoded.A_final, result.A_final)
    assert decoded.E_initial is None
    assert decoded.E_final is None
    assert decoded.source_intensity_stack is None
    assert decoded.checkpoint is None
    assert set(decoded.retention_summary["omitted_fields"]) == {
        "E_initial", "E_final", "source_intensity_stack", "checkpoint"
    }
    products = PR_TIMEDEPENDENT_OPERATION.to_run_data(decoded)
    assert tuple(products.fields) == (
        "input_intensity", "output_intensity", "far_field_intensity"
    )
    assert products.longitudinal_enabled is False
    assert "Full result retrieval" in products.longitudinal_message


@pytest.mark.parametrize("after_accepted_step", (False, True))
def test_reduced_td_cancelled_transport_is_explicit_and_preserves_boundary(
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
    result = PR_TIMEDEPENDENT_OPERATION.run(
        request,
        cancellation_token=token,
        progress_callback=callback,
    )
    assert result.status == "cancelled"
    assert result.completed_steps == int(after_accepted_step)
    encoded, decoded = _roundtrip_result(result)
    assert encoded.cancelled is True
    assert encoded.converged is None
    assert encoded.termination_reason == "cancelled_at_accepted_material_boundary"
    assert decoded.status == "cancelled"
    assert decoded.checkpoint.status == "cancelled"
    _assert_result_equal(decoded, result)


def test_reduced_td_transport_rejects_malformed_shape_and_checkpoint_mismatch():
    result = PR_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_timedependent_transport_result(result)
    arrays = dict(encoded.payload.arrays)
    arrays["result__E_final"] = arrays["result__E_final"][:, :-1, :]
    with pytest.raises(TransportCodecError, match="E_final shape"):
        decode_pr_timedependent_transport_result(
            encoded.payload.metadata, arrays
        )

    arrays = dict(encoded.payload.arrays)
    arrays["checkpoint__E_current"] = arrays[
        "checkpoint__E_current"
    ].copy()
    arrays["checkpoint__E_current"][0, 0, 0] += 1.0
    with pytest.raises(TransportCodecError, match="E_final disagrees"):
        decode_pr_timedependent_transport_result(
            encoded.payload.metadata, arrays
        )


def test_reduced_td_transport_rejects_missing_optical_array_and_bad_status():
    result = PR_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_timedependent_transport_result(result)
    arrays = dict(encoded.payload.arrays)
    del arrays["result__A_final"]
    with pytest.raises(TransportCodecError, match="missing transported PR array"):
        decode_pr_timedependent_transport_result(
            encoded.payload.metadata, arrays
        )

    metadata = dict(encoded.payload.metadata)
    metadata["status"] = "failed"
    with pytest.raises(TransportCodecError, match="status is invalid"):
        decode_pr_timedependent_transport_result(
            metadata, encoded.payload.arrays
        )


def test_reduced_td_transport_wraps_malformed_request_payload():
    encoded = encode_pr_timedependent_transport_request(_request())
    metadata = dict(encoded.payload.metadata)
    del metadata["beams"]
    with pytest.raises(TransportCodecError, match="invalid PR time-dependent"):
        decode_pr_timedependent_transport_request(
            metadata, encoded.payload.arrays
        )


def test_reduced_td_checkpoint_duplication_is_exact_and_measurable():
    result = PR_TIMEDEPENDENT_OPERATION.run(_request())
    encoded = encode_pr_timedependent_transport_result(result)
    arrays = encoded.payload.arrays
    duplicate_bytes = sum(
        arrays[name].nbytes
        for name in (
            "checkpoint__A0",
            "checkpoint__E_initial",
            "checkpoint__E_current",
        )
    )
    correctness_bytes = sum(array.nbytes for array in arrays.values())
    assert duplicate_bytes > 0
    assert correctness_bytes > duplicate_bytes
    np.testing.assert_array_equal(
        arrays["checkpoint__A0"], arrays["result__A_initial"]
    )
    np.testing.assert_array_equal(
        arrays["checkpoint__E_initial"], arrays["result__E_initial"]
    )
    np.testing.assert_array_equal(
        arrays["checkpoint__E_current"], arrays["result__E_final"]
    )


def test_reduced_td_verified_artifact_roundtrip_regenerates_products(tmp_path):
    registry = default_transport_registry()
    request = _screened_request()
    run_dir = tmp_path / "reduced-td"
    write_request_package(
        run_dir,
        registry=registry,
        material_id="pr",
        workflow_id="pr_timedependent",
        request=request,
        run_id="reduced-td-run",
        execution_target="local",
    )
    decoded_request = read_request_package(run_dir, registry=registry)
    result = PR_TIMEDEPENDENT_OPERATION.run(decoded_request.request)
    write_result_package(
        run_dir,
        codec=registry.codec("pr", "pr_timedependent"),
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
    assert regenerated.kind == "pr_timedependent"
    assert regenerated.run_data.workflow == "pr_timedependent"
    assert tuple(regenerated.run_data.fields) == tuple(
        PR_TIMEDEPENDENT_OPERATION.to_run_data(result).fields
    )
