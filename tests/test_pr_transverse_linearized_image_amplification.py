from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    ScreenPlacement,
)
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.image_amplification import (
    PRImageAmplificationExperimentRequest,
    prepare_image_amplification_base_request,
    run_image_amplification_experiment,
)
from lcprop.pr.image_sources import PRImageSource
from lcprop.pr.specs import PRMaterialSpec, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)
from lcprop.runners.base import RunnerResult
from lcprop.runners.local import LocalRunner
from lcprop.transport.defaults import default_transport_registry
from lcprop.transport.result_policy import FAST_RESULT_POLICY, FULL_RESULT_POLICY


_SPECIALIZED_FIELDS = (
    "image_source",
    "image_transmission",
    "input_signal_intensity",
    "output_signal_intensity",
    "amplified_image",
    "zero_response_image",
    "signal_carrier_mask",
    "far_field_intensity",
    "far_field_log_db",
)


def _experiment(
    *,
    strong: bool,
    linearized: bool,
    applied_field: float = 0.0,
) -> PRImageAmplificationExperimentRequest:
    grid = GridSpec(
        Nx=24,
        Ny=16,
        x_aperture_um=48.0,
        y_aperture_um=32.0,
        z_length_um=10.0,
        dz_um=5.0,
    )
    carrier = 2.0 * np.pi * 3.0 / grid.x_aperture_um
    signal_power = 0.08 if strong else 1.0e-8
    waist = 1000.0 if strong else 1.0e6
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="pump",
                power_mW=1.0,
                waist_x_um=waist,
                waist_y_um=waist,
                tilt_x_rad_per_um=carrier,
                coherence_group="linearized-ia-validation",
            ),
            BeamChannel(
                name="signal",
                power_mW=signal_power,
                waist_x_um=waist,
                waist_y_um=waist,
                tilt_x_rad_per_um=-carrier,
                coherence_group="linearized-ia-validation",
            ),
        )
    )
    yy, xx = np.indices((8, 8))
    checkerboard = ((xx + yy) % 2).astype(np.float64)
    raster = (
        0.15 + 0.85 * checkerboard
        if strong
        else 0.98 + 0.02 * checkerboard
    )
    source = PRImageSource.from_array(raster)
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(width_um=20.0, height_um=16.0),
    )
    launch = LaunchConfiguration(
        beams=beams,
        channel_elements=(ChannelLaunchElements(1, (screen,)),),
    )
    base = PRTransverseStaticRunRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=0.16 if strong else 0.01,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=20),
        backend=BackendSpec("numpy", "float64", False),
    )
    if linearized:
        base = replace(
            base,
            boundary=PRTransverseBoundaryProfile(
                profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
                applied_field_x=applied_field,
            ),
            material_response=PRTransverseMaterialResponseSpec(
                model=PR_MATERIAL_RESPONSE_LINEARIZED,
                reference_intensity=1.5,
            ),
        )
    return PRImageAmplificationExperimentRequest(
        base_workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
        base_request=base,
        launch_configuration=launch,
        pump_channel_index=0,
        signal_channel_index=1,
    )


def _run(request):
    return run_image_amplification_experiment(
        LocalRunner((PR_TRANSVERSE_STATIC_OPERATION,)), request
    )


def _relative_l2(left, right) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), np.finfo(np.float64).tiny)
    )


@pytest.fixture(scope="module")
def paired_results():
    return {
        (strong, linearized): _run(
            _experiment(strong=strong, linearized=linearized)
        )
        for strong in (False, True)
        for linearized in (False, True)
    }


def test_paired_requests_preserve_the_controlled_experiment_definition():
    nonlinear = _experiment(strong=False, linearized=False)
    linearized = _experiment(strong=False, linearized=True)

    assert nonlinear.base_workflow_id == linearized.base_workflow_id
    assert nonlinear.launch_configuration == linearized.launch_configuration
    assert nonlinear.pump_channel_index == linearized.pump_channel_index
    assert nonlinear.signal_channel_index == linearized.signal_channel_index
    for name in (
        "grid",
        "beams",
        "material",
        "transport",
        "dielectric",
        "projection",
        "solver",
        "backend",
        "launch_elements",
        "initial_A",
        "initial_psi",
        "scattering",
    ):
        assert getattr(nonlinear.base_request, name) == getattr(
            linearized.base_request, name
        )
    assert nonlinear.base_request.material_response.model == (
        PR_MATERIAL_RESPONSE_NONLINEAR
    )
    assert linearized.base_request.material_response == (
        PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        )
    )
    assert nonlinear.base_request.boundary.applied_field_x == 0.0
    assert linearized.base_request.boundary.applied_field_x == 0.0
    assert nonlinear.source.sha256 == linearized.source.sha256
    assert nonlinear.beams.coherence_groups == (
        "linearized-ia-validation",
        "linearized-ia-validation",
    )


def test_weak_modulation_agrees_and_preserves_ia_mathematics(paired_results):
    nonlinear = paired_results[(False, False)]
    linearized = paired_results[(False, True)]
    n_result = nonlinear.result.run_result
    l_result = linearized.result.run_result
    n_analysis = nonlinear.result.analysis_result
    l_analysis = linearized.result.analysis_result

    assert n_result.converged and l_result.converged
    assert nonlinear.result.analysis_status == "completed"
    assert linearized.result.analysis_status == "completed"
    assert _relative_l2(n_result.A_initial, l_result.A_initial) == 0.0
    assert _relative_l2(n_result.A_final, l_result.A_final) < 1.0e-10
    assert _relative_l2(
        nonlinear.run_data.fields["output_intensity"].data,
        linearized.run_data.fields["output_intensity"].data,
    ) < 1.0e-10
    assert _relative_l2(
        nonlinear.run_data.fields["far_field_intensity"].data,
        linearized.run_data.fields["far_field_intensity"].data,
    ) < 1.0e-10
    np.testing.assert_array_equal(
        n_analysis.signal_carrier_mask, l_analysis.signal_carrier_mask
    )
    assert n_analysis.transverse_phase_gradients_rad_per_um == (
        l_analysis.transverse_phase_gradients_rad_per_um
    )
    assert int(np.count_nonzero(n_analysis.signal_carrier_mask)) == int(
        np.count_nonzero(l_analysis.signal_carrier_mask)
    )
    assert _relative_l2(
        n_analysis.output_signal_field, l_analysis.output_signal_field
    ) < 1.0e-7
    amplified_image = nonlinear.run_data.fields["amplified_image"].data
    linearized_amplified_image = linearized.run_data.fields[
        "amplified_image"
    ].data
    amplified_image_relative_l2 = _relative_l2(
        amplified_image, linearized_amplified_image
    )
    amplified_image_max_absolute = float(
        np.max(np.abs(amplified_image - linearized_amplified_image))
    )
    assert amplified_image_relative_l2 < 1.0e-8
    assert amplified_image_max_absolute < 1.0e-18

    zero_response_image = nonlinear.run_data.fields["zero_response_image"].data
    linearized_zero_response_image = linearized.run_data.fields[
        "zero_response_image"
    ].data
    zero_response_image_relative_l2 = _relative_l2(
        zero_response_image, linearized_zero_response_image
    )
    zero_response_image_max_absolute = float(
        np.max(
            np.abs(zero_response_image - linearized_zero_response_image)
        )
    )
    assert zero_response_image_relative_l2 == 0.0
    assert zero_response_image_max_absolute == 0.0
    grid = make_grid(nonlinear.result.request.grid, real_dtype=np.float64)
    assert n_analysis.output_isolated_signal_power_normalized == pytest.approx(
        float(
            np.sum(np.abs(n_analysis.output_signal_field) ** 2)
            * grid.dx_um
            * grid.dy_um
        ),
        rel=0.0,
        abs=2.0e-18,
    )
    assert abs(
        n_analysis.measured_absolute_signal_gain
        - l_analysis.measured_absolute_signal_gain
    ) < 1.0e-9
    assert n_analysis.analytic_absolute_signal_gain == pytest.approx(
        l_analysis.analytic_absolute_signal_gain, rel=0.0, abs=0.0
    )
    assert abs(
        n_analysis.image_intensity_correlation
        - l_analysis.image_intensity_correlation
    ) < 1.0e-8
    assert abs(n_analysis.normalized_image_rmse - l_analysis.normalized_image_rmse) < (
        1.0e-8
    )
    assert abs(n_analysis.normalized_power_relative_drift) < 1.0e-12
    assert abs(l_analysis.normalized_power_relative_drift) < 1.0e-12

    for key in ("psi", "E_x", "E_y", "E_active", "equilibrium_residual"):
        left = nonlinear.run_data.fields[key].data
        right = linearized.run_data.fields[key].data
        assert np.all(np.isfinite(left))
        assert np.all(np.isfinite(right))
    for key in ("psi", "E_x", "E_y", "E_active"):
        assert _relative_l2(
            nonlinear.run_data.fields[key].data,
            linearized.run_data.fields[key].data,
        ) < 1.0e-3
    assert np.max(
        np.abs(
            nonlinear.run_data.fields["equilibrium_residual"].data
            - linearized.run_data.fields["equilibrium_residual"].data
        )
    ) < 2.0e-8


def test_stronger_modulation_exhibits_expected_model_divergence(paired_results):
    nonlinear = paired_results[(True, False)]
    linearized = paired_results[(True, True)]
    n_result = nonlinear.result.run_result
    l_result = linearized.result.run_result
    n_analysis = nonlinear.result.analysis_result
    l_analysis = linearized.result.analysis_result

    assert n_result.converged and l_result.converged
    assert _relative_l2(n_result.psi_final, l_result.psi_final) > 0.5
    assert _relative_l2(n_result.A_final, l_result.A_final) > 1.0e-3
    assert abs(
        n_analysis.measured_absolute_signal_gain
        - l_analysis.measured_absolute_signal_gain
    ) > 1.0e-4
    assert abs(
        n_analysis.image_intensity_correlation
        - l_analysis.image_intensity_correlation
    ) > 4.0e-5
    assert abs(n_analysis.normalized_image_rmse - l_analysis.normalized_image_rmse) > (
        2.0e-4
    )
    np.testing.assert_array_equal(
        n_analysis.signal_carrier_mask, l_analysis.signal_carrier_mask
    )


def test_linearized_ia_products_provenance_and_nonzero_bias(paired_results):
    local = paired_results[(False, True)]
    result = local.result.run_result
    analysis = local.result.analysis_result
    run_data = local.run_data

    assert run_data.workflow == "pr_image_amplification"
    assert set(_SPECIALIZED_FIELDS).issubset(run_data.fields)
    assert {"input_intensity", "output_intensity"}.issubset(run_data.fields)
    for key in _SPECIALIZED_FIELDS:
        assert np.all(np.isfinite(run_data.fields[key].data))
    profile = run_data.diagnostics["summary"].values["physics_profile"]
    assert profile["workflow"] == PR_TRANSVERSE_STATIC_WORKFLOW
    assert profile["material_response"] == {
        "model": PR_MATERIAL_RESPONSE_LINEARIZED,
        "reference_intensity": 1.5,
    }
    assert profile["physics_profile_id"] == (
        PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
    )
    assert profile["validation_status"] == "experimental"
    assert profile["reference_intensity"] == 1.5
    assert profile["applied_field"] == 0.0
    assert profile["requested_backend"] == "numpy"
    assert profile["precision"] == "float64"
    assert result.launch_summary["coherence_groups"] == [
        "linearized-ia-validation",
        "linearized-ia-validation",
    ]
    assert analysis.reconstruction_optical_runtime_s >= 0.0
    assert (
        analysis.reconstruction_runtime_s
        >= analysis.reconstruction_optical_runtime_s
    )

    grid = make_grid(local.result.request.grid, real_dtype=np.float64)
    inverse_kernel = linear_kernel(
        grid.fxy2_um,
        dz=-float(local.result.request.grid.z_length_um),
        wavelength=float(local.result.request.beams.channels[0].wavelength_um),
        n_ref=float(local.result.request.material.refractive_index),
        xp=np,
    )
    manual_backpropagation = np.fft.ifft2(
        np.fft.fft2(analysis.output_signal_field) * inverse_kernel
    )
    np.testing.assert_allclose(
        analysis.backpropagated_signal_field,
        manual_backpropagation,
        rtol=0.0,
        atol=2.0e-17,
    )

    biased = _run(
        _experiment(strong=False, linearized=True, applied_field=0.2)
    )
    biased_profile = biased.result.run_result.resolved_profile
    assert biased.result.status == "converged"
    assert biased.result.analysis_status == "completed"
    assert biased_profile["reference_intensity"] == 1.5
    assert biased_profile["applied_field"] == 0.2


class _ProjectedRunner:
    def __init__(self, policy):
        self.policy = policy

    def run_registered(self, material_id, workflow_id, request, **kwargs):
        local = LocalRunner((PR_TRANSVERSE_STATIC_OPERATION,)).run_registered(
            material_id, workflow_id, request, **kwargs
        )
        codec = default_transport_registry().codec(material_id, workflow_id)
        encoded = codec.encode_result_for_policy(local.result, self.policy)
        decoded = codec.decode_result(
            encoded.payload.metadata, encoded.payload.arrays
        )
        return replace(
            local,
            result=decoded,
            run_data=PR_TRANSVERSE_STATIC_OPERATION.to_run_data(decoded),
        )


def test_linearized_ia_fast_and_full_transport_are_equivalent():
    request = _experiment(strong=False, linearized=True, applied_field=0.2)
    full = run_image_amplification_experiment(
        _ProjectedRunner(FULL_RESULT_POLICY), request
    )
    fast = run_image_amplification_experiment(
        _ProjectedRunner(FAST_RESULT_POLICY), request
    )

    for name in (
        "image_transmission",
        "signal_carrier_mask",
        "input_signal_field",
        "output_signal_field",
        "backpropagated_signal_field",
        "zero_response_backpropagated_signal_field",
    ):
        np.testing.assert_array_equal(
            getattr(fast.result.analysis_result, name),
            getattr(full.result.analysis_result, name),
        )
    for name in (
        "measured_absolute_signal_gain",
        "analytic_absolute_signal_gain",
        "image_intensity_correlation",
        "normalized_image_rmse",
        "normalized_power_relative_drift",
    ):
        assert getattr(fast.result.analysis_result, name) == pytest.approx(
            getattr(full.result.analysis_result, name), rel=0.0, abs=0.0
        )
    for key in _SPECIALIZED_FIELDS:
        fast_field = fast.run_data.fields[key]
        full_field = full.run_data.fields[key]
        np.testing.assert_array_equal(fast_field.data, full_field.data)
        assert fast_field.axes == full_field.axes
        assert fast_field.units == full_field.units
        assert fast_field.kind == full_field.kind
        assert fast_field.quantity == full_field.quantity
        assert fast_field.default_display_extent == full_field.default_display_extent
        assert fast_field.coordinates.keys() == full_field.coordinates.keys()
        for coordinate in fast_field.coordinates:
            np.testing.assert_array_equal(
                fast_field.coordinates[coordinate],
                full_field.coordinates[coordinate],
            )
    fast_profile = fast.result.run_result.resolved_profile
    assert fast.result.run_result.A_initial is not None
    assert fast.result.run_result.A_final is not None
    assert fast_profile["material_response"]["model"] == (
        PR_MATERIAL_RESPONSE_LINEARIZED
    )
    assert fast_profile["reference_intensity"] == 1.5
    assert fast_profile["applied_field"] == 0.2
    assert fast_profile["requested_backend"] == "numpy"
    assert fast_profile["precision"] == "float64"


def test_linearized_ia_validation_rejects_missing_or_invalid_configuration():
    request = _experiment(strong=False, linearized=True)
    missing_i0 = replace(
        request,
        base_request=replace(
            request.base_request,
            material_response=PRTransverseMaterialResponseSpec(
                model=PR_MATERIAL_RESPONSE_LINEARIZED,
                reference_intensity=None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="reference_intensity"):
        _run(missing_i0)

    invalid_profile = replace(
        request,
        base_request=replace(
            request.base_request,
            boundary=PRTransverseBoundaryProfile(),
        ),
    )
    with pytest.raises(ValueError, match="profiles must match"):
        _run(invalid_profile)

    unsupported_td = replace(request, base_workflow_id=PR_TIMEDEPENDENT_WORKFLOW)
    with pytest.raises(TypeError, match="PRRunRequest"):
        prepare_image_amplification_base_request(unsupported_td)


class _FixedCanonicalRunner:
    def __init__(self, result, run_data):
        self.result = result
        self.run_data = run_data

    def run_registered(self, material_id, workflow_id, request, **kwargs):
        return RunnerResult(
            kind=workflow_id,
            result=self.result,
            run_data=self.run_data,
            material_id=material_id,
        )


def test_same_optical_endpoints_give_identical_ia_products_across_labels():
    nonlinear_request = _experiment(strong=False, linearized=False)
    linearized_request = _experiment(strong=False, linearized=True)
    canonical = LocalRunner((PR_TRANSVERSE_STATIC_OPERATION,)).run_registered(
        "pr",
        PR_TRANSVERSE_STATIC_WORKFLOW,
        prepare_image_amplification_base_request(nonlinear_request)[0],
    )
    nonlinear = run_image_amplification_experiment(
        _FixedCanonicalRunner(canonical.result, canonical.run_data),
        nonlinear_request,
    )
    relabeled_profile = dict(canonical.result.resolved_profile)
    relabeled_profile.update(
        {
            "physics_profile_id": PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            "material_response": {
                "model": PR_MATERIAL_RESPONSE_LINEARIZED,
                "reference_intensity": 1.5,
            },
            "reference_intensity": 1.5,
            "applied_field": 0.0,
            "validation_status": "experimental",
        }
    )
    relabeled_result = replace(
        canonical.result,
        resolved_profile=relabeled_profile,
    )
    relabeled_run_data = PR_TRANSVERSE_STATIC_OPERATION.to_run_data(
        relabeled_result
    )
    linearized = run_image_amplification_experiment(
        _FixedCanonicalRunner(relabeled_result, relabeled_run_data),
        linearized_request,
    )

    for name in (
        "signal_carrier_mask",
        "input_signal_field",
        "output_signal_field",
        "backpropagated_signal_field",
        "zero_response_backpropagated_signal_field",
    ):
        np.testing.assert_array_equal(
            getattr(nonlinear.result.analysis_result, name),
            getattr(linearized.result.analysis_result, name),
        )
    for key in _SPECIALIZED_FIELDS:
        np.testing.assert_array_equal(
            nonlinear.run_data.fields[key].data,
            linearized.run_data.fields[key].data,
        )
    for name in (
        "measured_absolute_signal_gain",
        "analytic_absolute_signal_gain",
        "image_intensity_correlation",
        "normalized_image_rmse",
        "normalized_power_relative_drift",
    ):
        nonlinear_value = getattr(nonlinear.result.analysis_result, name)
        linearized_value = getattr(linearized.result.analysis_result, name)
        assert np.isfinite(nonlinear_value)
        assert np.isfinite(linearized_value)
        assert nonlinear_value == linearized_value


def test_base_status_and_missing_endpoint_handling_remain_explicit():
    request = _experiment(strong=False, linearized=True)
    canonical = LocalRunner((PR_TRANSVERSE_STATIC_OPERATION,)).run_registered(
        "pr",
        PR_TRANSVERSE_STATIC_WORKFLOW,
        prepare_image_amplification_base_request(request)[0],
    )

    not_converged = replace(canonical.result, status="not_converged", converged=False)
    analyzed = run_image_amplification_experiment(
        _FixedCanonicalRunner(not_converged, canonical.run_data), request
    ).result
    assert analyzed.base_status == "not_converged"
    assert analyzed.analysis_status == "completed"
    assert analyzed.status == "not_converged"

    failed = replace(canonical.result, status="failed", converged=False)
    skipped = run_image_amplification_experiment(
        _FixedCanonicalRunner(failed, canonical.run_data), request
    ).result
    assert skipped.analysis_status == "not_run"
    assert skipped.analysis_result is None

    missing_endpoint = replace(canonical.result, A_final=None)
    missing = run_image_amplification_experiment(
        _FixedCanonicalRunner(missing_endpoint, canonical.run_data), request
    ).result
    assert missing.analysis_status == "failed"
    assert missing.analysis_result is None
    assert "field and mask" in missing.analysis_message
