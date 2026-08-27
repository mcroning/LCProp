"""Bounded reduced-TD versus transverse-static image-amplification check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    ScreenPlacement,
)
from lcprop.pr.image_amplification import (
    PRImageAmplificationExperimentRequest,
    run_image_amplification_experiment,
)
from lcprop.pr.image_sources import PRImageSource
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)
from lcprop.runners.local import LocalRunner


def validation_requests() -> tuple[
    PRImageAmplificationExperimentRequest,
    PRImageAmplificationExperimentRequest,
]:
    """Return physically identical launches over the two base algorithms."""

    grid = GridSpec(
        Nx=24,
        Ny=24,
        x_aperture_um=40.0,
        y_aperture_um=40.0,
        z_length_um=10.0,
        dz_um=5.0,
    )
    material = PRMaterialSpec(
        dark_intensity=0.4,
        uniform_background_intensity=0.1,
        applied_field=0.0,
        gain_length_product=1.0e-3,
        characteristic_wavenumber_per_um_override=0.1,
    )
    backend = BackendSpec(backend="numpy", precision="float64", verbose=False)
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="pump",
                power_mW=1.0,
                waist_x_um=10.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=0.15,
                coherence_group="image-validation",
            ),
            BeamChannel(
                name="signal",
                power_mW=0.2,
                waist_x_um=10.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=-0.15,
                coherence_group="image-validation",
            ),
        ),
        coherence="coherent",
    )
    source = PRImageSource.from_array(np.eye(8))
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            center_x_um=0.0,
            center_y_um=0.0,
            width_um=8.0,
            height_um=8.0,
        ),
    )
    launch = LaunchConfiguration(
        beams=beams,
        channel_elements=(ChannelLaunchElements(1, (screen,)),),
    )
    common = {
        "launch_configuration": launch,
        "pump_channel_index": 0,
        "signal_channel_index": 1,
    }
    td_base = PRRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRSolverOptions(
            Nt=2,
            dt_normalized=0.01,
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        ),
        backend=backend,
    )
    static_base = PRTransverseStaticRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=8),
        backend=backend,
    )
    return (
        PRImageAmplificationExperimentRequest(
            base_workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
            base_request=td_base,
            **common,
        ),
        PRImageAmplificationExperimentRequest(
            base_workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
            base_request=static_base,
            **common,
        ),
    )


def _analysis_metrics(result) -> dict[str, float | str | bool | int]:
    analysis = result.result.analysis_result
    if analysis is None:
        raise RuntimeError(result.result.analysis_message)
    base = result.result.run_result
    metrics: dict[str, float | str | bool | int] = {
        "base_status": result.result.base_status,
        "composite_status": result.result.status,
        "analysis_status": result.result.analysis_status,
        "measured_absolute_signal_gain": float(
            analysis.measured_absolute_signal_gain
        ),
        "analytic_absolute_signal_gain": float(
            analysis.analytic_absolute_signal_gain
        ),
        "image_intensity_correlation": float(
            analysis.image_intensity_correlation
        ),
        "normalized_image_rmse": float(analysis.normalized_image_rmse),
        "normalized_power_relative_drift": float(
            analysis.normalized_power_relative_drift
        ),
        "incident_pump_power_mW": float(analysis.incident_channel_powers_mW[0]),
        "incident_signal_power_mW": float(
            analysis.incident_channel_powers_mW[1]
        ),
        "transmitted_signal_power_mW": float(
            analysis.post_element_channel_powers_mW[1]
        ),
        "pump_kx_rad_per_um": float(
            analysis.transverse_phase_gradients_rad_per_um[0]
        ),
        "signal_kx_rad_per_um": float(
            analysis.transverse_phase_gradients_rad_per_um[1]
        ),
        "workflow_runtime_s": float(analysis.pr_workflow_runtime_s),
        "total_runtime_s": float(analysis.runtime_s),
    }
    if hasattr(base, "converged"):
        metrics.update({
            "converged": bool(base.converged),
            "coupled_iterations": int(base.completed_coupled_iterations),
            "equilibrium_residual_rms": float(
                base.diagnostics["equilibrium_residual_rms"]
            ),
            "equilibrium_residual_max": float(
                base.diagnostics["equilibrium_residual_max"]
            ),
            "replay_field_match": bool(base.replay_diagnostics["field_match"]),
        })
    return metrics


def run_validation() -> dict[str, object]:
    td_request, static_request = validation_requests()
    runner = LocalRunner(
        operations=(PR_TIMEDEPENDENT_OPERATION, PR_TRANSVERSE_STATIC_OPERATION)
    )
    td = run_image_amplification_experiment(runner, td_request)
    static = run_image_amplification_experiment(runner, static_request)
    td_analysis = td.result.analysis_result
    static_analysis = static.result.analysis_result
    if td_analysis is None or static_analysis is None:
        raise RuntimeError("both image analyses must complete")
    td_image = np.abs(td_analysis.backpropagated_signal_field) ** 2
    static_image = np.abs(static_analysis.backpropagated_signal_field) ** 2
    denominator = float(np.linalg.norm(td_image))
    relative_image_l2 = float(
        np.linalg.norm(static_image - td_image) / denominator
    )
    return {
        "fixture": {
            "grid": [24, 24, 2],
            "aperture_um": [40.0, 40.0],
            "interaction_length_um": 10.0,
            "dz_um": 5.0,
            "powers_mW": [1.0, 0.2],
            "waists_um": [10.0, 10.0],
            "carrier_kx_rad_per_um": [0.15, -0.15],
            "image_pixels": [8, 8],
            "image_footprint_um": [8.0, 8.0],
            "dark_intensity": 0.4,
            "uniform_background_intensity": 0.1,
            "gain_length_product": 1.0e-3,
            "backend": "numpy",
            "precision": "float64",
        },
        "launch_invariants": {
            "same_beams": td_request.launch_configuration.beams
            == static_request.launch_configuration.beams,
            "same_elements": td_request.launch_configuration.channel_elements
            == static_request.launch_configuration.channel_elements,
        },
        "reduced_td": _analysis_metrics(td),
        "transverse_static": _analysis_metrics(static),
        "comparison": {
            "reconstructed_intensity_relative_l2_static_vs_td": (
                relative_image_l2
            ),
            "same_carrier_mask": bool(np.array_equal(
                td_analysis.signal_carrier_mask,
                static_analysis.signal_carrier_mask,
            )),
            "same_image_transmission": bool(np.array_equal(
                td_analysis.image_transmission,
                static_analysis.image_transmission,
            )),
            "same_incident_channel_powers": (
                td_analysis.incident_channel_powers_mW
                == static_analysis.incident_channel_powers_mW
            ),
            "same_post_element_channel_powers": (
                td_analysis.post_element_channel_powers_mW
                == static_analysis.post_element_channel_powers_mW
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_validation()
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
