from dataclasses import replace

import numpy as np

from lcprop.core.beams import BeamStack
from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import (
    from_static_result,
    from_timedependent_result,
    from_soliton_result,
    from_soliton_existence_result,
    to_run_data,
)
from lcprop.optics.splitstep import total_intensity
from lcprop.workflows import run_static, run_timedependent, run_soliton, run_soliton_existence
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.soliton_existence import SolitonExistenceRequest
from tests.test_all_workflows import make_base_static_request


def test_static_result_to_run_data():
    result = run_static(make_base_static_request())
    data = from_static_result(result)
    assert data.workflow == "static"
    image_fields = [field for field in data.fields.values() if field.data.ndim == 2]
    volume_fields = [field for field in data.fields.values() if field.data.ndim == 3]
    assert [field.display_name for field in image_fields] == [
        "Input Plane Intensity",
        "Output Plane Intensity",
        "Input Plane Δθ",
        "Output Plane Δθ",
        "Output Far-Field Intensity",
        "Output Far Field (dB relative to peak)",
    ]
    assert [field.display_name for field in volume_fields] == ["Intensity", "Δθ"]
    assert "theta" not in data.fields
    assert "theta_stack" not in data.fields

    expected_input = np.sum(np.abs(np.asarray(result.A_initial)) ** 2, axis=0)
    expected_output = np.sum(np.abs(np.asarray(result.A_final)) ** 2, axis=0)
    expected_delta = (
        np.asarray(result.theta_final)[None, :, :]
        - np.asarray(result.theta_bias)[None, :, :]
    )
    assert np.allclose(data.fields["input_intensity"].data, expected_input)
    assert np.allclose(data.fields["final_intensity"].data, expected_output)
    assert np.allclose(data.fields["input_delta_theta"].data, expected_delta[0])
    assert np.allclose(data.fields["output_delta_theta"].data, expected_delta[-1])
    assert np.allclose(data.fields["delta_theta_stack"].data, expected_delta)
    assert np.max(np.abs(np.diff(data.fields["intensity_stack"].data, axis=0))) > 0.0
    assert data.fields["input_intensity"].value_unit == "1/µm²"
    assert data.fields["intensity_stack"].value_unit == "1/µm²"
    assert data.fields["output_delta_theta"].value_unit == "rad"
    far_field = data.fields["far_field_intensity"]
    assert far_field.axes == ("s_x", "s_y")
    assert far_field.coordinates["s_x"].shape == (
        result.grid_summary["Nx"],
    )
    dsx = np.diff(far_field.coordinates["s_x"])[0]
    dsy = np.diff(far_field.coordinates["s_y"])[0]
    assert np.isclose(
        np.sum(far_field.data) * dsx * dsy,
        result.power_final,
        rtol=2e-14,
    )
    assert np.max(data.fields["far_field_log_db"].data) == 0.0
    summary = data.diagnostics["summary"].values
    assert summary["normalized_field_integral_initial"] == result.power_initial
    assert summary["physical_power_initial_mW"] == 0.05
    assert "power_initial" not in summary


def test_timedependent_result_to_run_data():
    base = make_base_static_request()
    result = run_timedependent(
        TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=1),
            output=base.output,
        )
    )
    data = from_timedependent_result(result)
    assert data.workflow == "timedependent"
    image_fields = [field for field in data.fields.values() if field.data.ndim == 2]
    volume_fields = [field for field in data.fields.values() if field.data.ndim == 3]
    expected_labels = [
        "Initial Intensity",
        "Final Intensity",
        "Initial Δθ",
        "Final Δθ",
    ]
    assert [field.display_name for field in image_fields] == [
        *expected_labels,
        "Output Far-Field Intensity",
        "Output Far Field (dB relative to peak)",
    ]
    assert [field.display_name for field in volume_fields] == [
        "Initial Intensity",
        "Final Intensity",
        "Initial Delta Theta",
        "Final Delta Theta",
    ]
    assert all(field.axes == ("z", "x", "y") for field in volume_fields)
    assert all(
        name not in [field.display_name for field in data.fields.values()]
        for name in ("Initial θ", "Final θ", "Initial TD Source Intensity")
    )

    initial_delta = np.asarray(result.theta_initial) - np.asarray(result.theta_bias)[None]
    final_delta = np.asarray(result.theta_final) - np.asarray(result.theta_bias)[None]
    selected_z = final_delta.shape[0] // 2
    assert np.allclose(data.fields["initial_delta_theta"].data, initial_delta[selected_z])
    assert np.allclose(data.fields["final_delta_theta"].data, final_delta[selected_z])
    assert np.allclose(data.fields["initial_delta_theta_stack"].data, initial_delta)
    assert np.allclose(data.fields["final_delta_theta_stack"].data, final_delta)
    assert np.array_equal(
        data.fields["initial_intensity_stack"].data,
        np.asarray(result.initial_intensity_stack),
    )
    assert np.array_equal(
        data.fields["final_intensity_stack"].data,
        np.asarray(result.final_intensity_stack),
    )
    assert data.fields["final_intensity"].value_unit == "1/µm²"
    assert data.fields["final_delta_theta_stack"].value_unit == "rad"
    assert data.fields["far_field_intensity"].coordinates["s_x"].shape == (
        base.grid.Nx,
    )
    summary = data.diagnostics["summary"].values
    assert summary["segment_start_time"] == 0.0
    assert summary["segment_elapsed_time"] == result.segment_elapsed_time
    assert summary["cumulative_time"] == result.cumulative_time
    assert summary["prior_completed_steps"] == 0
    assert summary["segment_completed_steps"] == 1
    assert summary["cumulative_completed_steps"] == 1
    assert list(data.curves.keys()) == [
        "beam_x_rms_width",
        "beam_y_rms_width",
    ]
    x_curve = data.curves["beam_x_rms_width"]
    y_curve = data.curves["beam_y_rms_width"]
    assert x_curve.display_name == "Beam x RMS width"
    assert y_curve.display_name == "Beam y RMS width"
    assert x_curve.x_label == "Cumulative TD time"
    assert x_curve.units["x RMS width"] == "µm"
    assert y_curve.units["y RMS width"] == "µm"
    np.testing.assert_allclose(x_curve.x, result.width_times)


def test_lc_far_field_preserves_coherent_group_semantics():
    base = make_base_static_request()
    channel = base.beams.channels[0]
    request = replace(
        base,
        beams=BeamStack(
            channels=(
                replace(
                    channel,
                    x0_um=-2.0,
                    tilt_x_rad_per_um=0.08,
                    coherence_group="shared",
                ),
                replace(
                    channel,
                    x0_um=2.0,
                    tilt_x_rad_per_um=-0.08,
                    phase_rad=0.3,
                    coherence_group="shared",
                ),
            ),
            coherence="coherent",
        ),
    )
    result = run_static(request)
    data = from_static_result(result)
    far_field = data.fields["far_field_intensity"]
    dsx = np.diff(far_field.coordinates["s_x"])[0]
    dsy = np.diff(far_field.coordinates["s_y"])[0]
    far_power = np.sum(far_field.data) * dsx * dsy
    near_intensity = total_intensity(
        np.asarray(result.A_final),
        coherence_groups=("shared", "shared"),
    )
    near_power = (
        np.sum(near_intensity)
        * result.grid_summary["dx_um"]
        * result.grid_summary["dy_um"]
    )

    assert np.isclose(far_power, near_power, rtol=2e-14)


def test_soliton_result_to_run_data():
    result = run_soliton(
        SolitonRequest(
            base=make_base_static_request(),
            max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    data = from_soliton_result(result)
    assert data.workflow == "soliton"
    assert "final_intensity" in data.fields
    assert "theta" in data.fields
    assert "far_field_intensity" in data.fields
    assert "far_field_log_db" in data.fields
    assert "summary" in data.diagnostics
    summary = data.diagnostics["summary"].values
    assert np.isfinite(summary["b"])
    assert np.isfinite(summary["bi"])
    assert np.isfinite(summary["beta"])
    assert summary["physical_power_mW"] == result.metrics["physical_power_mW"]
    base_grid = result.request.base.grid
    assert np.isclose(
        data.geometry.dx(),
        base_grid.x_aperture_um / base_grid.Nx,
    )
    assert np.isclose(
        data.geometry.dy(),
        base_grid.y_aperture_um / base_grid.Ny,
    )
    far_field = data.fields["far_field_intensity"]
    expected_sx = np.fft.fftshift(
        np.fft.fftfreq(base_grid.Nx, d=data.geometry.dx())
    ) * (result.request.base.beams.channels[0].wavelength_um / summary["n_ref"])
    expected_sy = np.fft.fftshift(
        np.fft.fftfreq(base_grid.Ny, d=data.geometry.dy())
    ) * (result.request.base.beams.channels[0].wavelength_um / summary["n_ref"])
    np.testing.assert_array_equal(far_field.coordinates["s_x"], expected_sx)
    np.testing.assert_array_equal(far_field.coordinates["s_y"], expected_sy)
    dsx = np.diff(expected_sx)[0]
    dsy = np.diff(expected_sy)[0]
    far_power = np.sum(far_field.data) * dsx * dsy
    near_power = (
        np.sum(np.asarray(result.intensity))
        * data.geometry.dx()
        * data.geometry.dy()
    )
    assert np.isclose(far_power, near_power, rtol=2e-14)


def test_soliton_existence_result_to_run_data():
    result = run_soliton_existence(
        SolitonExistenceRequest(
            base=make_base_static_request(),
            powers_mW=(0.03, 0.05),
            soliton_max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    data = from_soliton_existence_result(result)
    assert data.workflow == "soliton_existence"
    assert "beta" in data.curves
    assert "transverse_rms_widths" in data.curves
    assert len(data.curves["beta"].x) == 2
    widths = data.curves["transverse_rms_widths"]
    np.testing.assert_allclose(
        widths.y[:, 0],
        [row["sx_um"] for row in result.samples],
    )
    np.testing.assert_allclose(
        widths.y[:, 1],
        [row["sy_um"] for row in result.samples],
    )
    assert widths.units["RMS width"] == "µm"
    assert widths.series_labels == ("xs", "ys")
    assert list(data.fields.keys()) == [
        "existence_0_intensity",
        "existence_0_theta",
        "existence_1_intensity",
        "existence_1_theta",
    ]
    assert np.all(np.isfinite(data.curves["beta"].y))
    assert len(data.diagnostics["table"].values["rows"]) == 2


def test_to_run_data_dispatch():
    data = to_run_data(run_static(make_base_static_request()))
    assert data.workflow == "static"


def test_soliton_transverse_refinement_options_validate():
    req = SolitonRequest(
        base=make_base_static_request(),
        refine_transverse=True,
        transverse_max_outer=12,
        transverse_theta_steps_per_outer=7,
        transverse_field_mix=0.4,
        transverse_theta_mix=0.6,
    )

    req.validate()

    assert req.refine_transverse is True
    assert req.transverse_max_outer == 12
    assert req.transverse_theta_steps_per_outer == 7
