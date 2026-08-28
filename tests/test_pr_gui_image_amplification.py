from __future__ import annotations

from dataclasses import replace
import hashlib
import time

import numpy as np
import pytest
import lcprop.pr.image_amplification as image_amplification_module

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.gui.views.image_pane import ImagePane
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    ScreenPlacement,
)
from lcprop.pr.gui.image_input_panel import (
    PR_IMAGE_AMPLIFICATION_INPUT_MODE,
    PRImageInputPanel,
    decode_user_image,
)
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.image_amplification import (
    PR_IMAGE_AMPLIFICATION_WORKFLOW,
    PRBeamPanelImageAmplificationRunRequest,
    PRImageAmplificationCompositeResult,
    PRImageAmplificationExperimentRequest,
    PRImageAmplificationRunRequest,
    PRImageAmplificationSpec,
    PRImageLaunchSpec,
    apply_passive_field_transmittance,
    image_amplification_run_request,
    image_amplification_base_capabilities,
    image_amplification_experiment_request,
    intensity_transmission_to_field_transmittance,
    prepare_image_amplification_workflow_request,
    prepare_image_amplification_base_request,
    prepare_image_transmission,
    run_image_amplification,
    run_image_amplification_request,
    run_image_amplification_experiment,
)
from lcprop.pr.image_sources import PRImageSource, standard_image_catalog
from lcprop.pr.operations import (
    PR_IMAGE_AMPLIFICATION_OPERATION,
    PR_STATIC_OPERATION,
    PR_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.products import (
    _image_amplification_default_display_extent,
    pr_image_amplification_result_to_run_data,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticRunResult,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.specs import (
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.operations import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
)
from lcprop.runners.base import RunnerResult
from lcprop.runners.local import LocalRunner
from lcprop.transport.defaults import default_transport_registry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write_rgba(path, *, width=7, height=4):
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    for y in range(height):
        for x in range(width):
            image.setPixelColor(
                x,
                y,
                QColor((37 * x) % 256, (53 * y) % 256, (19 * (x + y)) % 256, 7),
            )
    assert image.save(str(path), "PNG")
    return image


def _composite(
    source,
    *,
    grid=None,
    pump_power_mW=3.0,
    signal_power_mW=0.5,
    image_size_um=8.0,
):
    return PRImageAmplificationRunRequest(
        grid=grid
        or GridSpec(
            Nx=32,
            Ny=8,
            x_aperture_um=64.0,
            y_aperture_um=16.0,
            z_length_um=20.0,
            dz_um=10.0,
        ),
        material=PRMaterialSpec(gain_length_product=-0.2),
        solver=PRSolverOptions(Nt=0, dt_normalized=0.01),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
        source=source,
        launch=PRImageLaunchSpec(
            positive_mode_index=2,
            beam_waist_x_um=10.0,
            beam_waist_y_um=6.0,
            image_physical_size_um=image_size_um,
            pump_incident_power_mW=pump_power_mW,
            signal_incident_power_mW=signal_power_mW,
            require_full_footprint=True,
        ),
    )


def test_empty_catalog_is_explicit_until_asset_provenance_is_approved():
    assert standard_image_catalog() == ()


def test_qt_decode_is_deterministic_pillow_compatible_and_ignores_alpha(tmp_path):
    path = tmp_path / "rgba.png"
    original = _write_rgba(path)

    decoded = decode_user_image(path)
    pixels = decoded.source.grayscale

    assert decoded.source.basename == "rgba.png"
    assert decoded.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (decoded.source.width, decoded.source.height) == (7, 4)
    assert decoded.source.encoded_format == "png"
    assert not pixels.flags.writeable
    rgba = np.empty((4, 7, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(7):
            color = original.pixelColor(x, y)
            rgba[y, x] = (color.red(), color.green(), color.blue(), color.alpha())
    expected = (
        19595 * rgba[..., 0].astype(np.uint32)
        + 38470 * rgba[..., 1].astype(np.uint32)
        + 7471 * rgba[..., 2].astype(np.uint32)
        + 32768
    ) >> 16
    assert np.array_equal(pixels, expected.astype(np.uint8))

    pillow = pytest.importorskip("PIL.Image")
    assert np.array_equal(pixels, np.asarray(pillow.open(path).convert("L")))


def test_user_decode_rejects_corrupt_raster(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="unsupported image format|decode"):
        decode_user_image(path)


def test_source_resolution_never_changes_grid_or_incident_power():
    low = np.array([[0.0, 1.0], [1.0, 0.0]])
    high = np.repeat(np.repeat(low, 40, axis=0), 40, axis=1)
    low_request = _composite(PRImageSource.from_array(low))
    high_request = _composite(PRImageSource.from_array(high))

    low_workflow, low_transmission, _ = prepare_image_amplification_workflow_request(
        low_request
    )
    high_workflow, high_transmission, _ = prepare_image_amplification_workflow_request(
        high_request
    )

    assert low_request.grid == high_request.grid
    assert low_request.grid.Nx == 32 and low_request.grid.Ny == 8
    assert low_transmission.shape == high_transmission.shape == (32, 8)
    assert np.array_equal(low_transmission, high_transmission)
    for workflow in (low_workflow, high_workflow):
        dxdy = workflow.grid.x_aperture_um / workflow.grid.Nx
        dxdy *= workflow.grid.y_aperture_um / workflow.grid.Ny
        integrals = np.sum(np.abs(workflow.initial_A) ** 2, axis=(1, 2)) * dxdy
        assert integrals[0] * 3.5 == pytest.approx(3.0)
        assert integrals[1] * 3.5 < 0.5
        assert sum(
            channel.power_mW for channel in workflow.beams.channels
        ) == pytest.approx(3.5)


def test_passive_transmittance_power_semantics():
    field = np.array(
        [[1.0 + 2.0j, -0.5j], [0.25 - 0.75j, -2.0 + 0.5j]],
        dtype=np.complex128,
    )
    incident_power = float(np.sum(np.abs(field) ** 2))

    identity = apply_passive_field_transmittance(field, np.ones(field.shape))
    assert np.array_equal(identity, field)
    assert np.sum(np.abs(identity) ** 2) == pytest.approx(incident_power)

    uniform_t = intensity_transmission_to_field_transmittance(
        np.full(field.shape, 0.25)
    )
    uniform = apply_passive_field_transmittance(field, uniform_t)
    assert np.array_equal(uniform, 0.5 * field)
    assert np.sum(np.abs(uniform) ** 2) == pytest.approx(0.25 * incident_power)

    opaque = apply_passive_field_transmittance(field, np.zeros(field.shape))
    assert np.count_nonzero(opaque) == 0
    assert np.sum(np.abs(opaque) ** 2) == 0.0

    nonuniform_T = np.array([[0.0, 0.25], [0.5, 1.0]])
    nonuniform = apply_passive_field_transmittance(
        field,
        intensity_transmission_to_field_transmittance(nonuniform_T),
    )
    expected = float(np.sum(nonuniform_T * np.abs(field) ** 2))
    assert np.sum(np.abs(nonuniform) ** 2) == pytest.approx(expected)


def test_phase_and_general_complex_transmittance_semantics():
    field = np.array(
        [[1.0 + 2.0j, -0.5j], [0.25 - 0.75j, -2.0 + 0.5j]],
        dtype=np.complex128,
    )
    phase = np.array([[0.0, 0.4], [-1.2, 2.1]])
    phase_t = np.exp(1j * phase)
    phase_output = apply_passive_field_transmittance(field, phase_t)
    assert not np.array_equal(phase_output, field)
    assert np.allclose(np.abs(phase_output) ** 2, np.abs(field) ** 2)
    assert np.sum(np.abs(phase_output) ** 2) == pytest.approx(
        np.sum(np.abs(field) ** 2), rel=2e-15
    )

    amplitude = np.array([[0.0, 0.25], [0.7, 1.0]])
    general_t = amplitude * phase_t
    general_output = apply_passive_field_transmittance(field, general_t)
    assert np.array_equal(general_output, general_t * field)
    assert np.sum(np.abs(general_output) ** 2) == pytest.approx(
        np.sum(amplitude**2 * np.abs(field) ** 2)
    )
    with pytest.raises(ValueError, match="magnitude must not exceed one"):
        apply_passive_field_transmittance(field, np.full(field.shape, 1.01))


def test_image_element_changes_only_signal_and_does_not_restore_absorption():
    open_request = _composite(PRImageSource.from_array(np.ones((8, 8))))
    absorbing_source = np.ones((8, 8))
    absorbing_source[2:6, 2:6] = 0.0
    absorbing_request = _composite(PRImageSource.from_array(absorbing_source))

    open_workflow, _open_T, _ = prepare_image_amplification_workflow_request(
        open_request
    )
    absorbing_workflow, _absorbing_T, _ = (
        prepare_image_amplification_workflow_request(absorbing_request)
    )
    dxdy = 2.0 * 2.0
    open_integrals = (
        np.sum(np.abs(open_workflow.initial_A) ** 2, axis=(1, 2)) * dxdy
    )
    absorbing_integrals = (
        np.sum(np.abs(absorbing_workflow.initial_A) ** 2, axis=(1, 2)) * dxdy
    )

    assert open_integrals * 3.5 == pytest.approx(np.array([3.0, 0.5]))
    assert np.array_equal(
        open_workflow.initial_A[0], absorbing_workflow.initial_A[0]
    )
    assert absorbing_integrals[0] == pytest.approx(open_integrals[0])
    assert absorbing_integrals[1] < open_integrals[1]
    assert np.sum(absorbing_integrals) < np.sum(open_integrals)


def test_megapixel_source_is_resampled_without_changing_small_grid():
    source = PRImageSource(
        source_kind="user",
        display_name="large.png",
        basename="large.png",
        sha256="1" * 64,
        width=2048,
        height=512,
        encoded_format="png",
        decoded_mode="L",
        grayscale=np.ones((512, 2048), dtype=np.uint8),
    )
    request = _composite(source)

    workflow, transmission, _grating = prepare_image_amplification_workflow_request(
        request
    )

    assert source.grayscale.shape == (512, 2048)
    assert workflow.grid == request.grid
    assert workflow.initial_A.shape == (2, 32, 8)
    assert transmission.shape == (32, 8)


def test_paper_scale_source_dimensions_do_not_select_the_grid():
    source = PRImageSource(
        source_kind="user",
        display_name="paper-scale.png",
        basename="paper-scale.png",
        sha256="2" * 64,
        width=16384,
        height=4096,
        encoded_format="png",
        decoded_mode="L",
        grayscale=np.zeros((4096, 16384), dtype=np.uint8),
    )
    request = _composite(source)

    assert source.grayscale.shape == (4096, 16384)
    assert request.grid.Nx == 32
    assert request.grid.Ny == 8
    assert request.grid.x_aperture_um == 64.0
    assert request.grid.y_aperture_um == 16.0
    assert request.grid.z_length_um == 20.0
    assert request.grid.dz_um == 10.0


def test_gui_full_footprint_policy_rejects_silent_clipping():
    source = PRImageSource.from_array(np.eye(4))
    request = _composite(source, image_size_um=40.0)
    with pytest.raises(ValueError, match="footprint extends outside"):
        prepare_image_amplification_workflow_request(request)


def test_full_footprint_policy_checks_the_discrete_destination():
    grid = make_grid(
        GridSpec(
            Nx=8,
            Ny=8,
            dz_um=1.0,
            x_aperture_um=8.0,
            y_aperture_um=8.0,
            z_length_um=1.0,
        ),
        real_dtype=np.float64,
    )
    with pytest.raises(ValueError, match="footprint extends outside"):
        prepare_image_transmission(
            np.eye(2),
            grid,
            center_x_um=3.5,
            center_y_um=0.0,
            physical_size_um=4.0,
            require_full_footprint=True,
        )


def test_physical_operation_is_distinct_from_historical_normalized_benchmark():
    image = np.eye(8)
    spec = replace(PRImageAmplificationSpec(), Nt=1, dt_normalized=0.01)
    historical = run_image_amplification(image, spec)
    composite_request = image_amplification_run_request(image, spec)
    physical = PR_IMAGE_AMPLIFICATION_OPERATION.run(composite_request)

    assert np.array_equal(
        physical.image_transmission,
        historical.image_transmission,
    )
    dxdy = 1.0
    historical_power = np.sum(np.abs(historical.request.initial_A) ** 2) * dxdy
    physical_power = np.sum(np.abs(physical.request.initial_A) ** 2) * dxdy
    assert historical_power == pytest.approx(1.0)
    assert physical_power < historical_power
    assert historical.transparency_policy == "historical_normalized_benchmark"
    assert physical.transparency_policy == "passive_intensity_transmission_v1"
    assert physical.signal_throughput_fraction < 1.0


def test_rectangular_result_products_have_independent_axes():
    result = run_image_amplification_request(
        _composite(PRImageSource.from_array(np.eye(8)))
    )
    run_data = PR_IMAGE_AMPLIFICATION_OPERATION.to_run_data(result)

    assert run_data.workflow == PR_IMAGE_AMPLIFICATION_WORKFLOW
    assert run_data.fields["image_transmission"].data.shape == (32, 8)
    far_field = run_data.fields["far_field_intensity"]
    assert far_field.data.shape == (32, 8)
    assert far_field.coordinates["s_x"].shape == (32,)
    assert far_field.coordinates["s_y"].shape == (8,)
    carrier_mask = run_data.fields["signal_carrier_mask"]
    assert carrier_mask.axes == ("s_x", "s_y")
    assert np.array_equal(
        carrier_mask.coordinates["s_x"], far_field.coordinates["s_x"]
    )
    assert np.array_equal(
        carrier_mask.coordinates["s_y"], far_field.coordinates["s_y"]
    )
    assert np.array_equal(
        carrier_mask.data,
        np.fft.fftshift(result.signal_carrier_mask).astype(float),
    )
    values = run_data.diagnostics["image_amplification"].values
    assert values["incident_pump_power_mW"] == pytest.approx(3.0)
    assert values["incident_signal_power_mW"] == pytest.approx(0.5)
    assert values["incident_total_power_mW"] == pytest.approx(3.5)
    assert values["post_element_pump_power_mW"] == pytest.approx(3.0)
    assert values["post_element_signal_power_mW"] < 0.5
    assert values["power_entering_pr_medium_mW"] < 3.5
    assert values["signal_throughput_fraction"] < 1.0
    assert values["transparency_policy"] == "passive_intensity_transmission_v1"
    metrics = run_data.diagnostics["image_amplification_metrics"].values
    assert metrics["measured_signal_gain"] == result.measured_absolute_signal_gain
    assert metrics["analytic_signal_gain"] == result.analytic_absolute_signal_gain
    assert metrics["image_correlation"] == result.image_intensity_correlation
    assert metrics["normalized_image_rmse"] == result.normalized_image_rmse
    assert metrics["incident_signal_power_mW"] == result.incident_channel_powers_mW[1]
    assert metrics["post_screen_signal_power_mW"] == (
        result.post_element_channel_powers_mW[1]
    )
    assert metrics["signal_screen_throughput"] == result.signal_throughput_fraction
    assert metrics["output_isolated_signal_power_mW"] == (
        result.output_isolated_signal_power_mW
    )
    assert metrics["total_power_entering_pr_medium_mW"] == (
        result.post_element_total_power_mW
    )
    assert metrics["normalized_optical_power_drift"] == (
        result.normalized_power_relative_drift
    )
    assert metrics["measured_gain_vs_z_available"] is False
    assert "no per-z complex optical-field history" in (
        metrics["measured_gain_vs_z_reason"]
    )
    assert run_data.geometry.x.shape == (32,)
    assert run_data.geometry.y.shape == (8,)


def test_image_products_default_to_offset_rectangular_signal_screen_frame(app):
    experiment = _presentation_image_experiment()
    composite = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        experiment,
    )
    analysis = composite.result.analysis_result
    assert analysis is not None
    run_data = composite.run_data
    base_run_data = composite.result.base_runner_result.run_data
    expected = (6.55, 36.45, -17.8, -2.2)
    full_extent = tuple(run_data.geometry.extent_xy())
    assert np.array_equal(run_data.geometry.x, base_run_data.geometry.x)
    assert np.array_equal(run_data.geometry.y, base_run_data.geometry.y)

    for key, source_values in (
        ("image_transmission", analysis.image_transmission),
        ("input_signal_intensity", np.abs(analysis.input_signal_field) ** 2),
        ("output_signal_intensity", np.abs(analysis.output_signal_field) ** 2),
        ("amplified_image", np.abs(analysis.backpropagated_signal_field) ** 2),
        (
            "zero_response_image",
            np.abs(analysis.zero_response_backpropagated_signal_field) ** 2,
        ),
    ):
        field = run_data.fields[key]
        assert field.default_display_extent == pytest.approx(expected)
        assert np.array_equal(field.data, source_values)
        assert field.default_display_extent[0] >= full_extent[0]
        assert field.default_display_extent[1] <= full_extent[1]
        assert field.default_display_extent[2] >= full_extent[2]
        assert field.default_display_extent[3] <= full_extent[3]
        assert field.axes == ("x", "y")

    pane = ImagePane()
    pane.set_run_data(run_data)
    assert pane.field_selector.currentData() == "amplified_image"
    assert tuple(pane.image_view.image.get_extent()) == pytest.approx(full_extent)
    assert pane.image_view.ax.get_xlim() == pytest.approx(expected[:2])
    assert pane.image_view.ax.get_ylim() == pytest.approx(expected[2:])
    pane.image_view.ax.set_xlim(10.0, 20.0)
    pane.image_view.ax.set_ylim(-15.0, -5.0)
    pane.fit_button.click()
    assert pane.image_view.ax.get_xlim() == pytest.approx(expected[:2])
    assert pane.image_view.ax.get_ylim() == pytest.approx(expected[2:])
    pane.full_aperture_button.click()
    assert pane.image_view.ax.get_xlim() == pytest.approx(full_extent[:2])
    assert pane.image_view.ax.get_ylim() == pytest.approx(full_extent[2:])
    pane.close()


def test_image_default_frame_supports_screen_centered_on_signal():
    experiment = _presentation_image_experiment(
        screen_center_x_um=20.0,
        screen_center_y_um=-10.0,
    )
    composite = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        experiment,
    )

    assert composite.run_data.fields[
        "amplified_image"
    ].default_display_extent == pytest.approx((7.0, 33.0, -17.8, -2.2))


@pytest.mark.parametrize(
    ("kwargs", "expected_x"),
    (
        (
            {"screen_center_x_um": 23.0, "screen_width_um": 40.0},
            (-3.0, 49.0),
        ),
        (
            {"signal_waist_x_um": 12.0, "screen_width_um": 8.0},
            (-11.2, 51.2),
        ),
    ),
)
def test_image_default_frame_uses_larger_of_screen_and_beam(kwargs, expected_x):
    experiment = _presentation_image_experiment(**kwargs)
    composite = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        experiment,
    )
    extent = composite.run_data.fields["amplified_image"].default_display_extent

    assert extent is not None
    assert extent[:2] == pytest.approx(expected_x)


def test_image_default_frame_clamps_to_aperture_and_has_safe_fallback():
    experiment = _presentation_image_experiment(
        screen_center_x_um=55.0,
        screen_width_um=8.0,
    )
    signal = replace(
        experiment.launch_configuration.beams.channels[1],
        x0_um=55.0,
    )
    beams = replace(
        experiment.launch_configuration.beams,
        channels=(
            experiment.launch_configuration.beams.channels[0],
            signal,
        ),
    )
    launch = replace(experiment.launch_configuration, beams=beams)
    experiment = replace(
        experiment,
        base_request=replace(experiment.base_request, beams=beams),
        launch_configuration=launch,
    )
    composite = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        experiment,
    )
    analysis = composite.result.analysis_result
    assert analysis is not None
    extent = composite.run_data.fields["amplified_image"].default_display_extent
    full_extent = tuple(composite.run_data.geometry.extent_xy())

    assert extent is not None
    assert extent[1] == full_extent[1]
    assert _image_amplification_default_display_extent(
        replace(analysis, image_request=object()),
        composite.result.base_runner_result.run_data,
    ) is None


def test_measured_gain_power_diagnostics_use_existing_carrier_isolation():
    experiment = _presentation_image_experiment()
    composite = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        experiment,
    )
    analysis = composite.result.analysis_result
    assert analysis is not None
    metrics = composite.run_data.diagnostics[
        "image_amplification_metrics"
    ].values

    assert analysis.measured_absolute_signal_gain == pytest.approx(
        analysis.output_isolated_signal_power_normalized
        / analysis.measured_gain_reference_signal_power_normalized,
        rel=0.0,
        abs=1.0e-15,
    )
    assert analysis.measured_absolute_signal_gain == pytest.approx(
        analysis.output_isolated_signal_power_mW
        / analysis.measured_gain_reference_signal_power_mW,
        rel=0.0,
        abs=1.0e-15,
    )
    assert metrics["measured_gain_reference_signal_power_mW"] == (
        analysis.measured_gain_reference_signal_power_mW
    )
    assert metrics["output_isolated_signal_power_mW"] == (
        analysis.output_isolated_signal_power_mW
    )
    assert metrics["measured_gain_reference"] == (
        "carrier-isolated post-screen field at z=0"
    )
    assert tuple(composite.run_data.curves) == tuple(
        composite.result.base_runner_result.run_data.curves
    )


def test_gui_user_mode_builds_and_dispatches_registered_operation(app, tmp_path):
    path = tmp_path / "source.png"
    _write_rgba(path, width=8, height=4)
    window = PRMainWindow()
    mode_index = window.input_panel.input_mode.findData(
        PR_IMAGE_AMPLIFICATION_INPUT_MODE
    )
    window.input_panel.input_mode.setCurrentIndex(mode_index)
    window.grid_panel.Nx.setValue(32)
    window.grid_panel.Ny.setValue(8)
    window.grid_panel.x_aperture_um.setValue(64.0)
    window.grid_panel.y_aperture_um.setValue(16.0)
    window.grid_panel.z_length_um.setValue(20.0)
    window.grid_panel.dz_um.setValue(10.0)
    window.evolution_panel.Nt.setValue(0)
    kx = 2.0 * np.pi * 2 / 64.0
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="pump",
                    power_mW=2.0,
                    waist_x_um=10.0,
                    waist_y_um=6.0,
                    tilt_x_rad_per_um=kx,
                    coherence_group="image-laser",
                ),
                BeamDefinition(
                    name="signal",
                    power_mW=0.25,
                    waist_x_um=10.0,
                    waist_y_um=6.0,
                    tilt_x_rad_per_um=-kx,
                    coherence_group="image-laser",
                ),
            )
        )
    )
    editor = window.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.load_user_image(path)
    editor.width_um.setValue(8.0)
    editor.height_um.setValue(8.0)

    request = window.build_request()
    before = window.grid_panel.grid()
    calls = []
    original_run = window.local_runner.run_registered

    def observed(material_id, workflow_id, supplied_request, **kwargs):
        calls.append((material_id, workflow_id, supplied_request))
        return original_run(material_id, workflow_id, supplied_request, **kwargs)

    window.local_runner.run_registered = observed
    window.run_button.click()
    deadline = time.monotonic() + 8.0
    while window._background_running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    app.processEvents()

    assert isinstance(request, PRImageAmplificationExperimentRequest)
    assert isinstance(request.base_request, PRRunRequest)
    assert request.source.grayscale.shape == (4, 8)
    assert request.incident_signal_to_pump_power_ratio == pytest.approx(0.125)
    assert window.grid_panel.grid() == before == request.grid
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.beam_panel))
    assert window.evolution_panel.workflow.isEnabled()
    assert not hasattr(window.input_panel, "preview")
    assert not window._background_running
    assert calls and calls[0][:2] == ("pr", PR_TIMEDEPENDENT_WORKFLOW)
    assert isinstance(calls[0][2], PRRunRequest)
    assert calls[0][2].initial_A is None
    assert (
        calls[0][2].launch_elements
        == request.launch_configuration.channel_elements
    )
    assert window.last_runner_result.kind == PR_IMAGE_AMPLIFICATION_WORKFLOW
    assert "amplified_image" in window.last_runner_result.run_data.fields
    assert "image_amplification" in window.last_runner_result.run_data.diagnostics
    saved_path = tmp_path / "image.lcprop.json"
    window.save_experiment_to(saved_path)
    assert saved_path.is_file()
    window.close()


def test_beampanel_request_matches_stage_a_prepared_launch_bit_for_bit():
    source = PRImageSource.from_array(np.eye(8))
    legacy = _composite(source, pump_power_mW=3.0, signal_power_mW=0.5)
    old_prepared, old_transmission, old_grating = (
        prepare_image_amplification_workflow_request(legacy)
    )
    signal = old_prepared.beams.channels[1]
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            center_x_um=signal.x0_um,
            center_y_um=signal.y0_um,
            width_um=legacy.launch.image_physical_size_um,
            height_um=legacy.launch.image_physical_size_um,
            boundary_policy="reject",
        ),
    )
    request = PRBeamPanelImageAmplificationRunRequest(
        grid=legacy.grid,
        material=legacy.material,
        solver=legacy.solver,
        backend=legacy.backend,
        launch_configuration=LaunchConfiguration(
            beams=old_prepared.beams,
            channel_elements=(ChannelLaunchElements(1, (screen,)),),
        ),
        pump_channel_index=0,
        signal_channel_index=1,
    )
    new_prepared, new_transmission, new_grating = (
        prepare_image_amplification_workflow_request(request)
    )

    assert new_prepared.beams == old_prepared.beams
    assert np.array_equal(new_transmission, old_transmission)
    assert new_grating == old_grating
    assert np.array_equal(new_prepared.initial_A, old_prepared.initial_A)


def test_role_validation_rejects_ambiguous_or_incompatible_launches():
    source = PRImageSource.from_array(np.eye(4))
    legacy = _composite(source)
    prepared, _transmission, _grating = prepare_image_amplification_workflow_request(
        legacy
    )
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(width_um=4.0, height_um=4.0),
    )
    launch = LaunchConfiguration(
        beams=prepared.beams,
        channel_elements=(ChannelLaunchElements(1, (screen,)),),
    )
    common = dict(
        grid=legacy.grid,
        material=legacy.material,
        solver=legacy.solver,
        backend=legacy.backend,
        launch_configuration=launch,
    )
    with pytest.raises(ValueError, match="must be distinct"):
        PRBeamPanelImageAmplificationRunRequest(
            **common, pump_channel_index=1, signal_channel_index=1
        ).validate()
    with pytest.raises(ValueError, match="exactly one IntensityRasterScreen"):
        PRBeamPanelImageAmplificationRunRequest(
            **{**common, "launch_configuration": LaunchConfiguration(prepared.beams)},
            pump_channel_index=0,
            signal_channel_index=1,
        ).validate()


def _valid_beampanel_image_amplification_request():
    source = PRImageSource.from_array(np.eye(4))
    legacy = _composite(source)
    prepared, _transmission, _grating = prepare_image_amplification_workflow_request(
        legacy
    )
    signal = prepared.beams.channels[1]
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            center_x_um=signal.x0_um,
            center_y_um=signal.y0_um,
            width_um=4.0,
            height_um=4.0,
        ),
    )
    return PRBeamPanelImageAmplificationRunRequest(
        grid=legacy.grid,
        material=legacy.material,
        solver=legacy.solver,
        backend=legacy.backend,
        launch_configuration=LaunchConfiguration(
            prepared.beams,
            (ChannelLaunchElements(1, (screen,)),),
        ),
        pump_channel_index=0,
        signal_channel_index=1,
    )


def _presentation_image_experiment(
    *,
    signal_waist_x_um=5.0,
    signal_waist_y_um=3.0,
    screen_center_x_um=23.0,
    screen_center_y_um=-8.0,
    screen_width_um=20.0,
    screen_height_um=8.0,
):
    grid = GridSpec(
        Nx=64,
        Ny=48,
        x_aperture_um=120.0,
        y_aperture_um=80.0,
        z_length_um=10.0,
        dz_um=5.0,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="pump",
                power_mW=1.0,
                waist_x_um=5.0,
                waist_y_um=3.0,
                x0_um=-20.0,
                y0_um=10.0,
                tilt_x_rad_per_um=0.15,
                coherence_group="image-presentation",
            ),
            BeamChannel(
                name="signal",
                power_mW=0.2,
                waist_x_um=signal_waist_x_um,
                waist_y_um=signal_waist_y_um,
                x0_um=20.0,
                y0_um=-10.0,
                tilt_x_rad_per_um=-0.15,
                coherence_group="image-presentation",
            ),
        ),
        coherence="coherent",
    )
    screen = IntensityRasterScreen(
        PRImageSource.from_array(np.eye(8)),
        ScreenPlacement(
            center_x_um=screen_center_x_um,
            center_y_um=screen_center_y_um,
            width_um=screen_width_um,
            height_um=screen_height_um,
            boundary_policy="reject",
        ),
    )
    launch = LaunchConfiguration(
        beams,
        (ChannelLaunchElements(1, (screen,)),),
    )
    base = PRRunRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(gain_length_product=0.0),
        solver=PRSolverOptions(Nt=0, dt_normalized=0.01),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )
    return PRImageAmplificationExperimentRequest(
        base_workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
        base_request=base,
        launch_configuration=launch,
        pump_channel_index=0,
        signal_channel_index=1,
    )


def _configured_multi_algorithm_image_window(app):
    window = PRMainWindow()
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(
            PR_IMAGE_AMPLIFICATION_INPUT_MODE
        )
    )
    window.grid_panel.Nx.setValue(24)
    window.grid_panel.Ny.setValue(24)
    window.grid_panel.x_aperture_um.setValue(40.0)
    window.grid_panel.y_aperture_um.setValue(40.0)
    window.grid_panel.z_length_um.setValue(10.0)
    window.grid_panel.dz_um.setValue(5.0)
    window.material_panel.dark_intensity.setValue(0.4)
    window.material_panel.uniform_background_intensity.setValue(0.1)
    window.material_panel.applied_field.setValue(0.0)
    window.material_panel.gain_length_product.setValue(1.0e-3)
    window.material_panel.use_characteristic_wavenumber_override.setChecked(True)
    window.material_panel.characteristic_wavenumber_per_um_override.setValue(0.1)
    window.evolution_panel.Nt.setValue(2)
    window.evolution_panel.dt_normalized.setValue(0.01)
    window.evolution_panel.max_coupled_passes.setValue(8)
    window.evolution_panel.backend.setCurrentText("numpy")
    window.evolution_panel.precision.setCurrentText("float64")
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="pump",
                    power_mW=1.0,
                    waist_x_um=10.0,
                    waist_y_um=10.0,
                    tilt_x_rad_per_um=0.15,
                    coherence_group="image-validation",
                ),
                BeamDefinition(
                    name="signal",
                    power_mW=0.2,
                    waist_x_um=10.0,
                    waist_y_um=10.0,
                    tilt_x_rad_per_um=-0.15,
                    coherence_group="image-validation",
                ),
            )
        )
    )
    editor = window.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.set_source(PRImageSource.from_array(np.eye(8)))
    editor.width_um.setValue(8.0)
    editor.height_um.setValue(8.0)
    window.input_panel.sync_channels()
    return window


@pytest.mark.parametrize("channel_count", [1, 3])
def test_beampanel_request_rejects_non_two_channel_launches(channel_count):
    request = _valid_beampanel_image_amplification_request()
    channels = request.launch_configuration.beams.channels
    if channel_count == 1:
        selected = channels[:1]
        elements = ()
    else:
        selected = channels + (replace(channels[1], name="extra"),)
        elements = request.launch_configuration.channel_elements
    launch = LaunchConfiguration(
        replace(request.launch_configuration.beams, channels=selected),
        elements,
    )

    with pytest.raises(ValueError, match="exactly two enabled beam channels"):
        replace(request, launch_configuration=launch).validate()


def test_beampanel_request_rejects_different_coherence_groups():
    request = _valid_beampanel_image_amplification_request()
    pump, signal = request.launch_configuration.beams.channels
    beams = replace(
        request.launch_configuration.beams,
        channels=(pump, replace(signal, coherence_group="other-laser")),
    )

    with pytest.raises(ValueError, match="share one coherence group"):
        replace(
            request,
            launch_configuration=replace(
                request.launch_configuration,
                beams=beams,
            ),
        ).validate()


def test_beampanel_request_rejects_different_wavelengths():
    request = _valid_beampanel_image_amplification_request()
    pump, signal = request.launch_configuration.beams.channels
    beams = replace(
        request.launch_configuration.beams,
        channels=(pump, replace(signal, wavelength_um=0.532)),
    )

    with pytest.raises(ValueError, match="wavelengths must match"):
        replace(
            request,
            launch_configuration=replace(
                request.launch_configuration,
                beams=beams,
            ),
        ).validate()


def test_beampanel_request_rejects_nonzero_y_carrier():
    request = _valid_beampanel_image_amplification_request()
    pump, signal = request.launch_configuration.beams.channels
    beams = replace(
        request.launch_configuration.beams,
        channels=(pump, replace(signal, tilt_y_rad_per_um=0.01)),
    )

    with pytest.raises(ValueError, match="carriers in the x-z plane"):
        replace(
            request,
            launch_configuration=replace(
                request.launch_configuration,
                beams=beams,
            ),
        ).validate()


def test_beampanel_request_requires_symmetric_x_carriers():
    request = _valid_beampanel_image_amplification_request()
    request.validate()
    pump, signal = request.launch_configuration.beams.channels
    beams = replace(
        request.launch_configuration.beams,
        channels=(pump, replace(signal, tilt_x_rad_per_um=-0.5)),
    )

    with pytest.raises(ValueError, match="symmetric pump/signal x carriers"):
        replace(
            request,
            launch_configuration=replace(
                request.launch_configuration,
                beams=beams,
            ),
        ).validate()


def test_beampanel_request_rejects_pump_screen():
    request = _valid_beampanel_image_amplification_request()
    signal_assignment = request.launch_configuration.channel_elements[0]
    screen = signal_assignment.elements[0]
    launch = replace(
        request.launch_configuration,
        channel_elements=(
            ChannelLaunchElements(0, (screen,)),
            signal_assignment,
        ),
    )

    with pytest.raises(ValueError, match="pump channel must not carry"):
        replace(request, launch_configuration=launch).validate()


def test_beampanel_request_rejects_multiple_signal_screens():
    request = _valid_beampanel_image_amplification_request()
    assignment = request.launch_configuration.channel_elements[0]
    launch = replace(
        request.launch_configuration,
        channel_elements=(
            replace(assignment, elements=assignment.elements * 2),
        ),
    )

    with pytest.raises(ValueError, match="exactly one IntensityRasterScreen"):
        replace(request, launch_configuration=launch).validate()


def test_static_image_amplification_is_allowed_by_capable_slurm_and_forwards_profile(app):
    class ObservedSlurmRunner:
        name = "Slurm"
        supports_parallel_sweeps = False
        registered_operations = (PR_STATIC_OPERATION,)

        def __init__(self):
            self.calls = []

        def run_registered(self, material_id, workflow_id, request, **kwargs):
            self.calls.append((material_id, workflow_id, request, kwargs))
            local_kwargs = dict(kwargs)
            local_kwargs.pop("resource_profile", None)
            return LocalRunner((PR_STATIC_OPERATION,)).run_registered(
                material_id, workflow_id, request, **local_kwargs
            )

    image_request = _valid_beampanel_image_amplification_request()
    base = PRStaticRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    request = image_amplification_experiment_request(
        image_request,
        base_workflow_id=PR_STATIC_WORKFLOW,
        base_request=base,
    )
    runner = ObservedSlurmRunner()
    window = PRMainWindow(slurm_runner=runner)
    window.runner = runner
    window.runner_label.setText("Runner: Slurm")
    window.remote_execution_controls.validate_backend = lambda _backend: None
    window.remote_execution_controls.runner_kwargs = lambda: {
        "resource_profile": "gpu-test"
    }
    window.build_request = lambda: request
    starts = []
    window._start_background = lambda supplied, **kwargs: starts.append(
        (supplied, kwargs)
    )

    window.run_clicked()

    assert starts and starts[0][0] is request
    result = window._run_registered(request)
    assert result.result.analysis_status == "completed"
    assert len(runner.calls) == 1
    assert runner.calls[0][0:2] == ("pr", PR_STATIC_WORKFLOW)
    assert runner.calls[0][3]["resource_profile"] == "gpu-test"
    window.close()


def test_image_amplification_rejects_unregistered_slurm_base_operation(app):
    class TransverseOnlySlurmRunner:
        name = "Slurm"
        supports_parallel_sweeps = False
        registered_operations = (PR_TRANSVERSE_STATIC_OPERATION,)

    image_request = _valid_beampanel_image_amplification_request()
    base = PRStaticRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    request = image_amplification_experiment_request(
        image_request,
        base_workflow_id=PR_STATIC_WORKFLOW,
        base_request=base,
    )
    runner = TransverseOnlySlurmRunner()
    window = PRMainWindow(slurm_runner=runner)
    window.runner = runner
    window.remote_execution_controls.validate_backend = lambda _backend: None
    window.build_request = lambda: request

    window.run_clicked()

    assert window.status_label.text() == "Invalid request"
    assert "base operation 'pr_static'" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()


def test_gui_slurm_eligibility_uses_registered_ordinary_operation(app):
    class RegisteredSlurmRunner:
        name = "Slurm"
        supports_parallel_sweeps = False
        registered_operations = (
            PR_TIMEDEPENDENT_OPERATION,
            PR_STATIC_OPERATION,
            PR_TRANSVERSE_STATIC_OPERATION,
        )

    image_request = _valid_beampanel_image_amplification_request()
    static_request = PRStaticRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    runner = RegisteredSlurmRunner()
    window = PRMainWindow(slurm_runner=runner)
    window.runner = runner
    window.remote_execution_controls.validate_backend = lambda _backend: None
    starts = []
    window._start_background = lambda supplied, **kwargs: starts.append(
        (supplied, kwargs)
    )
    window.build_request = lambda: static_request

    window.run_clicked()

    assert starts and starts[0][0] is static_request
    starts.clear()
    td_request = PRRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        solver=PRSolverOptions(Nt=0),
        backend=image_request.backend,
    )
    window.build_request = lambda: td_request

    window.run_clicked()

    assert starts and starts[0][0] is td_request
    assert window._slurm_supports_workflow(PR_TIMEDEPENDENT_WORKFLOW)
    assert not window._slurm_supports_workflow(
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    )
    starts.clear()
    transverse_request = PRTransverseRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    window.build_request = lambda: transverse_request

    window.run_clicked()

    assert not starts
    assert window.status_label.text() == "Invalid request"
    assert "pr_transverse_timedependent" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()


def test_beampanel_stage_a_full_result_equivalence():
    source = PRImageSource.from_array(np.eye(8))
    legacy = replace(
        _composite(source, pump_power_mW=3.0, signal_power_mW=0.5),
        solver=PRSolverOptions(Nt=1, dt_normalized=0.01),
    )
    old_prepared, _old_transmission, _old_grating = (
        prepare_image_amplification_workflow_request(legacy)
    )
    signal = old_prepared.beams.channels[1]
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            center_x_um=signal.x0_um,
            center_y_um=signal.y0_um,
            width_um=legacy.launch.image_physical_size_um,
            height_um=legacy.launch.image_physical_size_um,
            boundary_policy="reject",
        ),
    )
    rewired = PRBeamPanelImageAmplificationRunRequest(
        grid=legacy.grid,
        material=legacy.material,
        solver=legacy.solver,
        backend=legacy.backend,
        launch_configuration=LaunchConfiguration(
            old_prepared.beams,
            (ChannelLaunchElements(1, (screen,)),),
        ),
        pump_channel_index=0,
        signal_channel_index=1,
    )

    old_result = run_image_amplification_request(legacy)
    new_result = run_image_amplification_request(rewired)
    for name in (
        "image_transmission",
        "signal_carrier_mask",
        "input_signal_field",
        "output_signal_field",
        "backpropagated_signal_field",
        "zero_response_backpropagated_signal_field",
    ):
        assert np.array_equal(getattr(new_result, name), getattr(old_result, name))
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
        assert np.array_equal(
            getattr(new_result.run_result, name),
            getattr(old_result.run_result, name),
        )
    for name in (
        "incident_channel_powers_mW",
        "post_element_channel_powers_mW",
        "incident_total_power_mW",
        "post_element_total_power_mW",
        "signal_throughput_fraction",
        "measured_absolute_signal_gain",
        "analytic_absolute_signal_gain",
        "analytic_gamma_p_L",
        "image_intensity_correlation",
        "zero_response_image_intensity_correlation",
        "normalized_image_rmse",
        "normalized_power_relative_drift",
    ):
        assert getattr(new_result, name) == getattr(old_result, name)
    assert new_result.run_result.status == old_result.run_result.status
    assert (
        new_result.run_result.completed_steps
        == old_result.run_result.completed_steps
    )
    assert new_result.run_result.diagnostics == old_result.run_result.diagnostics


def test_image_experiment_capability_table_covers_current_pr_algorithms():
    capabilities = {
        capability.workflow_id: capability
        for capability in image_amplification_base_capabilities()
    }
    assert set(capabilities) == {
        "pr_timedependent",
        "pr_static",
        "pr_transverse_static",
        "pr_transverse_timedependent",
    }
    assert capabilities["pr_timedependent"].validation_status == (
        "compatible_and_validated"
    )
    assert capabilities["pr_transverse_static"].validation_status == (
        "compatible_and_validated"
    )
    assert capabilities["pr_static"].validation_status == (
        "compatible_validation_pending"
    )
    assert capabilities["pr_transverse_timedependent"].validation_status == (
        "compatible_and_validated"
    )
    assert capabilities["pr_transverse_timedependent"].launch_adapter == (
        "declarative_elements"
    )


def test_gui_image_mode_reuses_workflow_selector_with_capability_status(app):
    window = PRMainWindow()
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(
            PR_IMAGE_AMPLIFICATION_INPUT_MODE
        )
    )
    selector = window.evolution_panel.workflow
    expected = {
        PR_TIMEDEPENDENT_WORKFLOW: ("Reduced TD", "Validated", True),
        PR_STATIC_WORKFLOW: ("Static", "Experimental", True),
        PR_TRANSVERSE_STATIC_WORKFLOW: (
            "Transverse Static", "Validated", True
        ),
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: (
            "Transverse TD", "Validated", True
        ),
    }
    assert window.evolution_panel._form.labelForField(selector).text() == (
        "Algorithm"
    )
    for workflow_id, (name, status, enabled) in expected.items():
        index = selector.findData(workflow_id)
        assert index >= 0
        assert name in selector.itemText(index)
        assert status in selector.itemText(index)
        assert selector.model().item(index).isEnabled() is enabled
    selector.setCurrentIndex(selector.findData(PR_STATIC_WORKFLOW))
    assert "Experimental" in window.evolution_panel.algorithm_status.text()
    window.close()


def test_gui_image_algorithm_switch_preserves_experiment_inputs(app):
    window = _configured_multi_algorithm_image_window(app)
    material = window.material_panel.material()
    launch = window.beam_panel.launch_configuration()
    pump_index = window.input_panel.pump_channel.currentData()
    signal_index = window.input_panel.signal_channel.currentData()

    for workflow_id in (
        PR_TIMEDEPENDENT_WORKFLOW,
        PR_STATIC_WORKFLOW,
        PR_TRANSVERSE_STATIC_WORKFLOW,
    ):
        window.evolution_panel.set_workflow_id(workflow_id)
        request = window.build_request()
        assert request.base_workflow_id == workflow_id
        assert request.material == material
        assert request.launch_configuration == launch
        assert request.pump_channel_index == pump_index
        assert request.signal_channel_index == signal_index
    window.close()


@pytest.mark.parametrize(
    ("workflow_id", "base_type"),
    (
        (PR_TIMEDEPENDENT_WORKFLOW, PRRunRequest),
        (PR_STATIC_WORKFLOW, PRStaticRunRequest),
        (PR_TRANSVERSE_STATIC_WORKFLOW, PRTransverseStaticRunRequest),
    ),
)
def test_gui_image_algorithm_selection_builds_canonical_base_request(
    app,
    workflow_id,
    base_type,
):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(workflow_id)
    request = window.build_request()
    prepared, transmission, _grating = prepare_image_amplification_base_request(
        request
    )

    assert isinstance(request, PRImageAmplificationExperimentRequest)
    assert request.base_workflow_id == workflow_id
    assert isinstance(request.base_request, base_type)
    assert isinstance(prepared, base_type)
    assert prepared.initial_A is None
    assert prepared.beams == request.launch_configuration.beams
    assert prepared.launch_elements == (
        request.launch_configuration.channel_elements
    )
    assert transmission.shape == (24, 24)
    window.close()


def test_transverse_static_image_experiment_is_functional_and_coherent(app):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
    td_request = window.build_request()
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    static_request = window.build_request()
    window.close()

    prepared_td, transmission_td, grating_td = (
        prepare_image_amplification_base_request(td_request)
    )
    prepared_static, transmission_static, grating_static = (
        prepare_image_amplification_base_request(static_request)
    )
    assert prepared_td.beams == prepared_static.beams
    assert prepared_td.launch_elements == prepared_static.launch_elements
    assert np.array_equal(transmission_td, transmission_static)
    assert grating_td == grating_static

    runner = LocalRunner(
        operations=(
            PR_TIMEDEPENDENT_OPERATION,
            PR_TRANSVERSE_STATIC_OPERATION,
        )
    )
    td = run_image_amplification_experiment(runner, td_request)
    progress = []
    static = run_image_amplification_experiment(
        runner,
        static_request,
        progress_callback=progress.append,
    )
    td_analysis = td.result.analysis_result
    static_analysis = static.result.analysis_result
    assert td_analysis is not None
    assert static_analysis is not None
    assert static.result.base_status == "converged"
    assert static.result.status == "converged"
    assert static.result.analysis_status == "completed"
    assert static.result.run_result.converged
    assert static.result.run_result.replay_diagnostics["field_match"] is True
    for value in (
        static_analysis.measured_absolute_signal_gain,
        static_analysis.analytic_absolute_signal_gain,
        static_analysis.image_intensity_correlation,
        static_analysis.normalized_image_rmse,
        static_analysis.normalized_power_relative_drift,
    ):
        assert np.isfinite(value)
    assert -1.0 <= static_analysis.image_intensity_correlation <= 1.0
    assert static_analysis.normalized_image_rmse >= 0.0
    assert abs(static_analysis.normalized_power_relative_drift) < 1.0e-10
    assert np.array_equal(
        td_analysis.signal_carrier_mask,
        static_analysis.signal_carrier_mask,
    )
    assert static_analysis.backpropagated_signal_field.shape == (24, 24)
    assert "image_amplification" in static.run_data.diagnostics
    assert "image_amplification_metrics" in static.run_data.diagnostics
    assert "transverse_pr" in static.run_data.diagnostics
    assert static.run_data.diagnostics["image_amplification_metrics"].values[
        "measured_signal_gain"
    ] == static_analysis.measured_absolute_signal_gain
    assert tuple(static.run_data.curves) == tuple(
        static.result.base_runner_result.run_data.curves
    )
    assert any(
        item.workflow == PR_TRANSVERSE_STATIC_WORKFLOW for item in progress
    )
    assert [
        item.diagnostics["stage"]
        for item in progress
        if item.workflow == PR_IMAGE_AMPLIFICATION_WORKFLOW
    ] == list(image_amplification_module.PR_IMAGE_ANALYSIS_STAGES)

    gui = PRMainWindow()
    gui._thread_done = True
    gui._on_finished(static)
    assert gui.run_status == "completed"
    assert gui.status_label.text() == "Converged"
    assert "image_amplification" in gui.last_runner_result.run_data.diagnostics
    assert "transverse_pr" in gui.last_runner_result.run_data.diagnostics
    assert "Authoritative zero-flux residual" in (
        gui.results_panel.workspace.console.toPlainText()
    )
    gui.close()


def test_gui_runs_transverse_static_image_progress_and_completion(app):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    calls = []
    original_run = window.local_runner.run_registered

    def observed(material_id, workflow_id, request, **kwargs):
        calls.append((material_id, workflow_id, request))
        return original_run(material_id, workflow_id, request, **kwargs)

    window.local_runner.run_registered = observed
    window.run_button.click()
    deadline = time.monotonic() + 10.0
    while window._background_running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    app.processEvents()

    console = window.results_panel.workspace.console.toPlainText()
    assert not window._background_running
    assert calls and calls[0][:2] == ("pr", PR_TRANSVERSE_STATIC_WORKFLOW)
    assert isinstance(calls[0][2], PRTransverseStaticRunRequest)
    assert window.run_status == "completed"
    assert window.status_label.text() == "Converged"
    assert "PR transverse static progress" in console
    assert "Post-processing image amplification" in console
    assert "Image-amplification run complete" in console
    assert "image_amplification" in window.last_runner_result.run_data.diagnostics
    window.close()


def test_gui_runs_transverse_td_image_progress_and_completion(app):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    )
    window.evolution_panel.Nt.setValue(1)
    window.evolution_panel.dt_normalized.setValue(1.0e-4)
    calls = []
    original_run = window.local_runner.run_registered

    def observed(material_id, workflow_id, request, **kwargs):
        calls.append((material_id, workflow_id, request))
        return original_run(material_id, workflow_id, request, **kwargs)

    window.local_runner.run_registered = observed
    window.run_button.click()
    deadline = time.monotonic() + 10.0
    while window._background_running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    app.processEvents()

    console = window.results_panel.workspace.console.toPlainText()
    assert not window._background_running
    assert calls and calls[0][:2] == (
        "pr",
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    )
    assert isinstance(calls[0][2], PRTransverseRunRequest)
    assert calls[0][2].initial_A is None
    assert calls[0][2].launch_elements
    assert window.run_status == "completed"
    assert window.status_label.text() == "Completed"
    assert "PR progress" in console
    assert "Post-processing image amplification" in console
    assert "Image-amplification run complete" in console
    assert "image_amplification" in window.last_runner_result.run_data.diagnostics
    assert "transverse_pr" in window.last_runner_result.run_data.diagnostics
    window.close()


def test_transverse_static_image_experiment_forwards_base_cancellation(app):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    request = window.build_request()
    window.close()
    token = CancellationToken()
    progress = []

    def cancel_after_first_accepted_iteration(item):
        progress.append(item)
        if item.workflow == PR_TRANSVERSE_STATIC_WORKFLOW:
            token.cancel()

    result = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TRANSVERSE_STATIC_OPERATION,)),
        request,
        cancellation_token=token,
        progress_callback=cancel_after_first_accepted_iteration,
    ).result

    assert any(item.workflow == PR_TRANSVERSE_STATIC_WORKFLOW for item in progress)
    assert result.base_status == "cancelled"
    assert result.status == "cancelled"
    assert result.analysis_status == "not_run"
    assert result.analysis_result is None
    assert all(
        item.workflow != PR_IMAGE_AMPLIFICATION_WORKFLOW for item in progress
    )


def test_legacy_static_image_experiment_dispatches_registered_operation(app):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    request = window.build_request()
    window.close()

    result = run_image_amplification_experiment(
        LocalRunner(operations=(PR_STATIC_OPERATION,)),
        request,
    )

    assert result.result.base_runner_result.kind == PR_STATIC_WORKFLOW
    assert isinstance(result.result.run_result, PRStaticRunResult)
    assert result.result.analysis_result is not None
    assert result.result.analysis_status == "completed"


@pytest.mark.parametrize(
    ("workflow_id", "operation"),
    (
        (PR_TIMEDEPENDENT_WORKFLOW, PR_TIMEDEPENDENT_OPERATION),
        (PR_STATIC_WORKFLOW, PR_STATIC_OPERATION),
        (PR_TRANSVERSE_STATIC_WORKFLOW, PR_TRANSVERSE_STATIC_OPERATION),
    ),
)
def test_image_experiment_matches_in_process_transport_roundtrip(
    app, workflow_id, operation
):
    window = _configured_multi_algorithm_image_window(app)
    window.evolution_panel.set_workflow_id(workflow_id)
    request = window.build_request()
    window.close()
    registry = default_transport_registry()

    class InProcessTransportRunner:
        registered_operations = (operation,)

        def run_registered(
            self, material_id, selected_workflow_id, supplied, **kwargs
        ):
            codec = registry.codec(material_id, selected_workflow_id)
            encoded_request = codec.encode_request(supplied)
            decoded_request = codec.decode_request(
                encoded_request.payload.metadata,
                encoded_request.payload.arrays,
            )
            completed = operation.run(decoded_request, **kwargs)
            encoded_result = codec.encode_result(completed)
            decoded_result = codec.decode_result(
                encoded_result.payload.metadata,
                encoded_result.payload.arrays,
            )
            return RunnerResult(
                kind=selected_workflow_id,
                result=decoded_result,
                run_data=operation.to_run_data(decoded_result),
                material_id=material_id,
            )

    local = run_image_amplification_experiment(
        LocalRunner((operation,)), request
    ).result
    transported = run_image_amplification_experiment(
        InProcessTransportRunner(), request
    ).result

    assert transported.status == local.status
    assert transported.analysis_status == local.analysis_status
    for name in ("A_initial", "A_final"):
        np.testing.assert_array_equal(
            getattr(transported.run_result, name),
            getattr(local.run_result, name),
        )
    assert transported.analysis_result is not None
    assert local.analysis_result is not None
    for name in (
        "image_transmission", "signal_carrier_mask", "input_signal_field",
        "output_signal_field", "backpropagated_signal_field",
        "zero_response_backpropagated_signal_field",
    ):
        np.testing.assert_array_equal(
            getattr(transported.analysis_result, name),
            getattr(local.analysis_result, name),
        )
    for name in (
        "measured_absolute_signal_gain", "analytic_absolute_signal_gain",
        "image_intensity_correlation", "normalized_image_rmse",
        "normalized_power_relative_drift",
    ):
        assert getattr(transported.analysis_result, name) == pytest.approx(
            getattr(local.analysis_result, name), abs=0.0, rel=0.0
        )


def test_image_experiment_reduced_td_transformation_is_declarative():
    image_request = _valid_beampanel_image_amplification_request()
    experiment = image_amplification_experiment_request(image_request)
    prepared, transmission, _grating = prepare_image_amplification_base_request(
        experiment
    )

    assert experiment.base_workflow_id == PR_TIMEDEPENDENT_WORKFLOW
    assert prepared.initial_A is None
    assert (
        prepared.launch_elements
        == image_request.launch_configuration.channel_elements
    )
    assert prepared.beams == image_request.launch_configuration.beams
    assert transmission.shape == (image_request.grid.Nx, image_request.grid.Ny)


@pytest.mark.parametrize(
    ("workflow_id", "base_request_type", "expects_declarative"),
    (
        (PR_STATIC_WORKFLOW, PRStaticRunRequest, True),
        (PR_TRANSVERSE_STATIC_WORKFLOW, PRTransverseStaticRunRequest, True),
        (PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW, PRTransverseRunRequest, True),
    ),
)
def test_image_experiment_current_algorithm_launch_adapters(
    workflow_id,
    base_request_type,
    expects_declarative,
):
    image_request = _valid_beampanel_image_amplification_request()
    base = base_request_type(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    experiment = image_amplification_experiment_request(
        image_request,
        base_workflow_id=workflow_id,
        base_request=base,
    )
    prepared, _transmission, _grating = prepare_image_amplification_base_request(
        experiment
    )

    if expects_declarative:
        assert prepared.initial_A is None
        assert (
            prepared.launch_elements
            == image_request.launch_configuration.channel_elements
        )
    else:
        assert prepared.initial_A is not None
        assert not hasattr(prepared, "launch_elements")


def test_transverse_td_image_amplification_local_scientific_validation():
    image_request = _valid_beampanel_image_amplification_request()
    base = PRTransverseRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=replace(image_request.material, applied_field=0.0),
        solver=PRTransverseSolverOptions(
            Nt=2,
            dt_normalized=1.0e-4,
            optical_substeps=1,
        ),
        backend=BackendSpec("numpy", "float64", False),
    )
    experiment = image_amplification_experiment_request(
        image_request,
        base_workflow_id=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        base_request=base,
    )
    prepared, transmission, _grating = prepare_image_amplification_base_request(
        experiment
    )
    runner = LocalRunner(operations=(PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,))
    composite = run_image_amplification_experiment(runner, experiment).result
    reduced = image_amplification_experiment_request(
        image_request,
        base_workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
        base_request=PRRunRequest(
            grid=image_request.grid,
            beams=image_request.launch_configuration.beams,
            material=image_request.material,
            solver=PRSolverOptions(Nt=2, dt_normalized=1.0e-4),
            backend=BackendSpec("numpy", "float64", False),
        ),
    )
    reduced_result = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        reduced,
    ).result

    assert prepared.initial_A is None
    assert prepared.launch_elements == (
        image_request.launch_configuration.channel_elements
    )
    assert composite.status == "completed"
    assert composite.analysis_status == "completed"
    assert composite.run_result.status == "completed"
    assert composite.analysis_result is not None
    analysis = composite.analysis_result
    assert np.array_equal(analysis.image_transmission, transmission)
    np.testing.assert_array_equal(
        composite.run_result.A_initial,
        reduced_result.run_result.A_initial,
    )
    assert np.all(np.isfinite(composite.run_result.A_final))
    assert np.all(np.isfinite(analysis.backpropagated_signal_field))
    assert np.isfinite(analysis.measured_absolute_signal_gain)
    assert np.isfinite(analysis.image_intensity_correlation)
    assert np.isfinite(analysis.normalized_image_rmse)
    assert np.isfinite(
        reduced_result.analysis_result.image_intensity_correlation
    )
    assert analysis.image_intensity_correlation > 0.0
    assert abs(analysis.normalized_power_relative_drift) < 2.0e-13
    assert composite.run_result.diagnostics["complete_final_optical_replay"]


def test_image_experiment_uses_one_base_runner_call_and_forwards_controls():
    image_request = _valid_beampanel_image_amplification_request()
    experiment = image_amplification_experiment_request(image_request)
    runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
    calls = []
    base_progress = RunProgress(
        workflow=PR_TIMEDEPENDENT_WORKFLOW,
        status="running",
        completed_units=0,
        total_units=0,
        current_coordinate=0.0,
        coordinate_name="time",
        coordinate_unit="1",
        elapsed_wall_time=0.0,
    )
    observed_progress = []
    token = CancellationToken()
    ordinary_run = runner.run_registered

    def recorded(material_id, workflow_id, request, **kwargs):
        calls.append((material_id, workflow_id, request, kwargs))
        kwargs["progress_callback"](base_progress)
        return ordinary_run(material_id, workflow_id, request, **kwargs)

    runner.run_registered = recorded
    result = run_image_amplification_experiment(
        runner,
        experiment,
        cancellation_token=token,
        progress_callback=observed_progress.append,
    )

    assert len(calls) == 1
    assert calls[0][0:2] == ("pr", PR_TIMEDEPENDENT_WORKFLOW)
    assert calls[0][3]["cancellation_token"] is token
    assert observed_progress[0] is base_progress
    assert [
        progress.diagnostics["stage"]
        for progress in observed_progress[1:]
    ] == list(image_amplification_module.PR_IMAGE_ANALYSIS_STAGES)
    assert isinstance(result.result, PRImageAmplificationCompositeResult)
    assert result.result.analysis_status == "completed"
    assert result.result.base_runner_result.kind == PR_TIMEDEPENDENT_WORKFLOW
    assert result.kind == PR_IMAGE_AMPLIFICATION_WORKFLOW
    assert "image_amplification" in result.run_data.diagnostics


def test_gui_completion_accepts_d1_composite_and_installs_all_products(app):
    image_request = replace(
        _valid_beampanel_image_amplification_request(),
        solver=PRSolverOptions(Nt=0, dt_normalized=0.01),
    )
    runner_result = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        image_amplification_experiment_request(image_request),
    )
    composite = runner_result.result
    assert isinstance(composite, PRImageAmplificationCompositeResult)
    assert composite.status == "completed"

    window = PRMainWindow()
    window._thread_done = True
    window._on_finished(runner_result)

    selector = window.results_panel.workspace.image_pane.field_selector
    console = window.results_panel.workspace.console.toPlainText()
    diagnostics = window.results_panel.workspace.diagnostics_view.toPlainText()
    assert window.run_status == "completed"
    assert window.status_label.text() == "Completed"
    assert window.last_result is composite
    assert selector.findData("output_intensity") >= 0
    assert selector.findData("amplified_image") >= 0
    assert selector.findData("zero_response_image") >= 0
    assert "[image_amplification] Image Amplification" in diagnostics
    assert (
        "[image_amplification_metrics] Image Amplification Metrics"
        in diagnostics
    )
    assert "measured_signal_gain" in diagnostics
    assert "output_isolated_signal_power_mW" in diagnostics
    assert "Image-amplification run complete" in console
    assert (
        "Normalized optical power: "
        f"{composite.run_result.power_initial:.8g} -> "
        f"{composite.run_result.power_final:.8g}"
    ) in console
    assert "ERROR" not in console
    assert composite.analysis_result is not None
    assert np.array_equal(
        runner_result.run_data.fields["amplified_image"].data,
        np.abs(composite.analysis_result.backpropagated_signal_field) ** 2,
    )
    window.close()


def test_gui_completion_preserves_nonconverged_composite_base_status(app):
    image_request = replace(
        _valid_beampanel_image_amplification_request(),
        solver=PRSolverOptions(Nt=0, dt_normalized=0.01),
    )
    runner_result = run_image_amplification_experiment(
        LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,)),
        image_amplification_experiment_request(image_request),
    )
    base = runner_result.result.base_runner_result
    nonconverged_base = replace(
        base,
        result=replace(base.result, status="not_converged"),
    )
    composite = replace(
        runner_result.result,
        base_runner_result=nonconverged_base,
    )

    window = PRMainWindow()
    window._thread_done = True
    window._on_finished(replace(runner_result, result=composite))

    assert composite.analysis_status == "completed"
    assert composite.status == "not_converged"
    assert window.run_status == "not_converged"
    assert window.status_label.text() == "Not converged"
    assert "Base PR solve did not converge" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()


def test_image_experiment_honors_selected_registered_operation_name():
    image_request = _valid_beampanel_image_amplification_request()
    base = PRStaticRunRequest(
        grid=image_request.grid,
        beams=image_request.launch_configuration.beams,
        material=image_request.material,
        backend=image_request.backend,
    )
    experiment = image_amplification_experiment_request(
        image_request,
        base_workflow_id=PR_STATIC_WORKFLOW,
        base_request=base,
    )
    calls = []

    class RecordingRunner:
        def run_registered(self, material_id, workflow_id, request, **kwargs):
            calls.append((material_id, workflow_id, request, kwargs))
            raise RuntimeError("stop after dispatch evidence")

    with pytest.raises(RuntimeError, match="dispatch evidence"):
        run_image_amplification_experiment(RecordingRunner(), experiment)
    assert len(calls) == 1
    assert calls[0][0:2] == ("pr", PR_STATIC_WORKFLOW)
    assert isinstance(calls[0][2], PRStaticRunRequest)


def test_image_experiment_reduced_td_matches_direct_completed_result_bit_for_bit():
    image_request = replace(
        _valid_beampanel_image_amplification_request(),
        solver=PRSolverOptions(Nt=1, dt_normalized=0.01),
    )
    direct = run_image_amplification_request(image_request)
    runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
    composite = run_image_amplification_experiment(
        runner,
        image_amplification_experiment_request(image_request),
    )
    analyzed = composite.result.analysis_result
    assert analyzed is not None
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
        assert np.array_equal(
            getattr(analyzed.run_result, name),
            getattr(direct.run_result, name),
        )
    for name in (
        "image_transmission",
        "signal_carrier_mask",
        "input_signal_field",
        "output_signal_field",
        "backpropagated_signal_field",
        "zero_response_backpropagated_signal_field",
    ):
        assert np.array_equal(getattr(analyzed, name), getattr(direct, name))
    for name in (
        "measured_absolute_signal_gain",
        "analytic_absolute_signal_gain",
        "measured_gain_reference_signal_power_normalized",
        "output_isolated_signal_power_normalized",
        "measured_gain_reference_signal_power_mW",
        "output_isolated_signal_power_mW",
        "image_intensity_correlation",
        "zero_response_image_intensity_correlation",
        "normalized_image_rmse",
        "normalized_power_relative_drift",
    ):
        assert getattr(analyzed, name) == getattr(direct, name)
    direct_run_data = pr_image_amplification_result_to_run_data(direct)
    assert tuple(composite.run_data.fields) == tuple(direct_run_data.fields)
    assert tuple(composite.run_data.diagnostics) == tuple(direct_run_data.diagnostics)
    for key in direct_run_data.fields:
        assert np.array_equal(
            composite.run_data.fields[key].data,
            direct_run_data.fields[key].data,
        )
    for key in direct_run_data.diagnostics:
        composite_values = composite.run_data.diagnostics[key].values
        direct_values = direct_run_data.diagnostics[key].values
        if key != "summary":
            assert composite_values == direct_values
            continue
        assert {
            name: value
            for name, value in composite_values.items()
            if name != "launch"
        } == {
            name: value
            for name, value in direct_values.items()
            if name != "launch"
        }
        composite_launch = composite_values["launch"]
        direct_launch = direct_values["launch"]
        for name in (
            "Nch",
            "coherence",
            "coherence_groups",
            "physical_channel_powers_mW",
            "physical_total_power_mW",
            "power_fractions",
            "field_normalization",
            "wavelengths_um",
        ):
            assert composite_launch[name] == direct_launch[name]
        assert composite_launch["post_element_channel_powers_mW"] == list(
            analyzed.post_element_channel_powers_mW
        )
        assert direct_launch["channel_throughput_fractions"][1] == 1.0
        assert composite_launch["channel_throughput_fractions"][1] < 1.0


def test_image_experiment_cancellation_between_analysis_stages_preserves_base():
    image_request = _valid_beampanel_image_amplification_request()
    runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
    token = CancellationToken()

    def observe(progress):
        if (
            progress.workflow == PR_IMAGE_AMPLIFICATION_WORKFLOW
            and progress.diagnostics["stage"] == "carrier_isolation"
        ):
            token.cancel()

    composite = run_image_amplification_experiment(
        runner,
        image_amplification_experiment_request(image_request),
        cancellation_token=token,
        progress_callback=observe,
    ).result

    assert composite.base_status == "completed"
    assert composite.analysis_status == "cancelled"
    assert composite.status == "cancelled"
    assert composite.analysis_result is None
    assert composite.base_runner_result.run_data is not None


def test_image_experiment_preserves_nonconverged_base_status():
    image_request = _valid_beampanel_image_amplification_request()
    ordinary = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))

    class NonconvergedRunner:
        def run_registered(self, *args, **kwargs):
            result = ordinary.run_registered(*args, **kwargs)
            return replace(
                result,
                result=replace(result.result, status="not_converged"),
            )

    composite = run_image_amplification_experiment(
        NonconvergedRunner(),
        image_amplification_experiment_request(image_request),
    ).result

    assert composite.base_status == "not_converged"
    assert composite.analysis_status == "completed"
    assert composite.status == "not_converged"
    assert composite.analysis_result is not None


def test_image_experiment_analysis_failure_retains_completed_base(monkeypatch):
    image_request = _valid_beampanel_image_amplification_request()
    runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))

    def fail_metrics(*_args, **_kwargs):
        raise ValueError("diagnostic failure")

    monkeypatch.setattr(
        image_amplification_module,
        "_intensity_metrics",
        fail_metrics,
    )
    result = run_image_amplification_experiment(
        runner,
        image_amplification_experiment_request(image_request),
    )
    composite = result.result

    assert composite.base_status == "completed"
    assert composite.analysis_status == "failed"
    assert composite.status == "failed"
    assert composite.analysis_result is None
    assert "diagnostic failure" in composite.analysis_message
    assert result.run_data is composite.base_runner_result.run_data


def test_mode_switch_preserves_shared_beams_screens_and_role_names(app):
    window = PRMainWindow()
    kx = 0.1
    stack = BeamStackDefinition(
        beams=(
            BeamDefinition(
                name="pump",
                tilt_x_rad_per_um=kx,
                coherence_group="shared",
            ),
            BeamDefinition(
                name="signal",
                tilt_x_rad_per_um=-kx,
                coherence_group="shared",
            ),
        )
    )
    window.beam_panel.set_beam_stack_definition(stack)
    editor = window.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.set_source(PRImageSource.from_array(np.eye(4)))
    original_elements = window.beam_panel.launch_elements()

    image_index = window.input_panel.input_mode.findData(
        PR_IMAGE_AMPLIFICATION_INPUT_MODE
    )
    gaussian_index = window.input_panel.input_mode.findData(
        "gaussian_beams"
    )
    window.input_panel.input_mode.setCurrentIndex(image_index)
    window.input_panel.sync_channels()
    assert window.input_panel.pump_channel.currentText() == "0 — pump"
    assert window.input_panel.signal_channel.currentText() == "1 — signal"
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.beam_panel))
    window.input_panel.input_mode.setCurrentIndex(gaussian_index)

    assert window.beam_panel.beam_stack_definition == stack
    assert window.beam_panel.launch_elements() == original_elements
    window.close()


def test_signal_role_may_be_first_canonical_channel():
    source = PRImageSource.from_array(np.eye(4))
    legacy = _composite(source)
    prepared, _transmission, expected_grating = (
        prepare_image_amplification_workflow_request(legacy)
    )
    reversed_beams = replace(
        prepared.beams,
        channels=tuple(reversed(prepared.beams.channels)),
    )
    signal = reversed_beams.channels[0]
    screen = IntensityRasterScreen(
        source,
        ScreenPlacement(
            center_x_um=signal.x0_um,
            center_y_um=signal.y0_um,
            width_um=legacy.launch.image_physical_size_um,
            height_um=legacy.launch.image_physical_size_um,
            boundary_policy="reject",
        ),
    )
    request = PRBeamPanelImageAmplificationRunRequest(
        grid=legacy.grid,
        material=legacy.material,
        solver=legacy.solver,
        backend=legacy.backend,
        launch_configuration=LaunchConfiguration(
            reversed_beams,
            (ChannelLaunchElements(0, (screen,)),),
        ),
        pump_channel_index=1,
        signal_channel_index=0,
    )

    workflow, _transmission, grating = prepare_image_amplification_workflow_request(
        request
    )
    result = run_image_amplification_request(request)
    run_data = PR_IMAGE_AMPLIFICATION_OPERATION.to_run_data(result)

    assert grating == expected_grating
    assert np.any(np.asarray(workflow.initial_A[0]) != 0.0)
    assert run_data.diagnostics["image_amplification"].values[
        "incident_signal_power_mW"
    ] == pytest.approx(signal.power_mW)


def test_role_selectors_follow_unique_names_on_reorder_and_clear_disabled(app):
    window = PRMainWindow()
    pump = BeamDefinition(name="pump", coherence_group="shared")
    signal = BeamDefinition(name="signal", coherence_group="shared")
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(pump, signal))
    )
    assert window.input_panel.pump_channel.currentData() == 0
    assert window.input_panel.signal_channel.currentData() == 1

    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(signal, pump))
    )
    assert window.input_panel.pump_channel.currentData() == 1
    assert window.input_panel.signal_channel.currentData() == 0

    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(replace(signal, enabled=False), pump))
    )
    assert window.input_panel.signal_channel.currentData() is None
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(PR_IMAGE_AMPLIFICATION_INPUT_MODE)
    )
    with pytest.raises(ValueError, match="valid enabled signal channel"):
        window.build_request()
    window.close()


def test_role_selectors_follow_live_add_duplicate_rename_disable_and_delete(app):
    window = PRMainWindow()
    launch = window.beam_panel.launch_plane_widget
    assert window.input_panel.pump_channel.count() == 1
    assert window.input_panel.signal_channel.count() == 1

    launch._place_beam(4.0, -3.0)
    assert window.input_panel.pump_channel.count() == 2
    assert window.input_panel.signal_channel.count() == 2
    launch.delete_selected()
    assert window.input_panel.pump_channel.count() == 1
    assert window.input_panel.signal_channel.count() == 1

    launch.object_list.setCurrentRow(0)
    launch.duplicate_selected()
    assert window.input_panel.pump_channel.count() == 2
    assert window.input_panel.signal_channel.count() == 2
    assert window.input_panel.pump_channel.currentData() == 0
    assert window.input_panel.signal_channel.currentData() == 1

    launch.name_edit.setText("signal")
    launch.name_edit.editingFinished.emit()
    assert window.input_panel.signal_channel.currentData() == 1
    assert window.input_panel.signal_channel.currentText() == "1 — signal"

    editor = window.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.set_source(PRImageSource.from_array(np.eye(4)))
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(PR_IMAGE_AMPLIFICATION_INPUT_MODE)
    )
    request = window.build_request()
    assert request.pump_channel_index == 0
    assert request.signal_channel_index == 1

    launch.enabled_checkbox.setChecked(False)
    assert window.input_panel.pump_channel.count() == 1
    assert window.input_panel.signal_channel.count() == 1
    assert window.input_panel.signal_channel.currentData() is None
    launch.enabled_checkbox.setChecked(True)
    assert window.input_panel.pump_channel.count() == 2
    assert window.input_panel.signal_channel.count() == 2
    assert window.input_panel.signal_channel.currentData() == 1

    launch.delete_selected()
    assert window.input_panel.pump_channel.count() == 1
    assert window.input_panel.signal_channel.count() == 1
    window.close()
