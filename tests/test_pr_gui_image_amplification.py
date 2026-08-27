from __future__ import annotations

from dataclasses import replace
import hashlib
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
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
    PRImageAmplificationRunRequest,
    PRImageAmplificationSpec,
    PRImageLaunchSpec,
    apply_passive_field_transmittance,
    image_amplification_run_request,
    intensity_transmission_to_field_transmittance,
    prepare_image_amplification_workflow_request,
    prepare_image_transmission,
    run_image_amplification,
    run_image_amplification_request,
)
from lcprop.pr.image_sources import PRImageSource, standard_image_catalog
from lcprop.pr.operations import PR_IMAGE_AMPLIFICATION_OPERATION
from lcprop.pr.specs import PRMaterialSpec, PRSolverOptions


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
    assert run_data.geometry.x.shape == (32,)
    assert run_data.geometry.y.shape == (8,)


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

    assert isinstance(request, PRBeamPanelImageAmplificationRunRequest)
    assert request.source.grayscale.shape == (4, 8)
    assert request.incident_signal_to_pump_power_ratio == pytest.approx(0.125)
    assert window.grid_panel.grid() == before == request.grid
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.beam_panel))
    assert not window.evolution_panel.workflow.isEnabled()
    assert not hasattr(window.input_panel, "preview")
    assert not window._background_running
    assert calls and calls[0][:2] == ("pr", PR_IMAGE_AMPLIFICATION_WORKFLOW)
    assert calls[0][2] == request
    assert window.last_runner_result.kind == PR_IMAGE_AMPLIFICATION_WORKFLOW
    assert "amplified_image" in window.last_runner_result.run_data.fields
    assert "image_amplification" in window.last_runner_result.run_data.diagnostics
    with pytest.raises(ValueError, match="unsupported in C1"):
        window.save_experiment_to(tmp_path / "image.lcprop.json")
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


def test_beampanel_image_amplification_is_explicitly_rejected_by_slurm(app):
    class ObservedSlurmRunner:
        name = "Slurm"
        supports_parallel_sweeps = False
        registered_operations = ()

        def __init__(self):
            self.calls = []

        def run_registered(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("unsupported request reached Slurm dispatch")

    request = _valid_beampanel_image_amplification_request()
    assert request.launch_configuration.channel_elements
    assert request.pump_channel_index == 0
    assert request.signal_channel_index == 1
    runner = ObservedSlurmRunner()
    window = PRMainWindow(slurm_runner=runner)
    window.runner = runner
    window.runner_label.setText("Runner: Slurm")
    window.remote_execution_controls.validate_backend = lambda _backend: None
    window.build_request = lambda: request

    window.run_clicked()

    assert runner.calls == []
    assert window._active_request is None
    assert not window._background_running
    assert window.status_label.text() == "Invalid request"
    assert "supports only pr_transverse_static" in (
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
    window.input_panel.sync_channels()
    assert window.input_panel.pump_channel.currentData() == 0
    assert window.input_panel.signal_channel.currentData() == 1

    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(signal, pump))
    )
    window.input_panel.sync_channels()
    assert window.input_panel.pump_channel.currentData() == 1
    assert window.input_panel.signal_channel.currentData() == 0

    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(replace(signal, enabled=False), pump))
    )
    window.input_panel.sync_channels()
    assert window.input_panel.signal_channel.currentData() is None
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(PR_IMAGE_AMPLIFICATION_INPUT_MODE)
    )
    with pytest.raises(ValueError, match="valid enabled signal channel"):
        window.build_request()
    window.close()
