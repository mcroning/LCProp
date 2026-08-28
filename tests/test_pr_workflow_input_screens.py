from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
)
from lcprop.persistence import load_experiment, save_experiment
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.request_adapter import apply_pr_request
from lcprop.pr.persistence import save_pr_checkpoint
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_MATERIAL_ID,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
    run_pr_static,
)
from lcprop.pr.static_transport_codec import (
    decode_pr_static_transport_request,
    encode_pr_static_transport_request,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport_codec import (
    decode_pr_transverse_static_transport_request,
    encode_pr_transverse_static_transport_request,
)
from lcprop.pr.workflow import continue_pr_timedependent, run_pr_timedependent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _grid() -> GridSpec:
    return GridSpec(
        Nx=12,
        Ny=8,
        x_aperture_um=24.0,
        y_aperture_um=16.0,
        dz_um=2.0,
        z_length_um=2.0,
    )


def _beams(n_channels: int = 2) -> BeamStack:
    base = (
        BeamChannel(
            name="one",
            wavelength_um=0.633,
            power_mW=3.0,
            waist_x_um=6.0,
            waist_y_um=5.0,
            x0_um=-2.0,
            coherence_group="one",
        ),
        BeamChannel(
            name="two",
            wavelength_um=0.633,
            power_mW=1.0,
            waist_x_um=6.0,
            waist_y_um=5.0,
            x0_um=2.0,
            coherence_group="two",
        ),
        BeamChannel(
            name="three",
            wavelength_um=0.633,
            power_mW=2.0,
            waist_x_um=5.0,
            waist_y_um=4.0,
            coherence_group="three",
        ),
    )
    return BeamStack(channels=base[:n_channels])


def _screen(values, *, center_x_um: float = 0.0) -> IntensityRasterScreen:
    return IntensityRasterScreen(
        source=RasterSource.from_array(np.asarray(values, dtype=float)),
        placement=ScreenPlacement(
            center_x_um=center_x_um,
            width_um=12.0,
            height_um=8.0,
            outside_intensity_transmission=1.0,
        ),
    )


def _assignments(*indices: int) -> tuple[ChannelLaunchElements, ...]:
    patterns = (
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 0.2], [0.4, 1.0]],
        [[0.3, 1.0], [0.8, 0.1]],
    )
    return tuple(
        ChannelLaunchElements(index, (_screen(patterns[position]),))
        for position, index in enumerate(indices)
    )


def _material() -> PRMaterialSpec:
    return PRMaterialSpec(
        dark_intensity=0.5,
        uniform_background_intensity=0.1,
        applied_field=0.0,
        gain_length_product=0.0,
        characteristic_wavenumber_per_um_override=0.1,
    )


def _backend() -> BackendSpec:
    return BackendSpec(backend="numpy", precision="float64", verbose=False)


def _independent(request):
    grid = make_grid(request.grid, real_dtype=np.float64)
    return build_launch(
        request.beams,
        grid,
        complex_dtype=np.complex128,
        launch_elements=request.launch_elements,
    )


def test_launch_configuration_is_immutable_validated_and_rejects_stale_mapping():
    configuration = LaunchConfiguration(_beams(3), _assignments(0, 2))
    beams, elements = configuration
    assert beams is configuration.beams
    assert elements is configuration.channel_elements
    with pytest.raises(ValueError, match="canonical enabled beam channel"):
        LaunchConfiguration(_beams(2), _assignments(2))
    with pytest.raises(Exception):
        configuration.channel_elements = ()


def test_reduced_td_accepts_any_and_multiple_screened_channels_exactly():
    request = PRRunRequest(
        grid=_grid(),
        beams=_beams(3),
        material=_material(),
        solver=PRSolverOptions(Nt=0, dt_normalized=1.0e-3),
        backend=_backend(),
        launch_elements=_assignments(0, 2),
    )
    result = run_pr_timedependent(request)
    independent = _independent(request)
    incident = replace(request, launch_elements=())
    incident_result = run_pr_timedependent(incident)

    assert np.array_equal(result.A_initial, independent.A0)
    assert not np.array_equal(result.A_initial[0], incident_result.A_initial[0])
    assert np.array_equal(result.A_initial[1], incident_result.A_initial[1])
    assert not np.array_equal(result.A_initial[2], incident_result.A_initial[2])
    assert result.launch_summary["physical_channel_powers_mW"] == [3.0, 1.0, 2.0]
    assert result.launch_summary["channel_throughput_fractions"][1] == pytest.approx(1.0)
    assert result.launch_summary["post_element_total_power_mW"] < 6.0


def test_in_memory_td_continuation_preserves_prepared_screened_launch_once():
    request = PRRunRequest(
        grid=_grid(),
        beams=_beams(),
        material=_material(),
        solver=PRSolverOptions(Nt=1, dt_normalized=1.0e-3),
        backend=_backend(),
        launch_elements=_assignments(1),
    )
    first = run_pr_timedependent(request)
    continued = continue_pr_timedependent(request, first.checkpoint, 1)

    assert np.array_equal(continued.A_initial, first.A_initial)
    assert continued.checkpoint.request.launch_elements == request.launch_elements
    assert continued.completed_steps == 2


def test_reduced_static_screened_launch_and_empty_plan_are_exact():
    base = PRStaticRunRequest(
        grid=_grid(),
        beams=_beams(),
        material=_material(),
        solver=PRStaticWorkflowOptions(max_coupled_passes=3),
        backend=_backend(),
    )
    default = run_pr_static(base)
    explicit_empty = run_pr_static(replace(base, launch_elements=()))
    assert np.array_equal(default.A_initial, explicit_empty.A_initial)
    assert default.launch_summary == explicit_empty.launch_summary

    screened = replace(base, launch_elements=_assignments(1))
    result = run_pr_static(screened)
    independent = _independent(screened)
    assert np.array_equal(result.A_initial, independent.A0)
    assert np.array_equal(result.A_initial[0], default.A_initial[0])
    assert not np.array_equal(result.A_initial[1], default.A_initial[1])


def test_transverse_static_receives_exact_screened_launch():
    request = PRTransverseStaticRunRequest(
        grid=_grid(),
        beams=_beams(),
        material=_material(),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=2),
        backend=_backend(),
        launch_elements=_assignments(1),
    )
    result = run_pr_transverse_static(request)
    independent = _independent(request)
    assert np.array_equal(result.A_initial, independent.A0)
    assert result.source_intensity_stack.shape == (1, 12, 8)
    assert result.launch_summary["channel_throughput_fractions"][0] == pytest.approx(1.0)
    assert result.launch_summary["channel_throughput_fractions"][1] < 1.0


@pytest.mark.parametrize("kind", ("td", "static", "transverse"))
def test_explicit_initial_A_with_launch_elements_is_rejected(kind):
    fields = np.zeros((2, 12, 8), dtype=np.complex128)
    common = dict(
        grid=_grid(),
        beams=_beams(),
        material=_material(),
        backend=_backend(),
        launch_elements=_assignments(1),
        initial_A=fields,
    )
    if kind == "td":
        request = PRRunRequest(**common, solver=PRSolverOptions(Nt=0))
        runner = run_pr_timedependent
    elif kind == "static":
        request = PRStaticRunRequest(**common)
        runner = run_pr_static
    else:
        request = PRTransverseStaticRunRequest(**common)
        runner = run_pr_transverse_static
    with pytest.raises(ValueError, match="already-prepared runtime launch"):
        runner(request)


def test_experiment_and_static_transports_share_screen_plan_encoding(tmp_path):
    static = PRStaticRunRequest(
        grid=_grid(), beams=_beams(), launch_elements=_assignments(1)
    )
    path = tmp_path / "screen-plan.lcprop.json"
    save_experiment(
        static,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_STATIC_WORKFLOW,
    )
    assert load_experiment(path).request == static
    encoded_static = encode_pr_static_transport_request(static)
    decoded_static = decode_pr_static_transport_request(
        encoded_static.payload.metadata,
        encoded_static.payload.arrays,
    )
    assert decoded_static == static

    transverse = PRTransverseStaticRunRequest(
        grid=_grid(), beams=_beams(), launch_elements=_assignments(1)
    )
    encoded_transverse = encode_pr_transverse_static_transport_request(transverse)
    decoded_transverse = decode_pr_transverse_static_transport_request(
        encoded_transverse.payload.metadata,
        encoded_transverse.payload.arrays,
    )
    assert decoded_transverse == transverse

    td = PRRunRequest(
        grid=_grid(),
        beams=_beams(),
        solver=PRSolverOptions(Nt=0),
        launch_elements=_assignments(1),
    )
    checkpoint = run_pr_timedependent(td).checkpoint
    with pytest.raises(ValueError, match="launch_elements"):
        save_pr_checkpoint(checkpoint, tmp_path)


def test_static_transport_preserves_embedded_user_image_and_screen_geometry():
    encoded_bytes = b"portable-user-image"
    pixels = np.asarray([[0.1, 0.8, 0.3], [1.0, 0.2, 0.6]], dtype=np.float32)
    source = RasterSource(
        source_kind="user",
        display_name="user chart",
        basename="chart.png",
        sha256=hashlib.sha256(encoded_bytes).hexdigest(),
        width=3,
        height=2,
        encoded_format="png",
        decoded_mode="L",
        grayscale=pixels,
        encoded_bytes=encoded_bytes,
    )
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            center_x_um=2.5,
            center_y_um=-1.5,
            width_um=9.0,
            height_um=5.0,
            outside_intensity_transmission=0.25,
            boundary_policy="clip",
        ),
        invert=True,
    )
    request = PRStaticRunRequest(
        grid=_grid(),
        beams=_beams(3),
        launch_elements=(
            ChannelLaunchElements(0, (screen,)),
            ChannelLaunchElements(2, (_screen([[1.0, 0.2], [0.4, 0.8]]),)),
        ),
    )
    encoded = encode_pr_static_transport_request(request)
    decoded = decode_pr_static_transport_request(
        encoded.payload.metadata, encoded.payload.arrays
    )
    assert decoded == request
    restored = decoded.launch_elements[0].elements[0]
    assert restored.source.encoded_bytes == encoded_bytes
    assert restored.placement == screen.placement
    assert restored.invert


def _wait_for(app, predicate, *, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for PR GUI run")
        time.sleep(0.002)
    app.processEvents()


def test_pr_gui_build_and_registered_dispatch_carry_screen_plan(app):
    window = PRMainWindow()
    window.grid_panel.Nx.setValue(8)
    window.grid_panel.Ny.setValue(8)
    window.grid_panel.dz_um.setValue(2.0)
    window.grid_panel.z_length_um.setValue(2.0)
    window.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    editor = window.beam_panel.input_screen_editor
    assert editor.editor_enabled
    editor.set_source(RasterSource.from_array([[0.0, 1.0], [1.0, 0.0]]))
    editor.width_um.setValue(10.0)
    editor.height_um.setValue(10.0)

    request = window.build_request()
    assert len(request.launch_elements) == 1
    calls = []
    original = window.runner.run_registered

    def observed(material_id, workflow_id, supplied, **kwargs):
        calls.append((material_id, workflow_id, supplied))
        return original(material_id, workflow_id, supplied, **kwargs)

    window.runner.run_registered = observed
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)
    assert calls[0][1] == PR_STATIC_WORKFLOW
    assert calls[0][2].launch_elements == request.launch_elements
    assert np.array_equal(window.last_result.A_initial, _independent(request).A0)

    window.input_panel.input_mode.setCurrentIndex(1)
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.beam_panel))
    assert window.beam_panel.launch_elements() == request.launch_elements
    window.close()


def test_applying_supported_empty_plan_clears_transient_editor_assignments(app):
    window = PRMainWindow()
    editor = window.beam_panel.input_screen_editor
    editor.set_source(RasterSource.from_array([[0.0, 1.0], [1.0, 0.0]]))
    assert len(window.beam_panel.launch_elements()) == 1
    request = replace(window.build_request(), launch_elements=())

    apply_pr_request(
        request,
        material_panel=window.material_panel,
        beam_panel=window.beam_panel,
        grid_panel=window.grid_panel,
        evolution_panel=window.evolution_panel,
    )

    assert window.beam_panel.launch_elements() == ()
    window.close()
