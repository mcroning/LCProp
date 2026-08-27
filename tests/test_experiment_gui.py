from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import lcprop.gui.experiment_files as experiment_files
import lcprop.lc.gui.main_window as lc_main_window
import lcprop.pr.gui.main_window as pr_main_window
from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.lc.gui.request_adapter import (
    LC_STATIC_EXPERIMENT,
    LC_STATIC_WORKFLOW_ID,
    LC_TIMEDEPENDENT_EXPERIMENT,
    LC_TIMEDEPENDENT_WORKFLOW_ID,
)
from lcprop.lc.requests import OutputOptions
from lcprop.persistence import (
    ExperimentMaterialError,
    ExperimentPayloadError,
    ExperimentRuntimeStateError,
    load_experiment,
    save_experiment,
)
from lcprop.pr.gui.image_input_panel import PR_IMAGE_AMPLIFICATION_INPUT_MODE
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.image_amplification import PR_IMAGE_AMPLIFICATION_WORKFLOW
from lcprop.pr.specs import PR_MATERIAL_ID
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import PR_STATIC_WORKFLOW
from lcprop.pr.transverse.static_workflow import PR_TRANSVERSE_STATIC_WORKFLOW


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_angle_stack():
    model = pytest.importorskip("launchplane.model")
    return model.BeamStackDefinition(
        beams=(
            model.BeamDefinition.from_launch_angles(
                name="angle beam",
                wavelength_um=0.633,
                power_mW=1.0,
                x_um=-2.0,
                y_um=3.0,
                waist_x_um=6.0,
                waist_y_um=7.0,
                angle_x_rad=0.02,
                angle_y_rad=-0.01,
                launch_medium_index=1.6,
                phase_rad=0.3,
                coherence_group="laser-a",
            ),
            model.BeamDefinition(
                name="disabled q beam",
                x_um=500.0,
                y_um=-400.0,
                tilt_x_rad_per_um=0.4,
                tilt_y_rad_per_um=-0.2,
                launch_medium_index=None,
                launch_input_mode="transverse_wavevector",
                enabled=False,
            ),
        )
    )


def _make_coherent_two_beam_angle_stack():
    model = pytest.importorskip("launchplane.model")
    return model.BeamStackDefinition(
        beams=(
            model.BeamDefinition.from_launch_angles(
                name="signal",
                wavelength_um=0.633,
                power_mW=1.75,
                x_um=-12.0,
                y_um=2.5,
                waist_x_um=18.0,
                waist_y_um=19.0,
                angle_x_rad=0.013,
                angle_y_rad=-0.004,
                launch_medium_index=2.4,
                phase_rad=0.25,
                coherence_group="pr-laser",
            ),
            model.BeamDefinition.from_launch_angles(
                name="pump",
                wavelength_um=0.633,
                power_mW=3.5,
                x_um=13.0,
                y_um=-1.5,
                waist_x_um=21.0,
                waist_y_um=22.0,
                angle_x_rad=-0.011,
                angle_y_rad=0.006,
                launch_medium_index=2.4,
                phase_rad=-0.35,
                coherence_group="pr-laser",
            ),
        )
    )


def _configure_image_amplification_window(window, image_path, workflow_id):
    model = pytest.importorskip("launchplane.model")
    mode_index = window.input_panel.input_mode.findData(
        PR_IMAGE_AMPLIFICATION_INPUT_MODE
    )
    window.input_panel.input_mode.setCurrentIndex(mode_index)
    window.evolution_panel.set_workflow_id(workflow_id)
    window.grid_panel.Nx.setValue(32)
    window.grid_panel.Ny.setValue(16)
    window.grid_panel.x_aperture_um.setValue(80.0)
    window.grid_panel.y_aperture_um.setValue(40.0)
    window.grid_panel.z_length_um.setValue(20.0)
    window.grid_panel.dz_um.setValue(10.0)
    window.beam_panel.set_beam_stack_definition(
        model.BeamStackDefinition(
            beams=(
                model.BeamDefinition(
                    name="pump",
                    wavelength_um=0.633,
                    power_mW=2.5,
                    x_um=-8.0,
                    y_um=1.0,
                    waist_x_um=12.0,
                    waist_y_um=7.0,
                    tilt_x_rad_per_um=0.2,
                    phase_rad=0.35,
                    coherence_group="image-laser",
                ),
                model.BeamDefinition(
                    name="signal",
                    wavelength_um=0.633,
                    power_mW=0.25,
                    x_um=9.0,
                    y_um=-2.0,
                    waist_x_um=10.0,
                    waist_y_um=6.0,
                    tilt_x_rad_per_um=-0.2,
                    phase_rad=-0.15,
                    coherence_group="image-laser",
                ),
            )
        )
    )
    editor = window.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.load_user_image(image_path)
    editor.width_um.setValue(18.0)
    editor.height_um.setValue(9.0)
    editor.center_x_um.setValue(11.0)
    editor.center_y_um.setValue(-3.0)
    editor.invert.setChecked(True)
    window.input_panel.set_role_indices(0, 1)


@pytest.mark.parametrize(
    "workflow_id",
    ("pr_timedependent", PR_STATIC_WORKFLOW, PR_TRANSVERSE_STATIC_WORKFLOW),
)
def test_pr_image_experiment_survives_original_file_deletion(
    app,
    tmp_path,
    workflow_id,
):
    source_path = tmp_path / f"source-{workflow_id}.png"
    image = QImage(11, 7, QImage.Format.Format_RGBA8888)
    image.fill(0x6080A0FF)
    assert image.save(str(source_path), "PNG")
    encoded = source_path.read_bytes()
    encoded_sha = hashlib.sha256(encoded).hexdigest()

    source = PRMainWindow()
    _configure_image_amplification_window(source, source_path, workflow_id)
    expected = source.build_request()
    path = tmp_path / f"image-{workflow_id}.lcprop.json"
    source.save_experiment_to(path)
    source.close()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["request_payload"]["base_request"]["launch_elements"] == []
    assert len(
        document["request_payload"]["launch_configuration"]["channel_elements"]
    ) == 1
    source_path.unlink()

    loaded = load_experiment(path, expected_material_id=PR_MATERIAL_ID)
    assert loaded.workflow_id == PR_IMAGE_AMPLIFICATION_WORKFLOW
    assert loaded.request == expected
    assert loaded.request.source.sha256 == encoded_sha
    assert loaded.request.source.encoded_bytes == encoded
    assert loaded.request.source.basename == source_path.name
    assert loaded.request.source.encoded_format == "png"
    assert (loaded.request.source.width, loaded.request.source.height) == (11, 7)
    assert loaded.request.source.decoded_mode == "L"
    screen = loaded.request.launch_configuration.channel_elements[0].elements[0]
    assert loaded.request.source.preprocessing_policy == "raster_source_identity_v1"
    assert screen.preprocessing_policy == "even_square_nearest_transparent_v1"
    assert screen.invert is True
    assert screen.placement.center_x_um == pytest.approx(11.0)
    assert screen.placement.center_y_um == pytest.approx(-3.0)
    assert screen.placement.width_um == pytest.approx(18.0)
    assert screen.placement.height_um == pytest.approx(9.0)

    target = PRMainWindow()
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "opening an image experiment must not execute it"
    )
    target.load_experiment_from(path)
    rebuilt = target.build_request()

    assert rebuilt == expected
    assert rebuilt.base_workflow_id == workflow_id
    assert rebuilt.pump_channel_index == 0
    assert rebuilt.signal_channel_index == 1
    assert rebuilt.launch_configuration.channel_elements == (
        expected.launch_configuration.channel_elements
    )
    assert rebuilt.source.encoded_bytes == encoded
    assert target.input_panel.is_image_amplification()
    assert target.evolution_panel.workflow_id() == workflow_id
    target.close()


def test_pr_image_experiment_save_open_buttons_round_trip(app, tmp_path, monkeypatch):
    source_path = tmp_path / "button-source.png"
    image = QImage(9, 5, QImage.Format.Format_RGBA8888)
    image.fill(0x406080FF)
    assert image.save(str(source_path), "PNG")
    source = PRMainWindow()
    _configure_image_amplification_window(
        source,
        source_path,
        PR_TRANSVERSE_STATIC_WORKFLOW,
    )
    source.evolution_panel.backend.setCurrentText("cupy")
    source.evolution_panel.precision.setCurrentText("float32")
    expected = source.build_request()
    path = tmp_path / "button-image.lcprop.json"
    monkeypatch.setattr(
        pr_main_window,
        "choose_experiment_save_path",
        lambda _parent: path,
    )

    source.experiment_file_buttons.save_button.click()
    app.processEvents()
    assert source.status_label.text() == "Experiment saved"
    assert path.is_file()

    target = PRMainWindow()
    monkeypatch.setattr(
        pr_main_window,
        "choose_experiment_open_path",
        lambda _parent: path,
    )
    target.experiment_file_buttons.open_button.click()
    app.processEvents()

    assert target.status_label.text() == "Experiment opened"
    assert target.build_request() == expected
    assert target.evolution_panel.backend.currentText() == "cupy"
    assert target.evolution_panel.precision.currentText() == "float32"
    source.close()
    target.close()


def test_pr_image_experiment_rejects_runtime_base_state(app, tmp_path):
    source_path = tmp_path / "runtime-source.png"
    image = QImage(8, 6, QImage.Format.Format_RGBA8888)
    image.fill(0x204060FF)
    assert image.save(str(source_path), "PNG")
    window = PRMainWindow()
    _configure_image_amplification_window(
        window,
        source_path,
        "pr_timedependent",
    )
    request = window.build_request()
    request = replace(
        request,
        base_request=replace(
            request.base_request,
            initial_A=np.zeros((1, 1, 1), dtype=np.complex128),
        ),
    )

    with pytest.raises(ExperimentRuntimeStateError, match="runtime state"):
        save_experiment(
            request,
            tmp_path / "runtime.lcprop.json",
            material_id=PR_MATERIAL_ID,
            workflow_id=PR_IMAGE_AMPLIFICATION_WORKFLOW,
        )
    window.close()


def test_legacy_pr_schema_one_empty_launch_plan_still_loads(app, tmp_path):
    window = PRMainWindow()
    request = window.build_request()
    path = tmp_path / "legacy-pr.lcprop.json"
    window.save_experiment_to(path)
    window.close()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["request_payload"]["schema_version"] = 1
    document["request_payload"].pop("launch_elements")
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_experiment(path, expected_material_id=PR_MATERIAL_ID)
    assert loaded.request == request


def test_pr_image_experiment_rejects_corrupt_embedded_source(app, tmp_path):
    source_path = tmp_path / "source.png"
    image = QImage(8, 6, QImage.Format.Format_RGBA8888)
    image.fill(0x204060FF)
    assert image.save(str(source_path), "PNG")
    source = PRMainWindow()
    _configure_image_amplification_window(
        source,
        source_path,
        "pr_timedependent",
    )
    path = tmp_path / "corrupt-image.lcprop.json"
    source.save_experiment_to(path)
    source.close()
    document = json.loads(path.read_text(encoding="utf-8"))
    image_payload = document["request_payload"]["launch_configuration"][
        "channel_elements"
    ][0]["elements"][0]["source"]
    image_payload["encoded_bytes_base64"] = "AAAA"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExperimentPayloadError, match="checksum mismatch"):
        load_experiment(path, expected_material_id=PR_MATERIAL_ID)


def test_pr_image_experiment_rejects_unregistered_base_operation(app, tmp_path):
    source_path = tmp_path / "source.png"
    image = QImage(8, 6, QImage.Format.Format_RGBA8888)
    image.fill(0x204060FF)
    assert image.save(str(source_path), "PNG")
    source = PRMainWindow()
    _configure_image_amplification_window(
        source,
        source_path,
        "pr_timedependent",
    )
    path = tmp_path / "unsupported-operation.lcprop.json"
    source.save_experiment_to(path)
    source.close()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["request_payload"]["base_workflow_id"] = "pr_removed_operation"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExperimentPayloadError, match="not registered"):
        load_experiment(path, expected_material_id=PR_MATERIAL_ID)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pump_channel_index", 7, "canonical enabled beam channel"),
        ("signal_channel_index", 0, "must be distinct"),
    ),
)
def test_pr_image_experiment_rejects_invalid_roles(
    app,
    tmp_path,
    field,
    value,
    message,
):
    source_path = tmp_path / f"invalid-{field}.png"
    image = QImage(8, 6, QImage.Format.Format_RGBA8888)
    image.fill(0x204060FF)
    assert image.save(str(source_path), "PNG")
    source = PRMainWindow()
    _configure_image_amplification_window(
        source,
        source_path,
        "pr_timedependent",
    )
    path = tmp_path / f"invalid-{field}.lcprop.json"
    source.save_experiment_to(path)
    source.close()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["request_payload"][field] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExperimentPayloadError, match=message):
        load_experiment(path, expected_material_id=PR_MATERIAL_ID)


@pytest.mark.parametrize(
    ("experiment", "workflow_id"),
    [
        (LC_STATIC_EXPERIMENT, LC_STATIC_WORKFLOW_ID),
        (LC_TIMEDEPENDENT_EXPERIMENT, LC_TIMEDEPENDENT_WORKFLOW_ID),
    ],
)
def test_lc_experiment_cross_session_round_trip(
    app, tmp_path, experiment, workflow_id
):
    source = LCPropMainWindow()
    source.experiment_panel.set_current_experiment(experiment)
    source.beam_panel.set_beam_stack_definition(_make_angle_stack())
    expected = (
        source.build_request()
        if workflow_id == LC_STATIC_WORKFLOW_ID
        else source._build_timedependent_base_request()
    )
    path = tmp_path / f"lc-{workflow_id}.lcprop.json"
    source.save_experiment_to(path)

    target = LCPropMainWindow()
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "opening an experiment must not execute it"
    )
    loaded = target.load_experiment_from(path)

    rebuilt = (
        target.build_request()
        if workflow_id == LC_STATIC_WORKFLOW_ID
        else target._build_timedependent_base_request()
    )
    assert loaded.request == expected == rebuilt
    assert target.experiment_panel.current_experiment() == experiment
    beams = target.beam_panel.beam_stack_definition.beams
    assert beams[0].launch_input_mode == "angle"
    assert beams[0].launch_medium_index == 1.6
    assert beams[1].enabled is False
    assert (beams[1].x_um, beams[1].y_um) == (500.0, -400.0)


@pytest.mark.parametrize("workflow_id", ["pr_timedependent", PR_STATIC_WORKFLOW])
def test_pr_experiment_cross_session_round_trip(app, tmp_path, workflow_id):
    source = PRMainWindow()
    source.evolution_panel.set_workflow_id(workflow_id)
    source.beam_panel.set_beam_stack_definition(_make_angle_stack())
    expected = source.build_request()
    path = tmp_path / f"pr-{workflow_id}.lcprop.json"
    source.save_experiment_to(path)

    target = PRMainWindow()
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "opening an experiment must not execute it"
    )
    loaded = target.load_experiment_from(path)

    assert loaded.request == expected == target.build_request()
    assert target.evolution_panel.workflow_id() == workflow_id
    beams = target.beam_panel.beam_stack_definition.beams
    assert beams[0].launch_input_mode == "angle"
    assert beams[0].launch_medium_index == 1.6
    assert beams[1].enabled is False
    assert (beams[1].x_um, beams[1].y_um) == (500.0, -400.0)


def test_pr_transverse_static_save_open_buttons_round_trip_exactly(
    app, tmp_path, monkeypatch
):
    source = PRMainWindow()
    source.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    source.evolution_panel.backend.setCurrentText("cupy")
    source.evolution_panel.precision.setCurrentText("float32")
    source.evolution_panel.max_coupled_passes.setValue(37)
    source.evolution_panel.optical_substeps.setValue(5)
    source.beam_panel.set_beam_stack_definition(
        _make_coherent_two_beam_angle_stack()
    )
    expected = source.build_request()
    path = tmp_path / "coherent-transverse-static.lcprop.json"
    monkeypatch.setattr(
        pr_main_window,
        "choose_experiment_save_path",
        lambda _parent: path,
    )

    source.experiment_file_buttons.save_button.click()
    app.processEvents()

    assert path.is_file()
    assert source.status_label.text() == "Experiment saved"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["workflow_id"] == PR_TRANSVERSE_STATIC_WORKFLOW
    assert document["request_payload"]["backend"] == {
        "backend": "cupy",
        "precision": "float32",
        "verbose": False,
    }

    target = PRMainWindow()
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "opening an experiment must not execute it"
    )
    monkeypatch.setattr(
        pr_main_window,
        "choose_experiment_open_path",
        lambda _parent: path,
    )
    target.experiment_file_buttons.open_button.click()
    app.processEvents()

    assert target.status_label.text() == "Experiment opened"
    assert target.build_request() == expected
    assert target.evolution_panel.workflow_id() == PR_TRANSVERSE_STATIC_WORKFLOW
    assert target.evolution_panel.backend.currentText() == "cupy"
    assert target.evolution_panel.precision.currentText() == "float32"
    restored = target.beam_panel.beam_stack_definition.beams
    assert tuple(beam.coherence_group for beam in restored) == (
        "pr-laser",
        "pr-laser",
    )
    assert tuple(beam.launch_input_mode for beam in restored) == (
        "angle",
        "angle",
    )
    assert restored[0].angle_x_rad == pytest.approx(0.013)
    assert restored[1].angle_y_rad == pytest.approx(0.006)


def test_absent_presentation_uses_safe_canonical_q_mode(app, tmp_path):
    source = PRMainWindow()
    source.beam_panel.set_beam_stack_definition(_make_angle_stack())
    request = source.build_request()
    path = tmp_path / "no-presentation.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id="pr_timedependent",
    )

    target = PRMainWindow()
    target.load_experiment_from(path)

    beam = target.beam_panel.beam_stack_definition.beams[0]
    assert beam.launch_input_mode == "transverse_wavevector"
    assert beam.launch_medium_index is None
    assert target.build_request().beams == request.beams


def test_wrong_material_rejected_before_lc_state_or_checkpoint_changes(
    app, tmp_path
):
    source = PRMainWindow()
    path = tmp_path / "pr.lcprop.json"
    source.save_experiment_to(path)
    target = LCPropMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_timedependent_checkpoint = checkpoint

    with pytest.raises(ExperimentMaterialError) as captured:
        target.load_experiment_from(path)

    assert captured.value.actual_material_id == PR_MATERIAL_ID
    assert captured.value.expected_material_id == LC_MATERIAL_ID
    after = target._capture_experiment_gui_state()
    assert after == before
    assert target.last_timedependent_checkpoint is checkpoint


def test_wrong_material_rejected_before_pr_state_or_checkpoint_changes(
    app, tmp_path
):
    source = LCPropMainWindow()
    path = tmp_path / "lc.lcprop.json"
    source.save_experiment_to(path)
    target = PRMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_checkpoint = checkpoint

    with pytest.raises(ExperimentMaterialError) as captured:
        target.load_experiment_from(path)

    assert captured.value.actual_material_id == LC_MATERIAL_ID
    assert captured.value.expected_material_id == PR_MATERIAL_ID
    after = target._capture_experiment_gui_state()
    assert after == before
    assert target.last_checkpoint is checkpoint


def test_lc_open_button_warns_that_pr_file_requires_pr_application(
    app, tmp_path, monkeypatch
):
    source = PRMainWindow()
    path = tmp_path / "pr-for-lc-warning.lcprop.json"
    source.save_experiment_to(path)
    target = LCPropMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_timedependent_checkpoint = checkpoint
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "wrong-material open must not execute"
    )
    warnings = []
    monkeypatch.setattr(
        lc_main_window,
        "choose_experiment_open_path",
        lambda _parent: path,
    )
    monkeypatch.setattr(
        experiment_files.QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((parent, title, text)),
    )

    target.open_experiment_clicked()

    assert warnings == [
        (
            target,
            "Experiment Material Mismatch",
            "This experiment is for Photorefractive (PR). "
            "Open it with the PR application.",
        )
    ]
    assert target._capture_experiment_gui_state() == before
    assert target.last_timedependent_checkpoint is checkpoint


def test_pr_open_button_warns_that_lc_file_requires_lc_application(
    app, tmp_path, monkeypatch
):
    source = LCPropMainWindow()
    path = tmp_path / "lc-for-pr-warning.lcprop.json"
    source.save_experiment_to(path)
    target = PRMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_checkpoint = checkpoint
    target._run_registered = lambda *_args, **_kwargs: pytest.fail(
        "wrong-material open must not execute"
    )
    warnings = []
    monkeypatch.setattr(
        pr_main_window,
        "choose_experiment_open_path",
        lambda _parent: path,
    )
    monkeypatch.setattr(
        experiment_files.QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((parent, title, text)),
    )

    target.open_experiment_clicked()

    assert warnings == [
        (
            target,
            "Experiment Material Mismatch",
            "This experiment is for Liquid crystal (LC). "
            "Open it with the LC application.",
        )
    ]
    assert target._capture_experiment_gui_state() == before
    assert target.last_checkpoint is checkpoint


def test_pr_unrepresentable_static_policy_is_rejected_before_changes(
    app, tmp_path
):
    source = PRMainWindow()
    source.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    request = source.build_request()
    request = replace(
        request,
        solver=replace(
            request.solver,
            material_solver=PRStaticSolverOptions(),
        ),
    )
    path = tmp_path / "explicit-static-policy.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_STATIC_WORKFLOW,
    )
    target = PRMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()

    with pytest.raises(ValueError, match="cannot represent"):
        target.load_experiment_from(path)

    assert target._capture_experiment_gui_state() == before


def test_late_pr_representability_failure_rolls_back_all_state_and_checkpoint(
    app, tmp_path
):
    source = PRMainWindow()
    request = source.build_request()
    request = replace(
        request,
        material=replace(
            request.material,
            gain_length_product=0.1234567891,
        ),
    )
    path = tmp_path / "too-precise-for-widget.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id="pr_timedependent",
    )

    target = PRMainWindow()
    target.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    target.grid_panel.Nx.setValue(96)
    target.material_panel.applied_field.setValue(2.5)
    target.beam_panel.set_beam_stack_definition(_make_angle_stack())
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_checkpoint = checkpoint

    with pytest.raises(ValueError, match="not exactly representable"):
        target.load_experiment_from(path)

    assert target._capture_experiment_gui_state() == before
    assert target.last_checkpoint is checkpoint


def test_late_lc_widget_precision_failure_rolls_back_workflow_and_checkpoint(
    app, tmp_path
):
    source = LCPropMainWindow()
    request = replace(
        source.build_request(),
        material=replace(source.build_request().material, ne=1.7000001),
    )
    path = tmp_path / "lc-too-precise-for-widget.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=LC_MATERIAL_ID,
        workflow_id=LC_STATIC_WORKFLOW_ID,
    )

    target = LCPropMainWindow()
    target.experiment_panel.set_current_experiment(LC_TIMEDEPENDENT_EXPERIMENT)
    target.grid_panel.Nx.setValue(96)
    target.physics_panel.V_bias.setValue(1.5)
    target.beam_panel.set_beam_stack_definition(_make_angle_stack())
    before = target._capture_experiment_gui_state()
    checkpoint = object()
    target.last_timedependent_checkpoint = checkpoint

    with pytest.raises(ValueError, match="not exactly representable"):
        target.load_experiment_from(path)

    assert target._capture_experiment_gui_state() == before
    assert target.last_timedependent_checkpoint is checkpoint


def test_lc_hidden_output_options_are_rejected_before_changes(app, tmp_path):
    source = LCPropMainWindow()
    request = replace(
        source.build_request(),
        output=OutputOptions(save_full=True),
    )
    path = tmp_path / "lc-hidden-output.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=LC_MATERIAL_ID,
        workflow_id=LC_STATIC_WORKFLOW_ID,
    )
    target = LCPropMainWindow()
    target.grid_panel.Nx.setValue(96)
    before = target._capture_experiment_gui_state()

    with pytest.raises(ValueError, match="output options"):
        target.load_experiment_from(path)

    assert target._capture_experiment_gui_state() == before


def test_shared_experiment_buttons_are_exposed_in_both_apps(app):
    lc = LCPropMainWindow()
    pr = PRMainWindow()
    for window in (lc, pr):
        assert window.experiment_file_buttons.save_button.text() == (
            "Save Experiment..."
        )
        assert window.experiment_file_buttons.open_button.text() == (
            "Open Experiment..."
        )
