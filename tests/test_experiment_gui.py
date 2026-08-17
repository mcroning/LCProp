from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
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
from lcprop.persistence import ExperimentMaterialError, save_experiment
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.specs import PR_MATERIAL_ID
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import PR_STATIC_WORKFLOW


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
