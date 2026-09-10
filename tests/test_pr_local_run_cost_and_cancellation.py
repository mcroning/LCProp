from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.run_cost import LocalRunCostClass, classify_pr_run_cost
from lcprop.pr.specs import PRRunRequest
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _request(
    *,
    nx: int = 24,
    ny: int = 24,
    nz: int = 2,
    max_coupled: int = 5,
) -> PRTransverseStaticRunRequest:
    return PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=nx,
            Ny=ny,
            x_aperture_um=40.0,
            y_aperture_um=40.0,
            dz_um=5.0,
            z_length_um=5.0 * nz,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=10.0,
            coherence_group="cost-guard",
        ),)),
        solver=PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=max_coupled,
        ),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def test_cost_classifier_covers_required_local_and_slurm_cases():
    small_full = _request()
    large_full = _request(nx=256, ny=256, nz=100, max_coupled=20)
    extreme_full = _request(nx=1024, ny=1024, nz=300, max_coupled=20)
    large_linearized = replace(
        extreme_full,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.0,
        ),
    )
    small_reduced = PRRunRequest(
        grid=small_full.grid,
        beams=small_full.beams,
        backend=small_full.backend,
    )

    assert classify_pr_run_cost(
        small_reduced, execution_target="local"
    ).classification is LocalRunCostClass.NORMAL
    assert classify_pr_run_cost(
        small_full, execution_target="local"
    ).classification is LocalRunCostClass.NORMAL
    assert classify_pr_run_cost(
        large_full, execution_target="local"
    ).classification is LocalRunCostClass.POTENTIALLY_EXPENSIVE
    assert classify_pr_run_cost(
        extreme_full, execution_target="local"
    ).classification is LocalRunCostClass.VERY_EXPENSIVE
    assert classify_pr_run_cost(
        large_linearized, execution_target="local"
    ).classification is LocalRunCostClass.POTENTIALLY_EXPENSIVE
    assert classify_pr_run_cost(
        large_linearized, execution_target="local"
    ).classification is not classify_pr_run_cost(
        extreme_full, execution_target="local"
    ).classification
    assert classify_pr_run_cost(
        extreme_full, execution_target="slurm"
    ).classification is LocalRunCostClass.NORMAL


def test_cost_warning_contains_scientific_context_and_safe_stop_semantics():
    assessment = classify_pr_run_cost(
        _request(nx=256, ny=256, nz=100, max_coupled=20),
        execution_target="local",
    )
    text = PRMainWindow._local_cost_warning_text(assessment)

    assert "Full transverse PR transport" in text
    assert "Fully nonlinear" in text
    assert "256 x 256 x 100" in text
    assert "Execution: Local" in text
    assert "Use Slurm/H200" in text
    assert "safe solver cancellation checkpoint" in text
    assert "last accepted state" in text


def test_extreme_confirmation_dialog_exposes_all_three_choices(
    app, monkeypatch
):
    import lcprop.pr.gui.main_window as window_module

    buttons = []

    class FakeButton:
        def __init__(self, text, role):
            self.text = text
            self.role = role
            self.enabled = True

        def setEnabled(self, enabled):
            self.enabled = enabled

    class FakeMessageBox:
        class Icon:
            Warning = "warning"

        class ButtonRole:
            RejectRole = "reject"
            ActionRole = "action"
            DestructiveRole = "destructive"

        def __init__(self, _parent):
            self.default = None

        def setIcon(self, _icon):
            pass

        def setWindowTitle(self, title):
            assert title == "Very expensive local calculation"

        def setText(self, text):
            assert "Use Slurm/H200" in text

        def addButton(self, text, role):
            button = FakeButton(text, role)
            buttons.append(button)
            return button

        def setDefaultButton(self, button):
            self.default = button

        def exec(self):
            pass

        def clickedButton(self):
            return buttons[2]

    monkeypatch.setattr(window_module, "QMessageBox", FakeMessageBox)
    window = PRMainWindow(slurm_runner=_DummySlurmRunner())
    assessment = classify_pr_run_cost(
        _request(nx=1024, ny=1024, nz=300, max_coupled=20),
        execution_target="local",
    )

    assert window._confirm_very_expensive_local_run(assessment) == "run_local"
    assert [(button.text, button.role) for button in buttons] == [
        ("Cancel", "reject"),
        ("Use Slurm/H200", "action"),
        ("Run locally anyway", "destructive"),
    ]
    assert all(button.enabled for button in buttons)
    window.close()


class _DummySlurmRunner:
    name = "Slurm"
    registered_operations = ()
    supports_parallel_sweeps = False


@pytest.mark.parametrize(
    ("action", "starts", "selects_slurm"),
    (
        ("cancel", False, False),
        ("run_local", True, False),
        ("use_slurm", False, True),
    ),
)
def test_extreme_gui_guard_requires_explicit_choice(
    app, monkeypatch, action, starts, selects_slurm
):
    window = PRMainWindow(slurm_runner=_DummySlurmRunner())
    request = _request(nx=1024, ny=1024, nz=300, max_coupled=20)
    started = []
    monkeypatch.setattr(window, "build_request", lambda: request)
    monkeypatch.setattr(window, "describe_request", lambda _request: "summary")
    monkeypatch.setattr(
        window,
        "_confirm_very_expensive_local_run",
        lambda _assessment: action,
    )
    monkeypatch.setattr(
        window, "_start_background", lambda *args, **kwargs: started.append(args)
    )

    window.run_clicked()

    assert bool(started) is starts
    assert (window.execution_target_selector.currentData() == "slurm") is (
        selects_slurm
    )
    window.close()


def test_normal_gui_run_has_no_cost_dialog(app, monkeypatch):
    window = PRMainWindow()
    request = _request()
    started = []
    monkeypatch.setattr(window, "build_request", lambda: request)
    monkeypatch.setattr(window, "describe_request", lambda _request: "summary")
    monkeypatch.setattr(
        window,
        "_show_potentially_expensive_local_warning",
        lambda _assessment: pytest.fail("normal run displayed a warning"),
    )
    monkeypatch.setattr(
        window,
        "_confirm_very_expensive_local_run",
        lambda _assessment: pytest.fail("normal run requested confirmation"),
    )
    monkeypatch.setattr(
        window, "_start_background", lambda *args, **kwargs: started.append(args)
    )

    window.run_clicked()

    assert len(started) == 1
    window.close()


def test_potentially_expensive_gui_run_warns_but_does_not_require_override(
    app, monkeypatch
):
    window = PRMainWindow()
    request = _request(nx=256, ny=256, nz=100, max_coupled=20)
    warnings = []
    monkeypatch.setattr(
        window,
        "_show_potentially_expensive_local_warning",
        warnings.append,
    )
    monkeypatch.setattr(
        window,
        "_confirm_very_expensive_local_run",
        lambda _assessment: pytest.fail("moderate warning requested override"),
    )

    assert window._local_run_cost_guard(request) is True
    assert [item.classification for item in warnings] == [
        LocalRunCostClass.POTENTIALLY_EXPENSIVE
    ]
    window.close()


def test_slurm_selection_bypasses_local_cost_dialog(app, monkeypatch):
    window = PRMainWindow(slurm_runner=_DummySlurmRunner())
    request = _request(nx=1024, ny=1024, nz=300, max_coupled=20)
    window.execution_target_selector.setCurrentIndex(1)
    monkeypatch.setattr(
        window,
        "_show_potentially_expensive_local_warning",
        lambda _assessment: pytest.fail("Slurm request displayed local warning"),
    )
    monkeypatch.setattr(
        window,
        "_confirm_very_expensive_local_run",
        lambda _assessment: pytest.fail("Slurm request requested local confirmation"),
    )

    assert window._local_run_cost_guard(request) is True
    window.close()


def test_repeated_gui_stop_is_idempotent_and_describes_safe_checkpoint(app):
    window = PRMainWindow()
    token = CancellationToken()
    window._background_running = True
    window._cancellation_token = token
    window._active_request = _request()

    window.stop_clicked()
    window.stop_clicked()

    assert token.is_cancelled()
    assert window.run_status == "stopping"
    assert window.status_label.text() == (
        "Stopping at next safe solver checkpoint…"
    )
    assert "last accepted state will be preserved" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window._background_running = False
    window._cancellation_token = None
    window._active_request = None
    window.close()


def test_pre_cancelled_full_transverse_run_keeps_initial_accepted_state():
    request = _request()
    token = CancellationToken()
    token.cancel()

    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "coupled_iteration_boundary"
    )


def test_cancellation_during_material_residual_discards_trial(monkeypatch):
    import lcprop.pr.transverse.static as material_module

    request = _request(max_coupled=5)
    token = CancellationToken()
    original = material_module._static_equilibrium_residual_arrays
    calls = 0

    def cancelling_residual(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        # The outer workflow first evaluates the accepted source volume (two
        # planes); cancel during the first subsequent Newton residual.
        if calls == 3:
            token.cancel()
        return result

    monkeypatch.setattr(
        material_module, "_static_equilibrium_residual_arrays", cancelling_residual
    )
    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "material_residual_construction"
    )


def test_cancellation_during_pcg_discards_trial(monkeypatch):
    import lcprop.pr.transverse.static as material_module

    request = _request(max_coupled=5)
    token = CancellationToken()
    original = material_module._jacobian_action
    calls = 0

    def cancelling_action(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            token.cancel()
        return result

    monkeypatch.setattr(material_module, "_jacobian_action", cancelling_action)
    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "material_pcg_iteration"
    )


def test_cancellation_between_linearized_material_planes_discards_trial(
    monkeypatch,
):
    import lcprop.pr.transverse.static_workflow as workflow_module

    request = replace(
        _request(max_coupled=5),
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=0.2,
        ),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.0,
        ),
    )
    token = CancellationToken()
    original = workflow_module.solve_pr_biased_linearized_reference
    calls = 0

    def cancelling_solve(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        # The first two calls establish the accepted source residual. Cancel
        # after the first plane of the outer material proposal.
        if calls == 3:
            token.cancel()
        return result

    monkeypatch.setattr(
        workflow_module, "solve_pr_biased_linearized_reference", cancelling_solve
    )
    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "linearized_material_plane"
    )


def test_cancellation_between_material_planes_discards_volume_trial(monkeypatch):
    import lcprop.pr.transverse.static as material_module

    request = _request(max_coupled=5)
    token = CancellationToken()
    original = material_module._solve_plane
    calls = 0

    def cancelling_plane(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            token.cancel()
        return result

    monkeypatch.setattr(material_module, "_solve_plane", cancelling_plane)
    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "material_plane_boundary"
    )


def test_cancellation_during_coupled_optical_trial_discards_trial(monkeypatch):
    import lcprop.pr.transverse.static_workflow as workflow_module

    request = _request(max_coupled=5)
    token = CancellationToken()
    original = workflow_module.advance_pr_slice_with_midpoint_source
    calls = 0

    def cancelling_advance(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        # Two slices establish the accepted source. Cancel during the first
        # slice of the subsequent discardable coupled optical trial.
        if calls == 3:
            token.cancel()
        return result

    monkeypatch.setattr(
        workflow_module, "advance_pr_slice_with_midpoint_source", cancelling_advance
    )
    result = run_pr_transverse_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["cancellation_observed_stage"] == (
        "coupled_optical_z_march"
    )


def test_cancellation_after_one_accept_preserves_that_exact_state():
    request = _request(max_coupled=5)
    token = CancellationToken()
    accepted = []

    def stop_after_accept(progress):
        accepted.append(progress.latest_field_state["psi_current"].copy())
        token.cancel()

    result = run_pr_transverse_static(
        request,
        cancellation_token=token,
        progress_callback=stop_after_accept,
    )

    assert result.status == "cancelled"
    assert result.completed_coupled_iterations == 1
    np.testing.assert_array_equal(result.psi_final, accepted[0])
    assert result.diagnostics["cancellation_observed_stage"] == (
        "coupled_iteration_boundary"
    )


def test_uncancelled_token_is_bitwise_numerically_inert():
    request = _request(max_coupled=2)
    baseline = run_pr_transverse_static(request)
    controlled = run_pr_transverse_static(
        request, cancellation_token=CancellationToken()
    )

    assert controlled.status == baseline.status
    assert controlled.converged is baseline.converged
    assert controlled.iteration_records == baseline.iteration_records
    np.testing.assert_array_equal(controlled.A_final, baseline.A_final)
    np.testing.assert_array_equal(controlled.psi_final, baseline.psi_final)
    np.testing.assert_array_equal(
        controlled.source_intensity_stack, baseline.source_intensity_stack
    )
