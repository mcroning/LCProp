from __future__ import annotations

from dataclasses import replace
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from lcprop.core.execution import CancellationToken
from lcprop.gui.main_window import LCPropMainWindow
from lcprop.gui.panels.sweep_panel import SweepPanel
from lcprop.gui.retained_results import (
    ExperimentFamily,
    SolitonSource,
    TDInitialSourceMode,
    WorkflowKind,
)
from lcprop.runners.base import RunnerResult
from lcprop.products.data_model import to_run_data
from lcprop.workflows.soliton import SolitonRequest, SolitonResult, run_soliton
from lcprop.workflows.static import run_static
from lcprop.workflows.runtime import build_runtime_components
from lcprop.workflows.sweep import (
    ParameterSweepRequest,
    ParameterSweepResult,
    SolitonSweepMember,
    run_parameter_sweep,
)


def _window():
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.grid_panel.Nx.setValue(16)
    window.grid_panel.Ny.setValue(16)
    window.grid_panel.dz_um.setValue(5.0)
    window.grid_panel.z_length_um.setValue(10.0)
    window.experiment_panel.experiment.setCurrentText(
        "Time-dependent propagation"
    )
    return app, window


def _wait_for(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background soliton run")
        time.sleep(0.002)
    app.processEvents()


def _soliton_result(request, marker: float, *, status="completed"):
    nx, ny = request.base.grid.Nx, request.base.grid.Ny
    A = np.full((1, nx, ny), marker, dtype=np.complex128)
    theta = np.full((nx, ny), marker, dtype=float)
    return SolitonResult(
        metrics={"residual_rms": 0.1, "residual_max": 0.2, "beta": marker},
        A=A,
        theta=theta,
        intensity=np.abs(A[0]) ** 2,
        status=status,
        completed_iterations=1,
        total_iterations=2,
        request=request,
        usable_as_initial_condition=True,
    )


def _store_soliton(window, request, result):
    window.last_soliton_result = result
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON,
        request=request,
        result=result,
    )


def _store_static(window, request, result):
    window.last_static_result = result
    window.retained_results.store(
        family=ExperimentFamily.STANDARD,
        workflow=WorkflowKind.STATIC,
        request=request,
        result=result,
    )


def _completed_sweep(window, powers):
    base = window.build_request()
    members = []
    results = []
    samples = []
    for index, power in enumerate(powers):
        channel = replace(base.beams.channels[0], power_mW=float(power))
        source_base = replace(
            base,
            beams=replace(base.beams, channels=(channel,)),
        )
        request = SolitonRequest(base=source_base)
        result = _soliton_result(request, float(power))
        member = SolitonSweepMember(
            requested_index=index,
            requested_power_mW=float(power),
            status="completed",
            converged=True,
            result=result,
            beta=float(power) * 10.0,
            residual_rms=0.1,
            residual_max=0.2,
            completion_order=len(powers) - index,
        )
        members.append(member)
        results.append(result)
        samples.append({
            "i": index,
            "requested_power_mW": float(power),
            "converged": True,
            "beta": member.beta,
        })
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=tuple(float(power) for power in powers),
        base=SolitonRequest(base=base),
        continuation=False,
        execution="parallel",
        max_workers=len(powers),
    )
    return request, ParameterSweepResult(
        values=request.values,
        continuation=False,
        execution="parallel",
        metrics={
            "n_points": len(powers),
            "completed_points": len(powers),
            "total_points": len(powers),
            "failed_count": 0,
        },
        samples=samples,
        results=results,
        status="completed",
        completed_points=len(powers),
        total_points=len(powers),
        request=request,
        sweep_id="test-sweep",
        members=members,
    )


def test_td_last_soliton_selects_q1_not_earlier_static_s1():
    _app, window = _window()
    static_request = window.build_request()
    static_result = run_static(static_request)
    soliton_request = SolitonRequest(base=static_request, max_outer=1)
    soliton_result = _soliton_result(soliton_request, 7.0)
    _store_static(window, static_request, static_result)
    _store_soliton(window, soliton_request, soliton_result)
    window.update_run_button()

    window.use_last_soliton.setChecked(True)
    request = window.build_timedependent_request()

    np.testing.assert_array_equal(request.initial_A, soliton_result.A)
    np.testing.assert_array_equal(request.initial_theta, soliton_result.theta)
    assert not window.use_last_static.isChecked()
    window.close()


def test_fresh_td_defaults_to_beam_launch_and_ignores_retained_results():
    _app, window = _window()
    static_request = window.build_request()
    _store_static(window, static_request, run_static(static_request))
    soliton_request = SolitonRequest(base=static_request)
    _store_soliton(
        window,
        soliton_request,
        _soliton_result(soliton_request, 99.0),
    )
    window.update_run_button()

    assert (
        window.td_initial_condition_selector.currentData()
        is TDInitialSourceMode.BEAM_LAUNCH
    )
    assert window.td_initial_condition_selector.count() == 3
    assert not window.use_last_static.isVisible()
    assert not window.use_last_soliton.isVisible()
    assert not window.soliton_source_selector.isEnabled()
    request = window.build_timedependent_request()
    runtime = build_runtime_components(request)
    runner_result = window.runner.run_timedependent(request)

    assert request.initial_A is None
    assert request.initial_theta is None
    np.testing.assert_array_equal(
        runner_result.result.A_initial,
        runtime.launch.A0,
    )
    assert "Initial condition: Beam pane launch" in window.describe_request(request)
    window._active_request = request
    window._on_timedependent_finished(runner_result)
    assert window.last_timedependent_result.provenance[
        "initial_source_kind"
    ] == "beam_launch"
    window.close()


def test_newer_standard_static_does_not_overwrite_last_soliton():
    _app, window = _window()
    static_request = window.build_request()
    soliton_request = SolitonRequest(base=static_request)
    soliton_result = _soliton_result(soliton_request, 3.0)
    _store_soliton(window, soliton_request, soliton_result)
    _store_static(window, static_request, run_static(static_request))

    assert window.retained_results.last_soliton_result.result is soliton_result
    window.update_run_button()
    window.use_last_soliton.setChecked(True)
    np.testing.assert_array_equal(
        window.build_timedependent_request().initial_A, soliton_result.A
    )
    window.close()


def test_td_last_standard_static_selects_s1_not_newer_soliton():
    _app, window = _window()
    static_request = window.build_request()
    static_result = run_static(static_request)
    soliton_request = SolitonRequest(base=static_request)
    soliton_result = _soliton_result(soliton_request, 9.0)
    _store_static(window, static_request, static_result)
    _store_soliton(window, soliton_request, soliton_result)
    window.update_run_button()

    window.use_last_static.setChecked(True)
    request = window.build_timedependent_request()

    np.testing.assert_array_equal(request.initial_theta, static_result.theta_final)
    assert request.initial_A is None
    assert not window.use_last_soliton.isChecked()
    window.close()


def test_source_enablement_mutual_exclusion_and_family_specific_invalidation():
    _app, window = _window()
    current = window.build_request()
    soliton_request = SolitonRequest(base=current)
    _store_soliton(window, soliton_request, _soliton_result(soliton_request, 2.0))

    incompatible = replace(
        current,
        grid=replace(current.grid, x_aperture_um=current.grid.x_aperture_um + 1.0),
    )
    _store_static(window, incompatible, run_static(incompatible))
    window.update_run_button()

    assert window.use_last_soliton.isEnabled()
    assert not window.use_last_static.isEnabled()
    assert "incompatible" in window.use_last_static.toolTip()

    _store_static(window, current, run_static(current))
    window.update_run_button()
    window.use_last_soliton.setChecked(True)
    window.use_last_static.setChecked(True)
    assert window.use_last_static.isChecked()
    assert not window.use_last_soliton.isChecked()
    window.close()


def test_missing_sources_disabled_and_existence_does_not_replace_single_soliton():
    _app, window = _window()
    assert not window.use_last_soliton.isEnabled()
    assert not window.use_last_static.isEnabled()

    base = window.build_request()
    soliton_request = SolitonRequest(base=base)
    soliton_result = _soliton_result(soliton_request, 4.0)
    _store_soliton(window, soliton_request, soliton_result)
    sweep_request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=(0.1,),
        base=soliton_request,
    )
    sweep_result = run_parameter_sweep(sweep_request)
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON_EXISTENCE,
        request=sweep_request,
        result=sweep_result,
    )

    assert window.retained_results.last_soliton_result.result is soliton_result
    window.close()


def test_single_soliton_populates_typed_selector_and_initializes_exact_state():
    _app, window = _window()
    request = SolitonRequest(base=window.build_request())
    result = _soliton_result(request, 6.0)
    _store_soliton(window, request, result)
    window.update_run_button()

    assert window.soliton_source_selector.count() == 1
    source = window.soliton_source_selector.itemData(0)
    assert isinstance(source, SolitonSource)
    assert source.source_kind.value == "single"
    assert window.soliton_source_selector.itemText(0).startswith(
        "Single soliton: 1 mW"
    )
    soliton_mode_index = window.td_initial_condition_selector.findData(
        TDInitialSourceMode.SOLITON_RESULT
    )
    assert (
        window.td_initial_condition_selector.model()
        .item(soliton_mode_index)
        .isEnabled()
    )

    window.td_initial_condition_selector.setCurrentIndex(soliton_mode_index)
    assert window.soliton_source_selector.isEnabled()
    td_request = window.build_timedependent_request()

    np.testing.assert_array_equal(td_request.initial_A, result.A)
    np.testing.assert_array_equal(td_request.initial_theta, result.theta)
    window.close()


def test_completed_single_soliton_run_enables_soliton_source_mode():
    app, window = _window()
    request = SolitonRequest(base=window.build_request())
    result = _soliton_result(request, 3.0)
    window.experiment_panel.experiment.setCurrentText("Soliton")
    window.build_soliton_request = lambda: request
    window._run_registered = lambda _operation, _request, **_kwargs: RunnerResult(
        "soliton", result, "Completed locally"
    )

    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)
    window.experiment_panel.experiment.setCurrentText(
        "Time-dependent propagation"
    )

    assert (
        window.retained_results.last_single_soliton_result.result is result
    )
    assert window.soliton_source_selector.count() == 1
    soliton_index = window.td_initial_condition_selector.findData(
        TDInitialSourceMode.SOLITON_RESULT
    )
    assert (
        window.td_initial_condition_selector.model()
        .item(soliton_index)
        .isEnabled()
    )
    window.close()


def test_stopped_single_soliton_requires_explicit_usability():
    _app, window = _window()
    request = SolitonRequest(base=window.build_request())
    unusable = replace(
        _soliton_result(request, 2.0, status="stopped"),
        usable_as_initial_condition=False,
    )
    _store_soliton(window, request, unusable)
    window.update_run_button()

    assert window.soliton_source_selector.count() == 0
    window.close()


def test_stopped_soliton_retains_latest_accepted_candidate():
    _app, window = _window()
    request = SolitonRequest(
        base=window.build_request(),
        max_outer=4,
        theta_steps_per_outer=1,
    )
    token = CancellationToken()

    def stop_after_first(progress):
        token.cancel()

    result = run_soliton(
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )

    assert result.status == "stopped"
    assert result.completed_iterations == 1
    assert len(result.history) == 1
    assert result.A is not None and result.theta is not None
    window.close()


def test_stopped_sweep_retains_only_completed_members():
    _app, window = _window()
    soliton_request = SolitonRequest(
        base=window.build_request(),
        max_outer=1,
        theta_steps_per_outer=1,
    )
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=(0.05, 0.06, 0.07),
        base=soliton_request,
        continuation=True,
    )
    token = CancellationToken()

    result = run_parameter_sweep(
        request,
        cancellation_token=token,
        progress_callback=lambda _progress: token.cancel(),
    )

    assert result.status == "stopped"
    assert 1 <= result.completed_points < len(request.values)
    assert len(result.results) == len(result.samples) == result.completed_points
    assert [member.status for member in result.members] == [
        "completed", "not_started", "not_started"
    ]
    window.close()


def test_parallel_sweep_stop_is_cooperative_and_keeps_completed_members_only():
    _app, window = _window()
    soliton_request = SolitonRequest(
        base=window.build_request(),
        max_outer=1,
        theta_steps_per_outer=1,
    )
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=(0.05, 0.06, 0.07),
        base=soliton_request,
        continuation=False,
        execution="parallel",
        max_workers=1,
    )
    token = CancellationToken()

    result = run_parameter_sweep(
        request,
        cancellation_token=token,
        progress_callback=lambda _progress: token.cancel(),
    )

    assert result.status == "stopped"
    assert 1 <= result.completed_points < len(request.values)
    assert len(result.results) == len(result.samples) == result.completed_points
    assert result.members[0].status == "completed"
    assert all(
        member.status in {"cancelled", "not_started"}
        for member in result.members[1:]
    )
    window.close()


def test_soliton_worker_stop_is_cooperative_and_close_joins_thread():
    app, window = _window()
    window.experiment_panel.experiment.setCurrentText("Soliton")
    request = SolitonRequest(base=window.build_request(), max_outer=2)
    result = _soliton_result(request, 5.0, status="stopped")
    worker_threads = []

    def cooperative_runner(_request, *, cancellation_token, progress_callback):
        worker_threads.append(QThread.currentThread())
        while not cancellation_token.is_cancelled():
            time.sleep(0.002)
        return RunnerResult("soliton", result, "Stopped locally")

    window._run_registered = lambda _operation, request, **kwargs: cooperative_runner(
        request, **kwargs
    )
    window.run_button.click()
    _wait_for(app, lambda: window._background_running)
    thread = window._td_thread
    window.close()

    assert worker_threads
    assert thread is not None and not thread.isRunning()
    assert window._td_thread is None


def test_six_sweep_members_produce_ordered_selectable_result_fields():
    _app, window = _window()
    powers = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    _request, result = _completed_sweep(window, powers)

    window.results_panel.set_run_data(to_run_data(result))
    selector = window.results_panel.workspace.image_pane.field_selector

    assert selector.count() == 6
    assert [selector.itemText(i) for i in range(selector.count())] == [
        f"Soliton at {power:g} mW" for power in powers
    ]
    for index, power in enumerate(powers):
        selector.setCurrentIndex(index)
        key = selector.currentData()
        field = window.results_panel.workspace.image_pane._run_data.fields[key]
        np.testing.assert_array_equal(
            field.data,
            np.full((16, 16), power * power),
        )
    window._active_request = _request
    window._on_timedependent_finished(
        RunnerResult("parameter_sweep", result, "Completed locally")
    )
    assert (
        window.results_panel.workspace.image_pane.td_time_label.text()
        == "Completed soliton existence sweep: 6/6"
    )
    window.close()


def test_gui_parallel_sweep_worker_count_is_configurable():
    _app, window = _window()
    window.experiment_panel.experiment.setCurrentText("Soliton existence curve")
    window.sweep_panel.continuation.setChecked(False)
    window.sweep_panel.execution.setCurrentIndex(
        window.sweep_panel.execution.findData("parallel")
    )
    window.sweep_panel.worker_count.setValue(6)

    request = window.build_soliton_existence_request()

    assert request.execution == "parallel"
    assert request.max_workers == 6
    window.close()


def test_gui_defaults_to_parallel_sweeps_when_runner_supports_them():
    _app, window = _window()
    window.experiment_panel.experiment.setCurrentText("Soliton existence curve")

    request = window.build_soliton_existence_request()

    assert window.runner.supports_parallel_sweeps is True
    assert window.sweep_panel.continuation.isChecked() is False
    assert request.continuation is False
    assert request.execution == "parallel"
    assert window.sweep_panel.worker_count.isEnabled()

    window.sweep_panel.continuation.setChecked(True)
    continued = window.build_soliton_existence_request()
    assert continued.continuation is True
    assert continued.execution == "sequential"
    window.close()


def test_sweep_panel_defaults_to_sequential_when_parallel_is_unavailable():
    app = QApplication.instance() or QApplication([])
    panel = SweepPanel(parallel_available=False)

    assert panel.continuation.isChecked()
    assert panel.sweep_execution() == "sequential"
    assert not panel.execution.model().item(
        panel.execution.findData("parallel")
    ).isEnabled()
    assert not panel.worker_count.isEnabled()
    panel.close()


def test_stopped_sweep_exposes_every_completed_member_and_no_unfinished_fields():
    _app, window = _window()
    powers = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    _request, result = _completed_sweep(window, powers)
    partial_members = list(result.members)
    for index in range(3, 6):
        partial_members[index] = replace(
            partial_members[index],
            status="cancelled",
            result=None,
            converged=None,
            completion_order=None,
        )
    partial = replace(
        result,
        status="stopped",
        completed_points=3,
        results=result.results[:3],
        samples=result.samples[:3],
        members=partial_members,
        metrics={
            **result.metrics,
            "completed_points": 3,
            "failed_count": 0,
            "status": "stopped",
        },
    )

    window.results_panel.set_run_data(to_run_data(partial))
    selector = window.results_panel.workspace.image_pane.field_selector

    assert selector.count() == 3
    assert [selector.itemText(i) for i in range(3)] == [
        "Soliton at 0.5 mW",
        "Soliton at 1 mW",
        "Soliton at 2 mW",
    ]
    window._active_request = _request
    window._on_timedependent_finished(
        RunnerResult("parameter_sweep", partial, "Stopped locally")
    )
    assert (
        window.results_panel.workspace.image_pane.td_time_label.text()
        == "Stopped soliton existence sweep: 3/6"
    )
    window.close()


def test_sweep_sources_populate_td_selector_and_selected_power_is_exact():
    _app, window = _window()
    powers = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    sweep_request, sweep_result = _completed_sweep(window, powers)
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON_EXISTENCE,
        request=sweep_request,
        result=sweep_result,
    )

    stack = window.beam_panel.beam_stack_definition
    window.beam_panel.set_beam_stack_definition(
        replace(
            stack,
            beams=(replace(stack.beams[0], power_mW=2.0),),
        )
    )
    static_request = window.build_request()
    _store_static(window, static_request, run_static(static_request))
    window.update_run_button()

    selector = window.soliton_source_selector
    assert selector.count() == 6
    assert [selector.itemText(i).split(",")[0] for i in range(6)] == [
        f"Sweep: {power:g} mW" for power in powers
    ]
    two_mw_index = next(
        index
        for index in range(selector.count())
        if selector.itemData(index).requested_power_mW == 2.0
    )
    selector.setCurrentIndex(two_mw_index)
    window.td_initial_condition_selector.setCurrentIndex(
        window.td_initial_condition_selector.findData(
            TDInitialSourceMode.SOLITON_RESULT
        )
    )
    request = window.build_timedependent_request()
    selected = selector.itemData(two_mw_index)

    np.testing.assert_array_equal(request.initial_A, selected.result.A)
    np.testing.assert_array_equal(request.initial_theta, selected.result.theta)
    assert not np.array_equal(
        request.initial_A, selector.itemData(0).result.A
    )
    assert not np.array_equal(
        request.initial_A,
        selector.itemData(selector.count() - 1).result.A,
    )
    assert not window.use_last_static.isChecked()
    assert "Source sweep power: 2 mW" in window.describe_request(request)

    window._start_timedependent_background(request=request)
    _wait_for(_app, lambda: not window._background_running)
    assert window.last_timedependent_result.provenance == {
        "initial_source_kind": "existence_member",
        "source_result_id": selected.result_id,
        "source_sweep_id": "test-sweep",
        "source_member_index": two_mw_index,
        "source_power_mW": 2.0,
        "source_beta": 20.0,
    }
    assert "TD soliton source: Sweep: 2 mW" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()


def test_static_runs_do_not_erase_sweep_choices_and_new_sweep_replaces_only_sweep():
    _app, window = _window()
    single_request = SolitonRequest(base=window.build_request())
    single_result = _soliton_result(single_request, 8.0)
    _store_soliton(window, single_request, single_result)
    first_request, first_result = _completed_sweep(window, (1.0, 2.0, 3.0))
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON_EXISTENCE,
        request=first_request,
        result=first_result,
    )
    window.update_run_button()
    assert window.soliton_source_selector.count() == 4

    _store_static(window, window.build_request(), run_static(window.build_request()))
    window.update_run_button()
    assert window.soliton_source_selector.count() == 4

    second_request, second_result = _completed_sweep(window, (4.0, 5.0))
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON_EXISTENCE,
        request=second_request,
        result=second_result,
    )
    window.update_run_button()

    assert window.retained_results.last_single_soliton_result.result is single_result
    assert window.soliton_source_selector.count() == 3
    assert [
        window.soliton_source_selector.itemText(index).split(",")[0]
        for index in range(3)
    ] == ["Single soliton: 1 mW", "Sweep: 4 mW", "Sweep: 5 mW"]
    window.close()


def test_incompatible_selected_sweep_member_disables_td_start_with_reason():
    _app, window = _window()
    sweep_request, sweep_result = _completed_sweep(window, (1.0,))
    window.retained_results.store(
        family=ExperimentFamily.SOLITON,
        workflow=WorkflowKind.SOLITON_EXISTENCE,
        request=sweep_request,
        result=sweep_result,
    )
    window.update_run_button()
    assert window.use_last_soliton.isEnabled()
    window.td_initial_condition_selector.setCurrentIndex(
        window.td_initial_condition_selector.findData(
            TDInitialSourceMode.SOLITON_RESULT
        )
    )

    window.grid_panel.x_aperture_um.setValue(
        window.grid_panel.x_aperture_um.value() + 1.0
    )

    assert not window.use_last_soliton.isEnabled()
    assert "incompatible with current grid" in window.use_last_soliton.toolTip()
    assert (
        window.td_initial_source_mode()
        is TDInitialSourceMode.SOLITON_RESULT
    )
    assert not window.run_button.isEnabled()
    window.close()


def test_gui_shutdown_during_parallel_sweep_joins_worker_and_process_pool():
    app, window = _window()
    window.experiment_panel.experiment.setCurrentText("Soliton existence curve")
    base = SolitonRequest(
        base=window.build_request(),
        max_outer=50,
        theta_steps_per_outer=1,
    )
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=(0.1, 0.2, 0.3, 0.4),
        base=base,
        continuation=False,
        execution="parallel",
        max_workers=2,
    )
    window.build_soliton_existence_request = lambda: request

    window.run_button.click()
    _wait_for(app, lambda: window._background_running)
    thread = window._td_thread
    window.close()

    assert thread is not None and not thread.isRunning()
    assert window._td_thread is None
