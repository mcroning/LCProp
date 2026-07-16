from __future__ import annotations

from dataclasses import replace
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from lcprop.core.execution import CancellationToken
from lcprop.optics.splitstep import total_intensity
from lcprop.products.data_model import (
    from_timedependent_live_state,
    from_timedependent_result,
)
from lcprop.gui.views.image_pane import ImagePane
from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from lcprop.gui.main_window import LCPropMainWindow
from lcprop.workflows.static import run_static
from lcprop.workflows.timedependent import (
    continue_timedependent,
    run_timedependent,
    timedependent_state_from_static_result,
)
from tests.test_unified_execution import _static_request, _td_request


def _reference_from_result(result):
    return {
        "intensity_stack": np.asarray(result.initial_intensity_stack).copy(),
        "theta_stack": np.asarray(result.theta_initial).copy(),
        "theta_bias": np.asarray(result.theta_bias).copy(),
        "output_plane_intensity": np.asarray(
            result.initial_output_plane_intensity
        ).copy(),
    }


def _td_from_static(*, steps=2):
    static_request = _static_request(slices=2)
    static_result = run_static(static_request)
    request = timedependent_state_from_static_result(
        static_result,
        _td_request(static_request, steps=steps),
    )
    return static_result, request


def test_td_from_static_live_output_and_longitudinal_are_latest_z_snapshot():
    _, request = _td_from_static(steps=2)
    progress = []
    result = run_timedependent(request, progress_callback=progress.append)
    reference = {
        "intensity_stack": progress[0].latest_field_state[
            "initial_intensity_stack"
        ],
        "theta_stack": progress[0].latest_field_state["theta_initial"],
        "theta_bias": progress[0].latest_field_state["theta_bias"],
        "output_plane_intensity": progress[0].latest_field_state[
            "initial_output_plane_intensity"
        ],
    }
    original_reference = {
        key: np.asarray(value).copy() for key, value in reference.items()
    }

    assert len(progress) == 2
    assert not np.array_equal(
        progress[0].latest_field_state["output_plane_intensity"],
        progress[1].latest_field_state["output_plane_intensity"],
    )
    for item in progress:
        state = item.latest_field_state
        live = from_timedependent_live_state(
            state, td_from_static=True, static_reference=reference
        )
        np.testing.assert_array_equal(
            live.fields["final_intensity"].data,
            state["output_plane_intensity"],
        )
        assert live.fields["final_intensity"].axes == ("x", "y")
        assert live.fields["final_intensity"].display_name == (
            "Current Output Plane Intensity"
        )
        assert live.longitudinal_enabled
        np.testing.assert_array_equal(
            live.fields["final_intensity_stack"].data,
            state["current_intensity_stack"],
        )
        np.testing.assert_array_equal(
            live.fields["final_delta_theta_stack"].data,
            np.asarray(state["theta_current"])
            - np.asarray(state["theta_bias"])[None, :, :],
        )
        assert live.fields["final_intensity_stack"].axes == ("z", "x", "y")
        assert live.fields["final_delta_theta_stack"].axes == ("z", "x", "y")
        assert not any("t" in field.axes for field in live.fields.values())
        for key, expected in original_reference.items():
            np.testing.assert_array_equal(reference[key], expected)

    expected_output = total_intensity(
        result.A_final,
        coherence_groups=tuple(result.launch_summary["coherence_groups"]),
    )
    np.testing.assert_allclose(
        progress[-1].latest_field_state["output_plane_intensity"],
        expected_output,
        rtol=1e-13,
        atol=1e-15,
    )


def test_live_output_reconstruction_does_not_change_td_or_checkpoint_state():
    _, request = _td_from_static(steps=2)
    progress = []
    observed = run_timedependent(request, progress_callback=progress.append)
    unobserved = run_timedependent(request)

    np.testing.assert_array_equal(observed.theta_final, unobserved.theta_final)
    np.testing.assert_array_equal(observed.A_final, unobserved.A_final)
    np.testing.assert_array_equal(
        observed.checkpoint.theta, unobserved.checkpoint.theta
    )
    np.testing.assert_array_equal(
        observed.checkpoint.A0, unobserved.checkpoint.A0
    )
    assert observed.checkpoint.current_time == unobserved.checkpoint.current_time
    assert observed.checkpoint.completed_steps == unobserved.checkpoint.completed_steps


def test_td_from_static_stop_and_continuation_use_same_output_plane_state():
    _, request = _td_from_static(steps=3)
    token = CancellationToken()

    def stop_after_first(item):
        if item.completed_units == 1:
            token.cancel()

    stopped = run_timedependent(
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )
    reference = _reference_from_result(stopped)
    stopped_data = from_timedependent_result(
        stopped, td_from_static=True, static_reference=reference
    )
    checkpoint_output = total_intensity(
        stopped.A_final,
        coherence_groups=tuple(stopped.launch_summary["coherence_groups"]),
    )

    assert stopped.status == "cancelled"
    assert stopped_data.fields["final_intensity"].display_name == (
        "Output Plane Intensity at Stop"
    )
    np.testing.assert_array_equal(
        stopped_data.fields["final_intensity"].data, checkpoint_output
    )

    continued = continue_timedependent(
        request, stopped.checkpoint, additional_steps=1
    )
    np.testing.assert_allclose(
        continued.initial_output_plane_intensity,
        checkpoint_output,
        rtol=1e-13,
        atol=1e-15,
    )
    completed_data = from_timedependent_result(
        continued, td_from_static=True, static_reference=reference
    )
    assert completed_data.fields["final_intensity"].display_name == (
        "Final Output Plane Intensity"
    )
    for key in (
        "initial_static_intensity_stack",
        "initial_static_delta_theta_stack",
    ):
        np.testing.assert_array_equal(
            stopped_data.fields[key].data,
            completed_data.fields[key].data,
        )
        assert stopped_data.fields[key].axes == ("z", "x", "y")
        assert stopped_data.fields[key].display_name.startswith("Initial Static")


def test_td_from_static_panes_default_to_current_output_and_z_volume():
    app = QApplication.instance() or QApplication([])
    _, request = _td_from_static(steps=1)
    progress = []
    result = run_timedependent(request, progress_callback=progress.append)
    state = progress[0].latest_field_state
    live = from_timedependent_live_state(state, td_from_static=True)
    image = ImagePane()
    longitudinal = LongitudinalPane()

    image.set_run_data(live)
    longitudinal.set_run_data(live)
    assert image.field_selector.currentData() == "final_intensity"
    assert image.field_selector.currentText() == "Current Output Plane Intensity"
    assert not longitudinal.controls_widget.isHidden()
    assert longitudinal.no_data_label.isHidden()
    assert longitudinal.field_selector.currentData() == "final_intensity_stack"
    assert longitudinal.field_selector.currentText() == "Intensity at current t"

    reference = _reference_from_result(result)
    completed = from_timedependent_result(
        result, td_from_static=True, static_reference=reference
    )
    image.set_run_data(completed)
    longitudinal.set_run_data(completed)
    assert image.field_selector.currentData() == "final_intensity"
    assert image.field_selector.currentText() == "Final Output Plane Intensity"
    assert [
        longitudinal.field_selector.itemText(index)
        for index in range(longitudinal.field_selector.count())
    ] == [
        "Initial Static Intensity",
        "Initial Static Delta Theta",
        "Final Intensity",
        "Final Delta Theta",
    ]
    assert all(
        completed.fields[
            longitudinal.field_selector.itemData(index)
        ].axes == ("z", "x", "y")
        for index in range(longitudinal.field_selector.count())
    )
    app.processEvents()


def test_normal_td_live_and_completed_fields_remain_z_snapshots():
    static_request = _static_request(slices=2)
    request = _td_request(static_request, steps=1)
    progress = []
    result = run_timedependent(request, progress_callback=progress.append)
    live = from_timedependent_live_state(progress[0].latest_field_state)
    completed = from_timedependent_result(result)

    assert live.longitudinal_enabled
    assert live.fields["final_intensity"].display_name == "Intensity at current t"
    assert live.fields["final_delta_theta_stack"].axes == ("z", "x", "y")
    assert live.fields["final_delta_theta_stack"].display_name == (
        "Delta Theta at current t"
    )
    assert completed.fields["final_intensity"].display_name == "Final Intensity"
    assert completed.fields["final_intensity_stack"].axes == ("z", "x", "y")
    assert completed.fields["final_intensity_stack"].display_name == "Final Intensity"
    assert not any("t" in field.axes for field in completed.fields.values())


def test_main_window_td_from_static_live_stop_and_reference_fields():
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.grid_panel.Nx.setValue(16)
    window.grid_panel.Ny.setValue(16)
    window.grid_panel.z_length_um.setValue(10.0)
    window.grid_panel.dz_um.setValue(5.0)
    window.solver_panel.max_iterations.setValue(1)
    static_request = window.build_request()
    static_request = replace(
        static_request,
        solver=replace(
            static_request.solver,
            static_max_relax_iterations=1,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        ),
    )
    window.last_static_result = run_static(static_request)
    window.experiment_panel.experiment.setCurrentText(
        "Time-dependent propagation"
    )
    window.solver_panel.Nt.setValue(5)
    window.use_last_static.setChecked(True)

    original = window.runner.run_timedependent

    def slow_runner(request, **kwargs):
        gui_progress = kwargs["progress_callback"]

        def slow_progress(progress):
            gui_progress(progress)
            time.sleep(0.03)

        return original(
            request, **{**kwargs, "progress_callback": slow_progress}
        )

    window.runner.run_timedependent = slow_runner
    window.run_button.click()
    console = window.results_panel.workspace.console
    deadline = time.monotonic() + 10.0
    while "segment step 1/5" not in console.toPlainText():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for TD-from-static progress")
        time.sleep(0.002)

    image = window.results_panel.workspace.image_pane
    longitudinal = window.results_panel.workspace.longitudinal_pane
    assert image.field_selector.currentData() == "final_intensity"
    assert image.field_selector.currentText() == "Current Output Plane Intensity"
    assert not longitudinal.controls_widget.isHidden()
    assert longitudinal.field_selector.currentData() == "final_intensity_stack"
    assert longitudinal.field_selector.currentText() == "Intensity at current t"
    live_static_intensity = np.asarray(
        image._run_data.fields["initial_static_intensity_stack"].data
    ).copy()
    live_static_delta = np.asarray(
        image._run_data.fields["initial_static_delta_theta_stack"].data
    ).copy()

    window.stop_button.click()
    deadline = time.monotonic() + 10.0
    while window._background_running:
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out stopping TD-from-static run")
        time.sleep(0.002)
    app.processEvents()

    result = window.last_timedependent_result
    expected = total_intensity(
        result.A_final,
        coherence_groups=tuple(result.launch_summary["coherence_groups"]),
    )
    run_data = image._run_data
    assert image.field_selector.currentData() == "final_intensity"
    assert image.field_selector.currentText() == "Output Plane Intensity at Stop"
    np.testing.assert_array_equal(
        run_data.fields["final_intensity"].data, expected
    )
    np.testing.assert_array_equal(
        run_data.fields["final_intensity_stack"].data,
        result.final_intensity_stack,
    )
    np.testing.assert_array_equal(
        run_data.fields["final_delta_theta_stack"].data,
        np.asarray(result.checkpoint.theta)
        - np.asarray(result.theta_bias)[None, :, :],
    )
    np.testing.assert_array_equal(
        run_data.fields["initial_static_intensity_stack"].data,
        live_static_intensity,
    )
    np.testing.assert_array_equal(
        run_data.fields["initial_static_delta_theta_stack"].data,
        live_static_delta,
    )
    assert [
        longitudinal.field_selector.itemText(index)
        for index in range(longitudinal.field_selector.count())
    ] == [
        "Initial Static Intensity",
        "Initial Static Delta Theta",
        "Intensity at Stop",
        "Delta Theta at Stop",
    ]
