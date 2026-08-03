from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.execution import CancellationToken
from lcprop.core.requests import (
    OutputOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.persistence import (
    load_timedependent_checkpoint,
    save_timedependent_checkpoint,
)
from lcprop.runners.local import LocalRunner
from lcprop.workflows.timedependent import (
    continue_timedependent,
    run_timedependent,
)


def _request(*, steps: int) -> TimeDependentRunRequest:
    return TimeDependentRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=10,
            dz_um=5.0,
            x_aperture_um=40.0,
            y_aperture_um=50.0,
            z_length_um=10.0,
        ),
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    power_mW=0.1,
                    waist_x_um=4.0,
                    waist_y_um=4.0,
                    x0_um=-3.0,
                    tilt_x_rad_per_um=0.037,
                    tilt_y_rad_per_um=-0.021,
                ),
            )
        ),
        solver=TimeDependentSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            Nt=steps,
            dt=7.5e-4,
            max_picard_iter=2,
        ),
        output=OutputOptions(),
    )


def test_cancellation_is_observed_at_completed_step_boundary():
    token = CancellationToken()
    progress = []

    def after_step(item):
        progress.append(item)
        if item.completed_step == 1:
            token.cancel()

    result = run_timedependent(
        _request(steps=4),
        cancellation_token=token,
        progress_callback=after_step,
    )

    assert result.status == "cancelled"
    assert result.Nt == 1
    assert result.completed_steps == 1
    assert result.requested_steps == 4
    assert result.current_time == pytest.approx(7.5e-4)
    assert result.segment_start_time == 0.0
    assert result.segment_elapsed_time == pytest.approx(7.5e-4)
    assert result.cumulative_time == pytest.approx(7.5e-4)
    assert result.prior_completed_steps == 0
    assert result.segment_completed_steps == 1
    assert result.cumulative_completed_steps == 1
    assert [item.completed_step for item in progress] == [1]
    assert progress[0].total_steps == 4
    assert progress[0].current_time == pytest.approx(7.5e-4)
    assert progress[0].segment_elapsed_time == pytest.approx(7.5e-4)
    assert progress[0].cumulative_time == pytest.approx(7.5e-4)
    assert progress[0].elapsed_wall_time >= 0.0
    assert result.checkpoint.status == "cancelled"
    assert np.isfinite(np.asarray(result.checkpoint.theta)).all()
    assert np.isfinite(np.asarray(result.checkpoint.A0)).all()

    resumed = continue_timedependent(
        _request(steps=4),
        result.checkpoint,
        additional_steps=3,
    )
    uninterrupted = run_timedependent(_request(steps=4))
    assert resumed.segment_start_time == result.checkpoint.current_time
    assert resumed.segment_elapsed_time == pytest.approx(3 * 7.5e-4)
    assert resumed.cumulative_time == pytest.approx(4 * 7.5e-4)
    assert resumed.prior_completed_steps == 1
    assert resumed.segment_completed_steps == 3
    assert resumed.cumulative_completed_steps == 4
    np.testing.assert_allclose(resumed.theta_final, uninterrupted.theta_final)
    np.testing.assert_allclose(resumed.A_final, uninterrupted.A_final)


def test_checkpoint_save_load_preserves_minimal_state(tmp_path):
    result = run_timedependent(_request(steps=2))

    save_timedependent_checkpoint(result.checkpoint, tmp_path)
    loaded = load_timedependent_checkpoint(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "request.json",
        "checkpoint.npz",
        "provenance.json",
    }
    assert loaded.schema_version == 1
    assert loaded.request == result.checkpoint.request
    loaded_channel = loaded.request.beams.channels[0]
    assert loaded_channel.tilt_x_rad_per_um == 0.037
    assert loaded_channel.tilt_y_rad_per_um == -0.021
    assert loaded.completed_steps == 2
    assert loaded.requested_steps == 2
    assert loaded.current_time == result.checkpoint.current_time
    assert loaded.current_time == pytest.approx(2 * 7.5e-4)
    assert loaded.status == "completed"
    assert loaded.theta_dtype == result.checkpoint.theta_dtype
    assert loaded.A0_dtype == result.checkpoint.A0_dtype
    np.testing.assert_array_equal(loaded.theta, result.checkpoint.theta)
    np.testing.assert_array_equal(loaded.A0, result.checkpoint.A0)


def test_continuation_matches_uninterrupted_run(tmp_path):
    first_request = _request(steps=2)
    first = run_timedependent(first_request)
    assert first.segment_start_time == 0.0
    assert first.segment_elapsed_time == pytest.approx(2 * first_request.solver.dt)
    assert first.cumulative_time == pytest.approx(2 * first_request.solver.dt)
    save_timedependent_checkpoint(first.checkpoint, tmp_path)
    loaded = load_timedependent_checkpoint(tmp_path)
    progress = []
    resumed = continue_timedependent(
        first_request,
        loaded,
        3,
        progress_callback=progress.append,
    )
    uninterrupted = run_timedependent(_request(steps=5))

    assert resumed.status == "completed"
    assert resumed.completed_steps == 5
    assert resumed.requested_steps == 5
    assert resumed.current_time == pytest.approx(5 * first_request.solver.dt)
    assert resumed.segment_start_time == first.checkpoint.current_time
    assert resumed.segment_elapsed_time == pytest.approx(3 * first_request.solver.dt)
    assert resumed.cumulative_time == pytest.approx(5 * first_request.solver.dt)
    assert resumed.prior_completed_steps == 2
    assert resumed.segment_completed_steps == 3
    assert resumed.cumulative_completed_steps == 5
    assert progress[0].segment_completed_steps == 1
    assert progress[0].cumulative_completed_steps == 3
    assert progress[0].segment_start_time == loaded.current_time
    assert progress[0].cumulative_time == pytest.approx(
        loaded.current_time + first_request.solver.dt
    )
    np.testing.assert_allclose(resumed.theta_final, uninterrupted.theta_final)
    np.testing.assert_allclose(resumed.A_final, uninterrupted.A_final)


def test_in_memory_continuation_appends_cumulative_width_history():
    request = _request(steps=2)
    first = run_timedependent(request)
    resumed = continue_timedependent(request, first.checkpoint, 3)

    assert first.width_times == pytest.approx(
        [0.0, request.solver.dt, 2 * request.solver.dt]
    )
    assert resumed.width_times[: len(first.width_times)] == pytest.approx(
        first.width_times
    )
    assert resumed.width_times == pytest.approx(
        [request.solver.dt * step for step in range(6)]
    )
    assert len(resumed.beam_x_rms_width_um) == 6
    assert len(resumed.beam_y_rms_width_um) == 6


def test_continuation_rejects_incompatible_grid():
    request = _request(steps=1)
    first = run_timedependent(request)
    incompatible = replace(
        request,
        grid=replace(request.grid, x_aperture_um=41.0),
    )

    with pytest.raises(ValueError, match="incompatible grid"):
        continue_timedependent(incompatible, first.checkpoint, 1)


def test_multiple_stop_continue_segments_preserve_exact_state_and_counts():
    request = _request(steps=8)

    def cancel_after_one(run, *args):
        token = CancellationToken()

        def stop(progress):
            token.cancel()

        return run(
            *args,
            cancellation_token=token,
            progress_callback=stop,
        )

    first = cancel_after_one(run_timedependent, request)
    second = cancel_after_one(
        continue_timedependent,
        request,
        first.checkpoint,
        8,
    )
    third = cancel_after_one(
        continue_timedependent,
        request,
        second.checkpoint,
        8,
    )
    fourth = continue_timedependent(request, third.checkpoint, 1)
    uninterrupted = run_timedependent(_request(steps=4))

    segments = (first, second, third, fourth)
    assert [result.completed_steps for result in segments] == [1, 2, 3, 4]
    assert [result.current_time for result in segments] == pytest.approx(
        [request.solver.dt * step for step in (1, 2, 3, 4)]
    )
    np.testing.assert_array_equal(second.theta_initial, first.theta_final)
    np.testing.assert_array_equal(third.theta_initial, second.theta_final)
    np.testing.assert_array_equal(fourth.theta_initial, third.theta_final)
    np.testing.assert_array_equal(
        second.initial_intensity_stack,
        first.final_intensity_stack,
    )
    np.testing.assert_array_equal(
        third.initial_intensity_stack,
        second.final_intensity_stack,
    )
    np.testing.assert_array_equal(second.A_initial, first.checkpoint.A0)
    np.testing.assert_array_equal(third.A_initial, second.checkpoint.A0)
    assert np.max(
        np.abs(
            np.asarray(second.theta_initial)
            - np.asarray(second.theta_bias)[None, :, :]
        )
    ) > 0.0
    np.testing.assert_allclose(fourth.theta_final, uninterrupted.theta_final)
    np.testing.assert_allclose(fourth.A_final, uninterrupted.A_final)


def test_local_runner_synchronous_api_remains_usable():
    runner_result = LocalRunner().run_timedependent(_request(steps=1))

    assert runner_result.kind == "timedependent"
    assert runner_result.result.status == "completed"
    assert runner_result.message == "Completed locally"
