from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.execution import CancellationToken
from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_checkpoint,
    validate_pr_continuation,
)
from lcprop.pr.workflow import (
    continue_pr_timedependent,
    run_pr_timedependent,
)
from tests.test_pr_execution import _request


def _request_with_initial_state(*, steps: int = 4):
    request = _request(steps=steps)
    shape = (
        round(request.grid.z_length_um / request.grid.dz_um),
        request.grid.Nx,
        request.grid.Ny,
    )
    initial_E = np.linspace(0.0, 1e-4, np.prod(shape)).reshape(shape)
    return replace(request, initial_E=initial_E)


def _assert_same_cumulative_physics(actual, expected) -> None:
    np.testing.assert_array_equal(actual.A_initial, expected.A_initial)
    np.testing.assert_array_equal(actual.A_final, expected.A_final)
    np.testing.assert_array_equal(actual.E_initial, expected.E_initial)
    np.testing.assert_array_equal(actual.E_final, expected.E_final)
    np.testing.assert_array_equal(
        actual.source_intensity_stack,
        expected.source_intensity_stack,
    )
    assert actual.power_initial == expected.power_initial
    assert actual.power_final == expected.power_final
    assert actual.completed_steps == expected.completed_steps
    assert actual.time_normalized == expected.time_normalized


def _cancel_after(completed_step: int):
    token = CancellationToken()
    progress = []

    def observe(item) -> None:
        progress.append(item)
        if item.completed_units == completed_step:
            token.cancel()

    return token, progress, observe


def test_fresh_run_returns_valid_detached_checkpoint():
    result = run_pr_timedependent(_request_with_initial_state(steps=2))
    checkpoint = result.checkpoint

    assert isinstance(checkpoint, PRTimeDependentCheckpoint)
    validate_pr_checkpoint(checkpoint)
    assert checkpoint.status == "completed"
    assert checkpoint.completed_steps == 2
    assert checkpoint.requested_steps == 2
    assert checkpoint.time_normalized == pytest.approx(0.02)
    assert checkpoint.request.initial_A is None
    assert checkpoint.request.initial_E is None
    np.testing.assert_array_equal(checkpoint.A0, result.A_initial)
    np.testing.assert_array_equal(checkpoint.E_initial, result.E_initial)
    np.testing.assert_array_equal(checkpoint.E_current, result.E_final)
    assert not np.shares_memory(checkpoint.A0, result.A_initial)
    assert not np.shares_memory(checkpoint.E_initial, result.E_initial)
    assert not np.shares_memory(checkpoint.E_current, result.E_final)

    saved_value = checkpoint.E_current[0, 0, 0]
    result.E_final[0, 0, 0] = -999.0
    assert checkpoint.E_current[0, 0, 0] == saved_value


def test_cancel_resume_matches_uninterrupted_run_exactly():
    request = _request_with_initial_state(steps=4)
    token, progress, observe = _cancel_after(1)
    partial = run_pr_timedependent(
        request,
        cancellation_token=token,
        progress_callback=observe,
    )

    resumed = continue_pr_timedependent(
        request,
        partial.checkpoint,
        additional_steps=3,
    )
    uninterrupted = run_pr_timedependent(request)

    assert len(progress) == 1
    assert partial.status == "cancelled"
    assert partial.completed_steps == 1
    assert partial.checkpoint.status == "cancelled"
    assert resumed.status == "completed"
    assert resumed.completed_steps == 4
    assert resumed.requested_steps == 4
    assert resumed.time_normalized == pytest.approx(0.04)
    _assert_same_cumulative_physics(resumed, uninterrupted)
    np.testing.assert_array_equal(
        resumed.checkpoint.E_current,
        uninterrupted.checkpoint.E_current,
    )


def test_multiple_resume_segments_preserve_cumulative_progress():
    request = _request_with_initial_state(steps=4)
    first_token, _, first_observe = _cancel_after(1)
    first = run_pr_timedependent(
        request,
        cancellation_token=first_token,
        progress_callback=first_observe,
    )

    second_token, second_progress, second_observe = _cancel_after(2)
    second = continue_pr_timedependent(
        request,
        first.checkpoint,
        additional_steps=3,
        cancellation_token=second_token,
        progress_callback=second_observe,
    )
    final = continue_pr_timedependent(
        request,
        second.checkpoint,
        additional_steps=2,
    )
    uninterrupted = run_pr_timedependent(request)

    assert second.status == "cancelled"
    assert second.completed_steps == 2
    assert second.requested_steps == 4
    assert len(second_progress) == 1
    progress = second_progress[0]
    assert progress.completed_units == 2
    assert progress.total_units == 4
    assert progress.prior_completed_steps == 1
    assert progress.segment_completed_steps == 1
    assert progress.segment_total_steps == 3
    assert progress.cumulative_completed_steps == 2
    assert progress.segment_start_time == pytest.approx(0.01)
    assert progress.segment_elapsed_time == pytest.approx(0.01)
    assert progress.cumulative_time == pytest.approx(0.02)
    assert progress.checkpoint_available is True

    assert final.completed_steps == 4
    assert final.requested_steps == 4
    _assert_same_cumulative_physics(final, uninterrupted)


def test_pre_cancelled_checkpoint_is_valid_and_resumable():
    request = _request_with_initial_state(steps=3)
    token = CancellationToken()
    token.cancel()

    partial = run_pr_timedependent(request, cancellation_token=token)
    resumed = continue_pr_timedependent(
        request,
        partial.checkpoint,
        additional_steps=3,
    )
    uninterrupted = run_pr_timedependent(request)

    assert partial.status == "cancelled"
    assert partial.completed_steps == 0
    assert partial.checkpoint.completed_steps == 0
    assert partial.checkpoint.time_normalized == 0.0
    validate_pr_checkpoint(partial.checkpoint)
    _assert_same_cumulative_physics(resumed, uninterrupted)


def test_zero_step_continuation_preserves_accepted_state():
    request = _request_with_initial_state(steps=2)
    initial = run_pr_timedependent(request)

    continued = continue_pr_timedependent(
        request,
        initial.checkpoint,
        additional_steps=0,
    )

    assert continued.status == "completed"
    assert continued.completed_steps == 2
    assert continued.requested_steps == 2
    _assert_same_cumulative_physics(continued, initial)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (
            lambda request: replace(
                request,
                grid=replace(request.grid, x_aperture_um=81.0),
            ),
            "incompatible grid",
        ),
        (
            lambda request: replace(
                request,
                material=replace(request.material, applied_field=0.6),
            ),
            "incompatible material",
        ),
        (
            lambda request: replace(
                request,
                beams=replace(
                    request.beams,
                    channels=(
                        replace(request.beams.channels[0], power_mW=2.0),
                    ),
                ),
            ),
            "incompatible beams",
        ),
        (
            lambda request: replace(
                request,
                backend=replace(request.backend, precision="float32"),
            ),
            "incompatible backend",
        ),
        (
            lambda request: replace(
                request,
                solver=replace(request.solver, dt_normalized=0.005),
            ),
            "incompatible solver",
        ),
        (
            lambda request: replace(
                request,
                solver=replace(request.solver, optical_substeps=2),
            ),
            "incompatible solver",
        ),
    ),
)
def test_continuation_rejects_changed_configuration(change, message):
    request = _request_with_initial_state(steps=2)
    checkpoint = run_pr_timedependent(request).checkpoint

    with pytest.raises(ValueError, match=message):
        validate_pr_continuation(change(request), checkpoint)


def test_continuation_rejects_changed_initial_fields():
    request = _request_with_initial_state(steps=2)
    checkpoint = run_pr_timedependent(request).checkpoint
    changed_A = np.asarray(request.initial_A).copy()
    changed_A[0, 0, 0] += 1.0
    changed_E = np.asarray(request.initial_E).copy()
    changed_E[0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="incompatible initial_A"):
        validate_pr_continuation(
            replace(request, initial_A=changed_A),
            checkpoint,
        )
    with pytest.raises(ValueError, match="incompatible initial_E"):
        validate_pr_continuation(
            replace(request, initial_E=changed_E),
            checkpoint,
        )


@pytest.mark.parametrize(
    "change",
    (
        lambda checkpoint: replace(checkpoint, status="invalid"),
        lambda checkpoint: replace(checkpoint, requested_steps=0),
        lambda checkpoint: replace(checkpoint, time_normalized=9.0),
        lambda checkpoint: replace(
            checkpoint,
            E_current=np.asarray(checkpoint.E_current)[:-1],
        ),
        lambda checkpoint: replace(checkpoint, E_dtype="float32"),
    ),
)
def test_malformed_checkpoint_is_rejected(change):
    checkpoint = run_pr_timedependent(
        _request_with_initial_state(steps=2)
    ).checkpoint

    with pytest.raises(ValueError):
        validate_pr_checkpoint(change(checkpoint))


@pytest.mark.parametrize("additional_steps", (-1, 1.5, True))
def test_invalid_additional_steps_are_rejected(additional_steps):
    request = _request_with_initial_state(steps=1)
    checkpoint = run_pr_timedependent(request).checkpoint

    with pytest.raises(ValueError, match="additional_steps"):
        continue_pr_timedependent(request, checkpoint, additional_steps)
