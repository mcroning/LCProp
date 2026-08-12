from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.persistence import load_run_checkpoint, save_run_checkpoint
from lcprop.products.data_model import from_static_result
from lcprop.workflows.static import continue_static, run_static
from lcprop.workflows.timedependent import (
    run_timedependent,
    timedependent_state_from_static_result,
)


def _workflow():
    return StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )


def _static_request(*, slices=3):
    return StaticRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=10,
            dz_um=5.0,
            x_aperture_um=40.0,
            y_aperture_um=50.0,
            z_length_um=5.0 * slices,
        ),
        material=LCMaterial(),
        bias=BiasSpec(),
        beams=BeamStack(channels=(BeamChannel(
            power_mW=0.1,
            waist_x_um=4.0,
            waist_y_um=4.0,
            x0_um=-3.0,
            coherence_group="laser-A",
        ),)),
        solver=StaticSolverOptions(
            workflow=_workflow(),
            static_max_coupled_passes=1,
            static_max_relax_iterations=1,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        ),
        output=OutputOptions(),
    )


def _td_request(static_request, *, steps=1):
    return TimeDependentRunRequest(
        grid=static_request.grid,
        material=static_request.material,
        bias=static_request.bias,
        beams=static_request.beams,
        solver=TimeDependentSolverOptions(
            workflow=_workflow(), Nt=steps, dt=7.5e-4, max_picard_iter=1
        ),
        output=OutputOptions(),
        runtime=static_request.runtime,
    )


def _stop_static_after(request, completed):
    token = CancellationToken()
    progress = []

    def observe(item):
        progress.append(item)
        if item.completed_units == completed:
            token.cancel()

    return run_static(
        request, cancellation_token=token, progress_callback=observe
    ), progress


def test_static_and_td_share_run_progress_envelope():
    static_result, static_progress = _stop_static_after(_static_request(), 1)
    td_progress = []
    run_timedependent(_td_request(_static_request()), progress_callback=td_progress.append)

    assert isinstance(static_progress[0], RunProgress)
    assert isinstance(td_progress[0], RunProgress)
    assert static_progress[0].workflow == "static"
    assert static_progress[0].coordinate_name == "z"
    assert static_progress[0].coordinate_unit == "um"
    assert td_progress[0].workflow == "timedependent"
    assert td_progress[0].coordinate_name == "t"
    assert static_result.status == "stopped"


def test_static_stop_continue_preserves_exact_prefix_and_matches_direct():
    request = _static_request(slices=3)
    stopped, progress = _stop_static_after(request, 1)

    assert stopped.completed_slices == 1
    assert stopped.theta_final.shape[0] == 1
    assert stopped.intensity_stack.shape[0] == 1
    assert stopped.z_reached_um == pytest.approx(5.0)
    assert progress[0].current_coordinate == pytest.approx(5.0)

    resumed = continue_static(request, stopped.checkpoint)
    direct = run_static(request)
    assert resumed.completed_slices == 3
    assert [item.z_index for item in resumed.slice_summaries] == [0, 1, 2]
    np.testing.assert_allclose(resumed.A_final, direct.A_final, rtol=0, atol=0)
    np.testing.assert_allclose(resumed.theta_final, direct.theta_final, rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.intensity_stack, direct.intensity_stack, rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        resumed.theta_final[0], stopped.theta_final[0]
    )


def test_static_pre_cancelled_state_has_zero_valid_slices_and_can_continue():
    request = _static_request(slices=2)
    token = CancellationToken()
    token.cancel()
    stopped = run_static(request, cancellation_token=token)

    assert stopped.completed_slices == 0
    assert stopped.theta_final.shape == (0, 8, 10)
    assert list(from_static_result(stopped).fields.keys()) == ["input_intensity"]
    resumed = continue_static(request, stopped.checkpoint)
    direct = run_static(request)
    np.testing.assert_allclose(resumed.A_final, direct.A_final, rtol=0, atol=0)


def test_static_multi_stop_appends_without_repeating_slices():
    request = _static_request(slices=4)
    first, _ = _stop_static_after(request, 1)
    token = CancellationToken()

    def stop_at_three(item):
        if item.completed_units == 3:
            token.cancel()

    second = continue_static(
        request,
        first.checkpoint,
        cancellation_token=token,
        progress_callback=stop_at_three,
    )
    final = continue_static(request, second.checkpoint)

    assert (first.completed_slices, second.completed_slices, final.completed_slices) == (1, 3, 4)
    assert [item.z_index for item in final.slice_summaries] == [0, 1, 2, 3]


def test_fixed_theta_static_uses_same_slice_boundary_stop_and_continue():
    request = _static_request(slices=3)
    request = replace(
        request,
        solver=replace(
            request.solver,
            workflow=StaticWorkflowOptions(
                strategy="fixed_theta",
                theta_solver="none",
                optics_solver="splitstep",
                coupling="frozen",
            ),
        ),
    )
    stopped, _ = _stop_static_after(request, 1)
    resumed = continue_static(request, stopped.checkpoint)
    direct = run_static(request)

    assert stopped.status == "stopped"
    assert stopped.theta_final.shape[0] == 1
    np.testing.assert_allclose(resumed.A_final, direct.A_final, rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.intensity_stack, direct.intensity_stack, rtol=0, atol=0
    )


def test_static_checkpoint_round_trip_and_shared_dispatch(tmp_path):
    request = _static_request()
    stopped, _ = _stop_static_after(request, 2)
    save_run_checkpoint(stopped.checkpoint, tmp_path)
    loaded = load_run_checkpoint(tmp_path)

    np.testing.assert_array_equal(loaded.A_next, stopped.checkpoint.A_next)
    np.testing.assert_array_equal(loaded.theta_stack, stopped.theta_final)
    assert loaded.completed_slices == 2
    assert loaded.next_slice_index == 2
    assert len(loaded.slice_summaries) == 2
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["workflow"] == "static"
    assert provenance["completed_units"] == 2
    assert not any(
        str(tmp_path) in path.read_text(errors="ignore")
        for path in (tmp_path / "request.json", tmp_path / "provenance.json")
    )

    resumed = continue_static(request, loaded)
    direct = run_static(request)
    np.testing.assert_allclose(resumed.theta_final, direct.theta_final)


def test_shared_checkpoint_dispatch_preserves_timedependent_workflow(tmp_path):
    request = _static_request(slices=2)
    td_result = run_timedependent(_td_request(request))
    save_run_checkpoint(td_result.checkpoint, tmp_path)
    loaded = load_run_checkpoint(tmp_path)

    assert loaded.completed_steps == td_result.completed_steps
    np.testing.assert_array_equal(loaded.theta, td_result.checkpoint.theta)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["workflow"] == "timedependent"


def test_stopped_static_fields_default_data_are_latest_and_input_is_retained():
    stopped, _ = _stop_static_after(_static_request(), 2)
    run_data = from_static_result(stopped)

    assert run_data.fields["input_intensity"].display_name == "Input Plane Intensity"
    assert run_data.fields["final_intensity"].display_name == "Intensity at current z"
    assert run_data.fields["output_delta_theta"].display_name == "Δθ at current z"
    assert run_data.geometry.z.shape == (2,)


def test_timedependent_initialization_from_completed_static_is_exact_and_finite():
    static_request = _static_request(slices=2)
    static_result = run_static(static_request)
    theta_before = np.asarray(static_result.theta_final).copy()
    td_request = _td_request(static_request)

    initialized = timedependent_state_from_static_result(static_result, td_request)
    np.testing.assert_array_equal(initialized.initial_theta, theta_before)
    assert initialized.initial_A is None
    td_result = run_timedependent(initialized)
    np.testing.assert_array_equal(td_result.theta_initial, theta_before)
    assert np.isfinite(np.asarray(td_result.theta_final)).all()
    assert np.isfinite(np.asarray(td_result.A_final)).all()
    np.testing.assert_array_equal(static_result.theta_final, theta_before)


def test_timedependent_from_completed_fixed_static_uses_recorded_theta_volume():
    static_request = _static_request(slices=2)
    static_request = replace(
        static_request,
        solver=replace(
            static_request.solver,
            workflow=StaticWorkflowOptions(
                strategy="fixed_theta",
                theta_solver="none",
                optics_solver="splitstep",
                coupling="frozen",
            ),
        ),
    )
    static_result = run_static(static_request)
    initialized = timedependent_state_from_static_result(
        static_result, _td_request(static_request)
    )

    assert initialized.initial_theta.shape == (2, 8, 10)
    np.testing.assert_array_equal(
        initialized.initial_theta, static_result.checkpoint.theta_stack
    )


def test_timedependent_from_static_rejects_partial_and_names_mismatch():
    static_request = _static_request(slices=2)
    partial, _ = _stop_static_after(static_request, 1)
    td_request = _td_request(static_request)
    with pytest.raises(ValueError, match="fully completed static result"):
        timedependent_state_from_static_result(partial, td_request)

    completed = run_static(static_request)
    changed = replace(
        td_request,
        grid=replace(td_request.grid, x_aperture_um=41.0),
    )
    with pytest.raises(ValueError, match=r"grid\.x_aperture_um"):
        timedependent_state_from_static_result(completed, changed)
