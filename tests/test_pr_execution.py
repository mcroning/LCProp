from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.workflow import run_pr_timedependent


def _request(*, steps: int = 3) -> PRRunRequest:
    grid = GridSpec(
        Nx=8,
        Ny=6,
        x_aperture_um=80.0,
        y_aperture_um=60.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    return PRRunRequest(
        grid=grid,
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=20.0,
                    waist_y_um=20.0,
                    coherence_group="pr-execution",
                ),
            ),
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            applied_field=0.5,
            gain_length_product=0.1,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(
            Nt=steps,
            dt_normalized=0.01,
            optical_substeps=1,
        ),
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
        initial_A=np.ones((1, grid.Nx, grid.Ny), dtype=np.complex128),
    )


def _assert_same_physical_result(actual, expected) -> None:
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


def test_optional_execution_arguments_preserve_synchronous_result():
    request = _request()

    original_call = run_pr_timedependent(request)
    explicit_none = run_pr_timedependent(
        request,
        cancellation_token=None,
        progress_callback=None,
    )

    _assert_same_physical_result(explicit_none, original_call)
    assert original_call.status == "completed"
    assert original_call.completed_steps == 3
    assert original_call.requested_steps == 3


def test_progress_reports_complete_normalized_material_time_boundaries():
    progress = []

    result = run_pr_timedependent(
        _request(),
        progress_callback=progress.append,
    )

    assert len(progress) == 3
    assert all(isinstance(item, RunProgress) for item in progress)
    assert [item.workflow for item in progress] == [
        PR_TIMEDEPENDENT_WORKFLOW,
    ] * 3
    assert [item.completed_units for item in progress] == [1, 2, 3]
    assert [item.total_units for item in progress] == [3, 3, 3]
    assert [item.current_coordinate for item in progress] == pytest.approx(
        [0.01, 0.02, 0.03]
    )
    assert all(item.coordinate_name == "material_time" for item in progress)
    assert all(item.coordinate_unit == "normalized" for item in progress)
    assert all(item.status == "running" for item in progress)
    assert all(item.checkpoint_available is True for item in progress)
    assert all(item.elapsed_wall_time >= 0.0 for item in progress)

    latest = progress[-1].latest_field_state
    np.testing.assert_array_equal(latest["E_current"], result.E_final)
    np.testing.assert_array_equal(latest["A_current"], result.A_final)
    np.testing.assert_array_equal(
        latest["source_intensity_stack"],
        result.source_intensity_stack,
    )
    assert latest["material_time_normalized"] == pytest.approx(0.03)
    assert result.checkpoint is not None
    assert result.checkpoint.completed_steps == 3


def test_pre_cancelled_run_returns_untouched_material_state():
    request = _request()
    token = CancellationToken()
    token.cancel()

    result = run_pr_timedependent(
        request,
        cancellation_token=token,
    )

    assert result.status == "cancelled"
    assert result.completed_steps == 0
    assert result.requested_steps == 3
    assert result.time_normalized == 0.0
    np.testing.assert_array_equal(result.E_final, result.E_initial)
    assert np.isfinite(result.A_final).all()
    assert np.isfinite(result.source_intensity_stack).all()
    assert result.power_final == pytest.approx(result.power_initial)


def test_cancellation_after_progress_returns_latest_complete_step():
    request = _request()
    token = CancellationToken()
    observed = []

    def stop_after_first(item: RunProgress) -> None:
        observed.append(item)
        item.latest_field_state["E_current"][...] = -999.0
        token.cancel()

    partial = run_pr_timedependent(
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )
    direct_one_step = run_pr_timedependent(
        replace(request, solver=replace(request.solver, Nt=1))
    )

    assert len(observed) == 1
    assert partial.status == "cancelled"
    assert partial.completed_steps == 1
    assert partial.requested_steps == 3
    assert partial.time_normalized == pytest.approx(0.01)
    _assert_same_physical_result(partial, direct_one_step)
    assert np.max(np.abs(partial.E_final)) < 999.0


def test_cancelled_result_converts_to_shared_product_data():
    request = _request()
    token = CancellationToken()

    def stop(item: RunProgress) -> None:
        if item.completed_units == 1:
            token.cancel()

    result = run_pr_timedependent(
        request,
        cancellation_token=token,
        progress_callback=stop,
    )
    data = pr_result_to_run_data(result)

    summary = data.diagnostics["summary"].values
    assert data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    assert summary["status"] == "cancelled"
    assert summary["completed_material_steps"] == 1
    assert summary["requested_material_steps"] == 3
    assert summary["material_time_normalized"] == pytest.approx(0.01)
    np.testing.assert_array_equal(
        data.fields["final_E_stack"].data,
        result.E_final,
    )
