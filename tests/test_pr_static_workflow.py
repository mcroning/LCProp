from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)
from lcprop.pr.workflow import run_pr_timedependent


def _static_request(*, Nz=2, initial_A=None, gain_length_product=0.2):
    grid = GridSpec(
        Nx=12,
        Ny=4,
        x_aperture_um=24.0,
        y_aperture_um=16.0,
        dz_um=5.0,
        z_length_um=5.0 * Nz,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=8.0,
                waist_y_um=7.0,
            ),
        )
    )
    material = PRMaterialSpec(
        dark_intensity=0.2,
        uniform_background_intensity=0.1,
        applied_field=0.5,
        gain_length_product=gain_length_product,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.2,
    )
    solver = PRStaticWorkflowOptions(
        material_solver=PRStaticSolverOptions(max_iterations=20),
        max_coupled_passes=15,
        residual_rms_tolerance=1e-8,
        residual_max_tolerance=1e-7,
        optical_substeps=1,
    )
    return PRStaticRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=solver,
        backend=BackendSpec(
            backend="numpy", precision="float64", verbose=False
        ),
        initial_A=initial_A,
    )


def test_one_slice_coupled_solution_is_a_strict_fixed_source_root():
    request = _static_request(Nz=1)
    result = run_pr_static(request)
    dx_normalized = (
        request.material.characteristic_wavenumber_per_um
        * request.grid.x_aperture_um
        / request.grid.Nx
    )
    independent = solve_pr_static_intensity(
        result.source_intensity_stack[0],
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity,
        dx_normalized=dx_normalized,
        initial_E=result.E_final[0],
        options=request.solver.material_solver,
    )

    assert result.converged
    assert result.slice_summaries[0].converged
    assert independent.converged
    assert np.allclose(result.E_final[0], independent.E, rtol=0.0, atol=2e-9)
    assert result.slice_summaries[0].final_residual_rms <= (
        request.solver.residual_rms_tolerance
    )
    assert result.slice_summaries[0].final_residual_max <= (
        request.solver.residual_max_tolerance
    )


def test_coupled_histories_decrease_and_independent_replay_is_exact():
    request = _static_request(Nz=2)
    result = run_pr_static(request)

    assert result.converged
    assert len(result.iteration_records) >= 2
    assert all(record.accepted for record in result.iteration_records)
    assert all(
        record.residual_after_rms < record.residual_before_rms
        for record in result.iteration_records
    )
    assert all(
        0.0 < record.step_scale <= 1.0
        for record in result.iteration_records
    )
    assert all(
        summary.termination_reason == "residual_tolerance"
        for summary in result.slice_summaries
    )
    assert result.replay_diagnostics["field_consistent"]
    assert result.replay_diagnostics["source_consistent"]
    assert result.replay_diagnostics["residual_consistent"]
    assert result.replay_diagnostics["field_max_abs_difference"] == 0.0
    assert result.replay_diagnostics["source_max_abs_difference"] == 0.0
    assert result.replay_diagnostics["residual_max_abs_difference"] == 0.0
    assert result.power_final == pytest.approx(result.power_initial, rel=2e-14)


def test_optional_execution_arguments_preserve_completed_static_result():
    request = _static_request(Nz=2)
    progress = []

    original = run_pr_static(request)
    explicit_none = run_pr_static(
        request,
        cancellation_token=None,
        progress_callback=None,
    )
    reported = run_pr_static(request, progress_callback=progress.append)

    assert explicit_none.status == original.status == "converged"
    assert explicit_none.completed_slices == original.completed_slices == 2
    np.testing.assert_array_equal(explicit_none.A_final, original.A_final)
    np.testing.assert_array_equal(explicit_none.E_final, original.E_final)
    np.testing.assert_array_equal(
        explicit_none.source_intensity_stack,
        original.source_intensity_stack,
    )
    np.testing.assert_array_equal(
        explicit_none.residual_stack,
        original.residual_stack,
    )
    assert explicit_none.iteration_records == original.iteration_records
    assert explicit_none.slice_summaries == original.slice_summaries
    assert explicit_none.replay_diagnostics == original.replay_diagnostics
    assert explicit_none.tolerance_provenance == original.tolerance_provenance
    np.testing.assert_array_equal(reported.A_final, original.A_final)
    np.testing.assert_array_equal(reported.E_final, original.E_final)
    np.testing.assert_array_equal(
        reported.source_intensity_stack,
        original.source_intensity_stack,
    )
    np.testing.assert_array_equal(
        reported.residual_stack,
        original.residual_stack,
    )
    assert reported.converged == original.converged
    assert reported.replay_diagnostics == original.replay_diagnostics
    assert (progress[-1].diagnostics or {}).get("phase") == (
        "scientific_result"
    )


def test_static_progress_and_cancellation_stop_at_an_accepted_slice_boundary():
    request = _static_request(Nz=3)
    token = CancellationToken()
    progress = []

    def stop_after_first(item: RunProgress) -> None:
        progress.append(item)
        token.cancel()

    result = run_pr_static(
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )

    assert result.status == "cancelled"
    assert not result.converged
    assert result.completed_slices == 1
    assert len(result.slice_summaries) == 1
    assert result.E_initial.shape == (1, 12, 4)
    assert result.E_final.shape == (1, 12, 4)
    assert result.source_intensity_stack.shape == (1, 12, 4)
    assert result.residual_stack.shape == (1, 12, 4)
    assert result.replay_diagnostics["performed_slices"] == 1
    assert result.replay_diagnostics["requested_slices"] == 3
    assert result.replay_diagnostics["field_consistent"]
    assert result.replay_diagnostics["source_consistent"]
    assert result.replay_diagnostics["residual_consistent"]
    assert not result.replay_diagnostics["residual_converged"]

    solve_updates = [
        item
        for item in progress
        if (item.diagnostics or {}).get("phase") == "solve"
    ]
    assert len(solve_updates) == 1
    assert [
        (item.diagnostics or {}).get("phase") for item in progress
    ] == [
        "solve",
        "replay",
        "replay",
        "replay_validation",
        "scientific_result",
    ]
    update = solve_updates[0]
    assert update.workflow == "pr_static"
    assert update.completed_units == 1
    assert update.total_units == 3
    assert update.current_coordinate == pytest.approx(request.grid.dz_um)
    assert update.coordinate_name == "z"
    assert update.coordinate_unit == "um"
    assert update.checkpoint_available is False
    assert update.diagnostics["converged"]
    np.testing.assert_array_equal(
        update.latest_field_state["A_current"],
        result.A_final,
    )
    np.testing.assert_array_equal(
        update.latest_field_state["E_current"],
        result.E_final[0],
    )


def test_static_replay_progress_cadence_is_bounded_for_large_slice_counts():
    progress = []
    result = run_pr_static(
        _static_request(Nz=205, gain_length_product=0.0),
        progress_callback=progress.append,
    )

    replay = [
        item
        for item in progress
        if (item.diagnostics or {}).get("phase") == "replay"
    ]

    assert result.converged
    assert replay[0].completed_units == 0
    assert replay[-1].completed_units == 205
    assert len(replay) <= 102
    assert all(item.total_units == 205 for item in replay)


def test_pre_cancelled_static_run_reports_no_trial_slice_as_physical_state():
    request = _static_request(Nz=2)
    token = CancellationToken()
    token.cancel()

    result = run_pr_static(request, cancellation_token=token)

    assert result.status == "cancelled"
    assert not result.converged
    assert result.completed_slices == 0
    assert result.iteration_records == ()
    assert result.slice_summaries == ()
    assert result.E_initial.shape == (0, 12, 4)
    assert result.E_final.shape == (0, 12, 4)
    assert result.source_intensity_stack.shape == (0, 12, 4)
    assert result.residual_stack.shape == (0, 12, 4)
    np.testing.assert_array_equal(result.A_final, result.A_initial)
    assert result.power_final == result.power_initial


def test_stale_fixed_source_root_cannot_establish_coupled_convergence(monkeypatch):
    import lcprop.pr.static_workflow as static_workflow

    def state_dependent_source(A_in, E_slice, **_kwargs):
        return A_in.copy(), 1.0 + 0.5 * E_slice

    monkeypatch.setattr(
        static_workflow,
        "advance_pr_slice_with_midpoint_source",
        state_dependent_source,
    )
    initial_A = np.ones((1, 8, 3), dtype=np.complex128)
    request = PRStaticRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=3,
            x_aperture_um=8.0,
            y_aperture_um=3.0,
            dz_um=1.0,
            z_length_um=1.0,
        ),
        beams=BeamStack(
            channels=(BeamChannel(waist_x_um=2.0, waist_y_um=2.0),)
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            applied_field=0.5,
            characteristic_wavenumber_per_um_override=1.0,
        ),
        solver=PRStaticWorkflowOptions(
            material_solver=PRStaticSolverOptions(
                residual_rms_tolerance=1e-13,
                residual_max_tolerance=1e-13,
            ),
            max_coupled_passes=20,
            residual_rms_tolerance=1e-12,
            residual_max_tolerance=1e-12,
        ),
        initial_A=initial_A,
    )
    stale_root = request.material.applied_field * 0.2
    stale_fresh_residual = hopping_rhs(
        np.full((8, 3), stale_root),
        np.full((8, 3), 1.0 + 0.5 * stale_root),
        applied_field=request.material.applied_field,
        background_intensity=request.material.background_intensity,
        dx_normalized=1.0,
        xp=np,
    )

    result = run_pr_static(request)
    expected = -1.0 + np.sqrt(1.2)

    assert np.max(np.abs(stale_fresh_residual)) == pytest.approx(0.005)
    assert result.converged
    assert len(result.iteration_records) > 1
    assert np.allclose(result.E_final, expected, rtol=0.0, atol=1e-12)
    assert result.replay_diagnostics["residual_max"] < 1e-12


def test_long_time_td_workflow_approaches_coupled_static_plane_wave():
    grid = GridSpec(
        Nx=8,
        Ny=3,
        x_aperture_um=8.0,
        y_aperture_um=3.0,
        dz_um=2.0,
        z_length_um=2.0,
    )
    beams = BeamStack(
        channels=(BeamChannel(waist_x_um=2.0, waist_y_um=2.0),)
    )
    material = PRMaterialSpec(
        dark_intensity=0.2,
        uniform_background_intensity=0.1,
        applied_field=0.5,
        gain_length_product=0.3,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=1.0,
    )
    initial_A = np.ones((1, grid.Nx, grid.Ny), dtype=np.complex128)
    backend = BackendSpec(backend="numpy", precision="float64", verbose=False)
    static = run_pr_static(
        PRStaticRunRequest(
            grid=grid,
            beams=beams,
            material=material,
            solver=PRStaticWorkflowOptions(
                residual_rms_tolerance=1e-11,
                residual_max_tolerance=1e-11,
            ),
            backend=backend,
            initial_A=initial_A,
        )
    )
    transient = run_pr_timedependent(
        PRRunRequest(
            grid=grid,
            beams=beams,
            material=material,
            solver=PRSolverOptions(
                Nt=200,
                dt_normalized=0.1,
                optical_substeps=1,
                integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
            ),
            backend=backend,
            initial_A=initial_A,
        )
    )

    assert static.converged
    assert np.allclose(
        transient.E_final, static.E_final, rtol=0.0, atol=2e-11
    )
    assert np.allclose(
        transient.A_final, static.A_final, rtol=0.0, atol=2e-11
    )


def test_numpy_float32_static_workflow_agrees_with_float64_reference():
    reference_request = replace(
        _static_request(Nz=2),
        solver=PRStaticWorkflowOptions(),
    )
    float32_request = replace(
        reference_request,
        backend=BackendSpec(
            backend="numpy", precision="float32", verbose=False
        ),
    )
    reference = run_pr_static(reference_request)
    actual = run_pr_static(float32_request)

    assert reference.converged
    assert actual.converged
    assert actual.backend_summary["real_dtype"] == "float32"
    assert actual.tolerance_provenance == {
        "precision": "float32",
        "material_solver": {
            "source": "precision_default",
            "residual_rms_tolerance": 2e-6,
            "residual_max_tolerance": 1e-5,
        },
        "coupled_residual_rms_tolerance": {
            "value": 2e-6,
            "source": "precision_default",
        },
        "coupled_residual_max_tolerance": {
            "value": 1e-5,
            "source": "precision_default",
        },
        "replay_rtol": {
            "value": 2e-6,
            "source": "precision_default",
        },
        "replay_atol": {
            "value": 2e-7,
            "source": "precision_default",
        },
    }
    assert reference.tolerance_provenance["precision"] == "float64"
    assert reference.tolerance_provenance["material_solver"] == {
        "source": "precision_default",
        "residual_rms_tolerance": 1e-10,
        "residual_max_tolerance": 1e-9,
    }
    assert np.allclose(actual.E_final, reference.E_final, rtol=3e-6, atol=3e-7)
    assert np.allclose(actual.A_final, reference.A_final, rtol=3e-6, atol=3e-7)
    assert actual.power_final == pytest.approx(actual.power_initial, rel=2e-6)


def test_explicit_static_tolerances_are_preserved_and_provenanced():
    material_solver = PRStaticSolverOptions(
        max_iterations=23,
        residual_rms_tolerance=3e-6,
        residual_max_tolerance=2e-5,
    )
    request = replace(
        _static_request(Nz=1),
        solver=PRStaticWorkflowOptions(
            material_solver=material_solver,
            residual_rms_tolerance=4e-6,
            residual_max_tolerance=3e-5,
            replay_rtol=5e-6,
            replay_atol=6e-7,
        ),
        backend=BackendSpec(
            backend="numpy", precision="float32", verbose=False
        ),
    )

    result = run_pr_static(request)

    assert result.converged
    assert request.solver.material_solver is material_solver
    assert result.tolerance_provenance == {
        "precision": "float32",
        "material_solver": {
            "source": "explicit_PRStaticSolverOptions",
            "residual_rms_tolerance": 3e-6,
            "residual_max_tolerance": 2e-5,
        },
        "coupled_residual_rms_tolerance": {
            "value": 4e-6,
            "source": "explicit_override",
        },
        "coupled_residual_max_tolerance": {
            "value": 3e-5,
            "source": "explicit_override",
        },
        "replay_rtol": {
            "value": 5e-6,
            "source": "explicit_override",
        },
        "replay_atol": {
            "value": 6e-7,
            "source": "explicit_override",
        },
    }


def test_explicit_strict_float32_tolerances_are_not_automatically_relaxed():
    request = replace(
        _static_request(Nz=1),
        solver=PRStaticWorkflowOptions(
            material_solver=PRStaticSolverOptions(),
            residual_rms_tolerance=1e-8,
            residual_max_tolerance=1e-7,
            replay_rtol=1e-11,
            replay_atol=1e-12,
        ),
        backend=BackendSpec(
            backend="numpy", precision="float32", verbose=False
        ),
    )

    result = run_pr_static(request)

    assert not result.converged
    assert result.status == "not_converged"
    assert result.slice_summaries[0].termination_reason == (
        "material_line_search_failed"
    )
    provenance = result.tolerance_provenance
    assert provenance["material_solver"] == {
        "source": "explicit_PRStaticSolverOptions",
        "residual_rms_tolerance": 1e-10,
        "residual_max_tolerance": 1e-9,
    }
    for name in (
        "coupled_residual_rms_tolerance",
        "coupled_residual_max_tolerance",
        "replay_rtol",
        "replay_atol",
    ):
        assert provenance[name]["source"] == "explicit_override"


def test_cupy_static_workflow_agrees_with_numpy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    cpu_request = _static_request(Nz=2)
    gpu_request = replace(
        cpu_request,
        backend=BackendSpec(
            backend="cupy", precision="float64", verbose=False
        ),
    )
    cpu = run_pr_static(cpu_request)
    gpu = run_pr_static(gpu_request)

    assert cpu.converged
    assert gpu.converged
    assert gpu.backend_summary["backend"] == "cupy"
    assert gpu.backend_summary["is_gpu"]
    assert np.allclose(gpu.E_final, cpu.E_final, rtol=2e-11, atol=2e-12)
    assert np.allclose(gpu.A_final, cpu.A_final, rtol=2e-11, atol=2e-12)
    assert np.allclose(
        gpu.source_intensity_stack,
        cpu.source_intensity_stack,
        rtol=2e-11,
        atol=2e-12,
    )
    assert gpu.power_final == pytest.approx(gpu.power_initial, rel=2e-13)


def test_coupled_workflow_does_not_call_dense_newton_direction(monkeypatch):
    import lcprop.pr.static as static_material

    def fail_dense_direction(*_args, **_kwargs):
        raise AssertionError("coupled workflow called dense Newton direction")

    monkeypatch.setattr(
        static_material,
        "_newton_direction",
        fail_dense_direction,
    )

    result = run_pr_static(_static_request(Nz=1))

    assert result.converged
