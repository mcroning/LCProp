from __future__ import annotations

from dataclasses import replace

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PRTransverseMarchingStaticOptions,
    PRTransverseMarchingStaticRunRequest,
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
    run_pr_transverse_static_marching,
)
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.static import (
    project_production_resolved_modes,
    static_equilibrium_residual,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.workflow import advance_pr_slice_with_midpoint_source


def _request(
    *,
    dz_um: float = 5.0,
    z_length_um: float = 15.0,
    gain_length_product: float = 1.0e-3,
    scattering=None,
    initial_A=None,
    solver=None,
):
    return PRTransverseMarchingStaticRunRequest(
        grid=GridSpec(
            Nx=16,
            Ny=16,
            x_aperture_um=40.0,
            y_aperture_um=40.0,
            dz_um=dz_um,
            z_length_um=z_length_um,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=10.0,
            coherence_group="marching-static",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=gain_length_product,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=solver or PRTransverseMarchingStaticOptions(),
        backend=BackendSpec(
            backend="numpy", precision="float64", verbose=False
        ),
        initial_A=initial_A,
        scattering=scattering,
    )


def _global_request(request):
    return PRTransverseStaticRunRequest(
        grid=request.grid,
        beams=request.beams,
        material=request.material,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
        solver=PRTransverseStaticWorkflowOptions(
            material_solver=request.solver.material_solver,
            max_coupled_iterations=10,
            equilibrium_rms_tolerance=request.solver.equilibrium_rms_tolerance,
            equilibrium_max_tolerance=request.solver.equilibrium_max_tolerance,
            optical_substeps=request.solver.optical_substeps,
        ),
        backend=request.backend,
        initial_A=request.initial_A,
        scattering=request.scattering,
    )


def _relative_l2(left, right):
    return float(np.linalg.norm((left - right).ravel()) / np.linalg.norm(right.ravel()))


def test_uniform_intensity_is_exact_uniform_zero_flux_equilibrium():
    initial_A = np.ones((1, 16, 16), dtype=np.complex128)
    result = run_pr_transverse_static_marching(
        _request(initial_A=initial_A, gain_length_product=0.2)
    )

    assert result.converged
    assert np.array_equal(result.psi_accepted, np.zeros_like(result.psi_accepted))
    assert all(summary.local_iterations == 0 for summary in result.interval_summaries)
    assert all(summary.carrier_minimum == 1.0 for summary in result.interval_summaries)
    assert all(summary.carrier_mean_error == 0.0 for summary in result.interval_summaries)
    assert all(summary.potential_gauge_error == 0.0 for summary in result.interval_summaries)


def test_zero_optical_response_matches_existing_global_optical_propagation():
    request = _request(gain_length_product=0.0)
    marching = run_pr_transverse_static_marching(request)
    global_result = run_pr_transverse_static(_global_request(request))

    assert marching.converged and global_result.converged
    np.testing.assert_allclose(marching.A_final, global_result.A_final, rtol=0, atol=2e-14)
    assert abs(marching.diagnostics["optical_power_relative_drift"]) < 2e-14


def test_no_scattering_march_converges_and_records_local_work():
    result = run_pr_transverse_static_marching(_request())

    assert result.converged
    assert result.completed_intervals == 3
    assert result.failed_interval is None
    assert all(summary.converged for summary in result.interval_summaries)
    assert all(summary.local_iterations <= 2 for summary in result.interval_summaries)
    assert sum(summary.backtracks for summary in result.interval_summaries) == 0
    assert result.psi_accepted.shape == (3, 16, 16)
    assert result.incoming_field_stack.shape == (4, 1, 16, 16)
    assert result.pre_scattering_exit_stack.shape == (3, 1, 16, 16)
    assert PR_TRANSVERSE_STATIC_OPERATION.run is run_pr_transverse_static


def test_accepted_source_is_exact_entrance_exit_arithmetic_midpoint():
    request = _request()
    result = run_pr_transverse_static_marching(request)
    peak = channel_peak_intensity_reference(result.A_initial, xp=np)

    for index in range(result.completed_intervals):
        before = pr_driving_intensity(
            result.incoming_field_stack[index],
            peak_intensity_reference=peak,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=np,
        )
        after = pr_driving_intensity(
            result.pre_scattering_exit_stack[index],
            peak_intensity_reference=peak,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=np,
        )
        np.testing.assert_allclose(
            result.source_intensity_stack[index],
            0.5 * (before + after),
            rtol=0,
            atol=2e-15,
        )


def test_every_accepted_plane_satisfies_refreshed_zero_flux_residual():
    request = _request()
    result = run_pr_transverse_static_marching(request)
    dxn = request.material.characteristic_wavenumber_per_um * result.grid_summary["dx_um"]
    dyn = request.material.characteristic_wavenumber_per_um * result.grid_summary["dy_um"]

    for index, summary in enumerate(result.interval_summaries):
        residual = static_equilibrium_residual(
            result.psi_accepted[index],
            result.source_intensity_stack[index],
            dx_normalized=dxn,
            dy_normalized=dyn,
        )
        resolved = project_production_resolved_modes(
            residual, dx_normalized=dxn, dy_normalized=dyn
        )
        rms = float(np.sqrt(np.mean(resolved * resolved)))
        maximum = float(np.max(np.abs(resolved)))
        assert rms <= request.solver.equilibrium_rms_tolerance
        assert maximum <= request.solver.equilibrium_max_tolerance
        np.testing.assert_allclose(
            residual, result.equilibrium_residual_stack[index], rtol=0, atol=0
        )
        assert rms == summary.equilibrium_rms
        assert maximum == summary.equilibrium_max


def test_scattering_is_applied_once_after_source_and_is_phase_only():
    scattering = PRCanonicalScatteringSpec(
        epsilon=1.0e-8,
        transverse_correlation_um=2.0,
        realization_seed=9182,
        canonical_dz_um=5.0,
        algorithm_version=PR_CANONICAL_SCATTERING_V2,
    )
    request = _request(scattering=scattering)
    first = run_pr_transverse_static_marching(request)
    second = run_pr_transverse_static_marching(request)

    assert first.converged and second.converged
    np.testing.assert_array_equal(first.psi_accepted, second.psi_accepted)
    np.testing.assert_array_equal(first.A_final, second.A_final)
    for index in range(first.completed_intervals):
        np.testing.assert_allclose(
            np.abs(first.pre_scattering_exit_stack[index]) ** 2,
            np.abs(first.incoming_field_stack[index + 1]) ** 2,
            rtol=2e-15,
            atol=2e-15,
        )
    assert abs(first.diagnostics["optical_power_relative_drift"]) < 2e-14


def test_failure_preserves_bitwise_identical_accepted_upstream_prefix(monkeypatch):
    import lcprop.pr.transverse.marching_static as module

    baseline = run_pr_transverse_static_marching(_request())
    original_scattering = module._apply_canonical_scattering_after_slice
    original_solve = module.solve_pr_transverse_static_intensity
    accepted_boundaries = 0

    def counted_scattering(*args, **kwargs):
        nonlocal accepted_boundaries
        original_scattering(*args, **kwargs)
        accepted_boundaries += 1

    def fail_after_two_intervals(*args, **kwargs):
        material = original_solve(*args, **kwargs)
        if accepted_boundaries >= 2:
            return replace(material, converged=False, status="forced_test_failure")
        return material

    monkeypatch.setattr(module, "_apply_canonical_scattering_after_slice", counted_scattering)
    monkeypatch.setattr(module, "solve_pr_transverse_static_intensity", fail_after_two_intervals)
    failed = run_pr_transverse_static_marching(_request())

    assert not failed.converged
    assert failed.completed_intervals == 2
    assert failed.failed_interval == 2
    assert failed.interval_summaries[-1].termination_reason == "material_forced_test_failure"
    np.testing.assert_array_equal(failed.psi_accepted, baseline.psi_accepted[:2])
    np.testing.assert_array_equal(
        failed.incoming_field_stack, baseline.incoming_field_stack[:3]
    )
    assert failed.psi_accepted.shape[0] == 2
    assert failed.source_intensity_stack.shape[0] == 2


def test_marching_and_global_workflows_agree_on_benign_case():
    request = _request(gain_length_product=1.0e-3)
    marching = run_pr_transverse_static_marching(request)
    global_result = run_pr_transverse_static(_global_request(request))

    assert marching.converged and global_result.converged
    assert _relative_l2(marching.A_final, global_result.A_final) < 2e-10
    assert _relative_l2(marching.psi_accepted, global_result.psi_final) < 2e-7
    assert (
        _relative_l2(
            marching.source_intensity_stack,
            global_result.source_intensity_stack,
        )
        < 2e-10
    )


def test_small_longitudinal_refinement_probe_runs_at_two_spacings():
    coarse = run_pr_transverse_static_marching(
        _request(dz_um=5.0, z_length_um=10.0)
    )
    fine = run_pr_transverse_static_marching(
        _request(dz_um=2.5, z_length_um=10.0)
    )

    assert coarse.converged and fine.converged
    assert coarse.completed_intervals == 2
    assert fine.completed_intervals == 4
    assert np.isfinite(_relative_l2(coarse.A_final, fine.A_final))


def test_accepted_local_state_replays_one_interval_without_scattering():
    request = _request(z_length_um=5.0)
    result = run_pr_transverse_static_marching(request)
    dx = request.grid.x_aperture_um / request.grid.Nx
    dy = request.grid.y_aperture_um / request.grid.Ny
    fxy2 = (
        np.fft.fftfreq(request.grid.Nx, d=dx)[:, None] ** 2
        + np.fft.fftfreq(request.grid.Ny, d=dy)[None, :] ** 2
    )
    kernel = linear_kernel(
        fxy2,
        dz=request.grid.dz_um,
        wavelength=0.633,
        n_ref=request.material.refractive_index,
        xp=np,
    )
    dxn = request.material.characteristic_wavenumber_per_um * dx
    dyn = request.material.characteristic_wavenumber_per_um * dy
    state = state_from_potential(
        result.psi_accepted[0], dx_normalized=dxn, dy_normalized=dyn
    )
    active = project_active_field(
        state.E_x, state.E_y, profile=request.projection, xp=np
    )
    peak = channel_peak_intensity_reference(result.A_initial, xp=np)
    replay_A, replay_source = advance_pr_slice_with_midpoint_source(
        result.A_initial,
        active,
        kernel=kernel,
        optical_substeps=1,
        dz_um=request.grid.dz_um,
        wavelength_um=0.633,
        interaction_length_um=request.grid.z_length_um,
        gain_length_product=request.material.gain_length_product,
        peak_intensity_reference=peak,
        background_intensity=request.material.background_intensity,
        coherence_groups=request.beams.coherence_groups,
        xp=np,
    )
    np.testing.assert_allclose(
        replay_A, result.pre_scattering_exit_stack[0], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        replay_source, result.source_intensity_stack[0], rtol=0, atol=0
    )
