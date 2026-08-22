from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.marching_static import (
    PR_MARCHING_FLOAT64_MATERIAL_COMPLEX64_OPTICS_V1,
    PRTransverseMarchingStaticOptions,
    PRTransverseMarchingStaticRunRequest,
    run_pr_transverse_static_marching,
)
from lcprop.pr.transverse.static import PRTransverseStaticMaterialSolverOptions


def _material_options(precision: str):
    if precision == "float64":
        return PRTransverseStaticMaterialSolverOptions()
    return PRTransverseStaticMaterialSolverOptions(
        pcg_relative_tolerance=2.0e-5,
        pcg_absolute_tolerance=1.0e-7,
        pcg_near_gate_relative_tolerance=1.0e-7,
        pcg_near_gate_absolute_tolerance=1.0e-9,
        equilibrium_rms_tolerance=4.0e-6,
        equilibrium_max_tolerance=2.0e-5,
        td_rhs_rms_tolerance=3.0e-6,
        td_rhs_max_tolerance=2.0e-5,
    )


def _request(*, backend, precision="float64", scattering=None, uniform=False):
    initial_A = (
        np.ones((1, 64, 64), dtype=np.complex128 if precision == "float64" else np.complex64)
        if uniform
        else None
    )
    float32 = precision == "float32"
    return PRTransverseMarchingStaticRunRequest(
        grid=GridSpec(
            Nx=64,
            Ny=64,
            x_aperture_um=40.0,
            y_aperture_um=40.0,
            dz_um=5.0,
            z_length_um=20.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=10.0,
            coherence_group="marching-static-gpu",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=1.0e-3,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseMarchingStaticOptions(
            material_solver=_material_options(precision),
            equilibrium_rms_tolerance=5.0e-6 if float32 else 1.0e-8,
            equilibrium_max_tolerance=5.0e-5 if float32 else 1.0e-7,
        ),
        backend=BackendSpec(backend=backend, precision=precision, verbose=False),
        initial_A=initial_A,
        scattering=scattering,
    )


def _scattering():
    return PRCanonicalScatteringSpec(
        epsilon=1.0e-8,
        transverse_correlation_um=2.0,
        realization_seed=9182,
        canonical_dz_um=5.0,
        algorithm_version=PR_CANONICAL_SCATTERING_V2,
    )


def _assert_histories_equal(left, right):
    assert [item.local_iterations for item in left.interval_summaries] == [
        item.local_iterations for item in right.interval_summaries
    ]
    np.testing.assert_allclose(
        [item.accepted_damping for item in left.interval_summaries],
        [item.accepted_damping for item in right.interval_summaries],
        rtol=0,
        atol=0,
    )
    assert [item.backtracks for item in left.interval_summaries] == [
        item.backtracks for item in right.interval_summaries
    ]
    assert [item.accepted for item in left.iteration_records] == [
        item.accepted for item in right.iteration_records
    ]


def test_cupy_uniform_equilibrium_and_backend_provenance():
    result = run_pr_transverse_static_marching(
        _request(backend="cupy", uniform=True)
    )
    assert result.converged
    assert result.backend_summary["backend"] == "cupy"
    assert result.backend_summary["is_gpu"] is True
    assert np.array_equal(result.psi_accepted, np.zeros_like(result.psi_accepted))
    assert all(item.carrier_minimum == 1.0 for item in result.interval_summaries)


@pytest.mark.parametrize("scattering", (None, _scattering()))
def test_cupy_float64_complete_march_matches_numpy(scattering):
    expected = run_pr_transverse_static_marching(
        _request(backend="numpy", scattering=scattering)
    )
    actual = run_pr_transverse_static_marching(
        _request(backend="cupy", scattering=scattering)
    )
    assert expected.converged and actual.converged
    _assert_histories_equal(actual, expected)
    np.testing.assert_allclose(actual.A_final, expected.A_final, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual.psi_accepted, expected.psi_accepted, rtol=2e-10, atol=2e-11
    )
    np.testing.assert_allclose(
        actual.source_intensity_stack,
        expected.source_intensity_stack,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.pre_scattering_exit_stack,
        expected.pre_scattering_exit_stack,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.equilibrium_residual_stack,
        expected.equilibrium_residual_stack,
        rtol=2e-10,
        atol=2e-11,
    )


@pytest.mark.parametrize("scattering", (None, _scattering()))
def test_cupy_mixed_precision_matches_numpy_mixed_precision_and_float64_reference(
    scattering,
):
    reference = run_pr_transverse_static_marching(
        _request(backend="numpy", scattering=scattering)
    )
    expected = run_pr_transverse_static_marching(
        _request(backend="numpy", precision="float32", scattering=scattering)
    )
    actual = run_pr_transverse_static_marching(
        _request(backend="cupy", precision="float32", scattering=scattering)
    )
    assert reference.converged and expected.converged and actual.converged
    assert actual.A_final.dtype == np.complex64
    assert actual.psi_accepted.dtype == np.float64
    assert actual.source_intensity_stack.dtype == np.float64
    assert (
        actual.diagnostics["precision_policy_id"]
        == PR_MARCHING_FLOAT64_MATERIAL_COMPLEX64_OPTICS_V1
    )
    np.testing.assert_allclose(actual.A_final, expected.A_final, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        actual.psi_accepted, expected.psi_accepted, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        actual.source_intensity_stack,
        expected.source_intensity_stack,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        actual.A_final, reference.A_final, rtol=2e-6, atol=2e-7
    )
    assert abs(actual.diagnostics["optical_power_relative_drift"]) < 2e-6


def test_cupy_scattering_restart_is_deterministic():
    request = _request(
        backend="cupy", precision="float32", scattering=_scattering()
    )
    baseline = run_pr_transverse_static_marching(request)
    partial = run_pr_transverse_static_marching(
        request, stop_after_accepted_intervals=2
    )
    assert partial.restart_state is not None
    assert partial.restart_state.previous_psi.dtype == np.float64
    assert partial.restart_state.A_incoming.dtype == np.complex64
    assert partial.restart_state.previous_material_source.dtype == np.float64
    resumed = run_pr_transverse_static_marching(
        request, restart_state=partial.restart_state
    )
    assert resumed.converged
    np.testing.assert_array_equal(resumed.A_final, baseline.A_final)
    np.testing.assert_array_equal(
        np.concatenate((partial.psi_accepted, resumed.psi_accepted)),
        baseline.psi_accepted,
    )


def test_cupy_failure_prefix_retains_restart_and_failure_arrays(monkeypatch):
    import lcprop.pr.transverse.marching_static as module

    request = _request(backend="cupy", precision="float32")
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
    result = run_pr_transverse_static_marching(request)
    assert result.completed_intervals == 2
    assert result.failed_interval == 2
    assert result.restart_state is not None
    assert result.failure_state is not None
    assert result.failure_state.trial_psi.shape == (64, 64)
    assert result.failure_state.midpoint_source.shape == (64, 64)
    assert result.failure_state.midpoint_source.dtype == np.float64
    assert result.failure_state.A_incoming.dtype == np.complex64
    assert result.failure_state.material_iteration_records
