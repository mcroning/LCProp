from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.pr import workflow as reduced_workflow
from lcprop.pr.checkpoint import validate_pr_continuation
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
)
from lcprop.pr.static_streaming import _volume_noise_phase
from lcprop.pr.transverse import (
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    run_pr_transverse_timedependent,
)
from lcprop.pr.workflow import run_pr_timedependent


def _scattering(*, epsilon: float = 0.02, seed: int = 2137319267):
    return PRCanonicalScatteringSpec(
        epsilon=epsilon,
        transverse_correlation_um=1.0,
        realization_seed=seed,
        canonical_dz_um=2.0,
    )


def _common(*, dz_um: float = 2.0, optical_substeps: int = 1, scattering=None):
    grid = GridSpec(
        Nx=16,
        Ny=12,
        x_aperture_um=32.0,
        y_aperture_um=24.0,
        dz_um=dz_um,
        z_length_um=8.0,
    )
    beams = BeamStack(channels=(BeamChannel(
        wavelength_um=0.633,
        waist_x_um=7.0,
        waist_y_um=6.0,
        coherence_group="canonical-scattering-test",
    ),))
    material = PRMaterialSpec(
        dark_intensity=0.1,
        applied_field=0.0,
        gain_length_product=0.05,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.1,
    )
    backend = BackendSpec(backend="numpy", precision="float64", verbose=False)
    reduced = PRRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRSolverOptions(
            Nt=0,
            dt_normalized=1.0e-4,
            optical_substeps=optical_substeps,
        ),
        backend=backend,
        scattering=scattering,
    )
    transverse = PRTransverseRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRTransverseSolverOptions(
            Nt=0,
            dt_normalized=1.0e-4,
            optical_substeps=optical_substeps,
        ),
        backend=backend,
        scattering=scattering,
    )
    return reduced, transverse


@pytest.mark.parametrize("workflow", ["reduced", "transverse"])
def test_disabled_scattering_preserves_prior_td_fields_exactly(workflow):
    reduced, transverse = _common(scattering=None)
    reduced_zero, transverse_zero = _common(scattering=_scattering(epsilon=0.0))
    if workflow == "reduced":
        baseline = run_pr_timedependent(reduced)
        zero = run_pr_timedependent(reduced_zero)
        assert "canonical_scattering" not in baseline.diagnostics
    else:
        baseline = run_pr_transverse_timedependent(transverse)
        zero = run_pr_transverse_timedependent(transverse_zero)
        assert "canonical_scattering" not in baseline.diagnostics
        assert "canonical_scattering" not in baseline.resolved_profile

    np.testing.assert_array_equal(zero.A_initial, baseline.A_initial)
    np.testing.assert_array_equal(zero.A_final, baseline.A_final)
    assert zero.power_final == baseline.power_final


def test_same_seed_is_deterministic_and_shared_across_td_workflows():
    reduced, transverse = _common(scattering=_scattering())
    reduced_first = run_pr_timedependent(reduced)
    reduced_second = run_pr_timedependent(reduced)
    transverse_first = run_pr_transverse_timedependent(transverse)
    transverse_second = run_pr_transverse_timedependent(transverse)

    np.testing.assert_array_equal(reduced_first.A_final, reduced_second.A_final)
    np.testing.assert_array_equal(
        transverse_first.A_final, transverse_second.A_final
    )
    np.testing.assert_array_equal(reduced_first.A_final, transverse_first.A_final)
    assert abs(
        (reduced_first.power_final - reduced_first.power_initial)
        / reduced_first.power_initial
    ) < 2.0e-14
    assert abs(
        (transverse_first.power_final - transverse_first.power_initial)
        / transverse_first.power_initial
    ) < 2.0e-14
    assert (
        reduced_first.diagnostics["canonical_scattering"]
        == transverse_first.diagnostics["canonical_scattering"]
    )
    assert (
        transverse_first.resolved_profile["canonical_scattering"]
        == transverse_first.diagnostics["canonical_scattering"]
    )

    reduced_step = replace(reduced, solver=replace(reduced.solver, Nt=1))
    transverse_step = replace(
        transverse, solver=replace(transverse.solver, Nt=1)
    )
    reduced_step_first = run_pr_timedependent(reduced_step)
    reduced_step_second = run_pr_timedependent(reduced_step)
    transverse_step_first = run_pr_transverse_timedependent(transverse_step)
    transverse_step_second = run_pr_transverse_timedependent(transverse_step)
    np.testing.assert_array_equal(
        reduced_step_first.E_final, reduced_step_second.E_final
    )
    np.testing.assert_array_equal(
        reduced_step_first.A_final, reduced_step_second.A_final
    )
    np.testing.assert_array_equal(
        transverse_step_first.psi_final, transverse_step_second.psi_final
    )
    np.testing.assert_array_equal(
        transverse_step_first.A_final, transverse_step_second.A_final
    )


def _capture_phases(monkeypatch, request, *, transverse: bool):
    original = reduced_workflow._canonical_scattering_phase_for_slice
    captured = []

    def recording(*args, **kwargs):
        phase = original(*args, **kwargs)
        captured.append((int(kwargs["z_index"]), np.asarray(phase).copy()))
        return phase

    monkeypatch.setattr(
        reduced_workflow,
        "_canonical_scattering_phase_for_slice",
        recording,
    )
    if transverse:
        run_pr_transverse_timedependent(request)
    else:
        run_pr_timedependent(request)
    monkeypatch.setattr(
        reduced_workflow,
        "_canonical_scattering_phase_for_slice",
        original,
    )
    return captured


def test_td_dispatch_is_partition_independent_and_applies_one_screen_per_slice(
    monkeypatch,
):
    scattering = _scattering()
    coarse, coarse_transverse = _common(
        dz_um=4.0, optical_substeps=1, scattering=scattering
    )
    coarse_substepped, _ = _common(
        dz_um=4.0, optical_substeps=2, scattering=scattering
    )
    fine, _ = _common(dz_um=2.0, optical_substeps=1, scattering=scattering)

    coarse_phases = _capture_phases(monkeypatch, coarse, transverse=False)
    coarse_substepped_phases = _capture_phases(
        monkeypatch, coarse_substepped, transverse=False
    )
    fine_phases = _capture_phases(monkeypatch, fine, transverse=False)
    transverse_phases = _capture_phases(
        monkeypatch, coarse_transverse, transverse=True
    )

    assert len(coarse_phases) == coarse.grid.z_length_um / coarse.grid.dz_um
    assert len(coarse_substepped_phases) == len(coarse_phases)
    assert len(fine_phases) == fine.grid.z_length_um / fine.grid.dz_um
    assert len(transverse_phases) == len(coarse_phases)
    for index, (_, coarse_phase) in enumerate(coarse_phases):
        np.testing.assert_array_equal(
            coarse_substepped_phases[index][1], coarse_phase
        )
        np.testing.assert_array_equal(transverse_phases[index][1], coarse_phase)
        fine_sum = fine_phases[2 * index][1] + fine_phases[2 * index + 1][1]
        np.testing.assert_allclose(coarse_phase, fine_sum, rtol=0.0, atol=2.0e-16)


def test_td_and_static_construct_the_same_matched_physical_screen():
    spec = _scattering()
    reduced, _ = _common(dz_um=2.0, scattering=spec)
    grid = make_grid(reduced.grid, xp=np, real_dtype=np.float64)
    td_phase = reduced_workflow._canonical_scattering_phase_for_slice(
        spec,
        z_index=2,
        grid=grid,
        z_length_um=reduced.grid.z_length_um,
        xp=np,
    )
    static_request = SimpleNamespace(
        grid=reduced.grid,
        solver=SimpleNamespace(partition_independent_scattering=spec),
    )
    static_phase = _volume_noise_phase(
        2,
        request=static_request,
        grid=grid,
        xp=np,
    )
    np.testing.assert_array_equal(td_phase, static_phase)


def test_scattering_preserves_cancellation_boundaries_and_checkpoint_identity():
    reduced, transverse = _common(scattering=_scattering())
    reduced = replace(reduced, solver=replace(reduced.solver, Nt=1))
    transverse = replace(transverse, solver=replace(transverse.solver, Nt=1))
    token = CancellationToken()
    token.cancel()

    reduced_result = run_pr_timedependent(reduced, cancellation_token=token)
    transverse_result = run_pr_transverse_timedependent(
        transverse, cancellation_token=token
    )
    assert reduced_result.status == "cancelled"
    assert reduced_result.completed_steps == 0
    assert reduced_result.diagnostics["final_optical_observation"] == (
        "unpropagated_launch_fallback"
    )
    assert transverse_result.status == "cancelled"
    assert transverse_result.completed_steps == 0
    assert transverse_result.diagnostics["cancellation_observed_stage"] == (
        "material_step_boundary"
    )

    assert reduced_result.checkpoint is not None
    incompatible = replace(reduced, scattering=_scattering(seed=123))
    with pytest.raises(ValueError, match="incompatible scattering"):
        validate_pr_continuation(incompatible, reduced_result.checkpoint)
