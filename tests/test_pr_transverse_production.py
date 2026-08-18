from __future__ import annotations

import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    pr_transverse_result_to_run_data,
    run_pr_transverse_timedependent,
)
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.transport import (
    explicit_euler_step,
    potential_rhs,
    state_from_potential,
)
from lcprop.pr.transverse_reference import (
    TransverseReferenceOptions,
    potential_rhs as reference_potential_rhs,
    state_from_potential as reference_state_from_potential,
)
from lcprop.runners.local import LocalRunner


def _spectral_x(field: np.ndarray, dx: float) -> np.ndarray:
    kx = 2.0 * math.pi * np.fft.fftfreq(field.shape[0], d=dx)[:, None]
    return np.fft.ifft(1j * kx * np.fft.fft(field, axis=0), axis=0).real


def _spectral_xx(field: np.ndarray, dx: float) -> np.ndarray:
    kx = 2.0 * math.pi * np.fft.fftfreq(field.shape[0], d=dx)[:, None]
    return np.fft.ifft(-(kx * kx) * np.fft.fft(field, axis=0), axis=0).real


def _request(*, steps: int = 2, precision: str = "float64"):
    grid = GridSpec(
        Nx=12,
        Ny=10,
        x_aperture_um=48.0,
        y_aperture_um=40.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    return PRTransverseRunRequest(
        grid=grid,
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=10.0,
            waist_y_um=8.0,
            coherence_group="transverse-pr",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            applied_field=0.0,
            gain_length_product=0.05,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseSolverOptions(
            Nt=steps,
            dt_normalized=1.0e-4,
            optical_substeps=1,
        ),
        backend=BackendSpec(backend="numpy", precision=precision, verbose=False),
    )


def test_transport_matches_transparent_reference_oracle():
    rng = np.random.default_rng(145)
    psi = rng.normal(scale=1.0e-3, size=(17, 15))
    intensity = 0.2 + rng.random(psi.shape)
    options = TransverseReferenceOptions(0.4, 0.3, 1.0e-4, 1)

    expected_state = reference_state_from_potential(psi, options=options)
    actual_state = state_from_potential(
        psi, dx_normalized=0.4, dy_normalized=0.3
    )
    np.testing.assert_allclose(actual_state.psi, expected_state.psi, rtol=0, atol=2e-16)
    np.testing.assert_allclose(actual_state.carrier_density, expected_state.carrier_density, rtol=0, atol=2e-15)
    np.testing.assert_allclose(actual_state.E_x, expected_state.E_x, rtol=0, atol=2e-16)
    np.testing.assert_allclose(actual_state.E_y, expected_state.E_y, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        potential_rhs(psi, intensity, dx_normalized=0.4, dy_normalized=0.3),
        reference_potential_rhs(psi, intensity, options=options),
        rtol=0,
        atol=3e-15,
    )


def test_y_uniform_production_transport_reduces_to_a7():
    nx = 96
    dx = 2.0 * math.pi / nx
    x = np.arange(nx) * dx
    psi_x = 0.03 * np.cos(x) + 0.01 * np.cos(2.0 * x)
    intensity_x = 1.0 + 0.2 * np.cos(x) + 0.05 * np.cos(3.0 * x)
    psi = np.repeat(psi_x[:, None], 5, axis=1)
    intensity = np.repeat(intensity_x[:, None], 5, axis=1)
    state = state_from_potential(psi, dx_normalized=dx, dy_normalized=1.0)
    psi_tau = potential_rhs(psi, intensity, dx_normalized=dx, dy_normalized=1.0)
    E_tau = -_spectral_x(psi_tau, dx)
    spectral_a7 = (
        -(state.E_x * intensity - _spectral_x(intensity, dx))
        * (1.0 + _spectral_x(state.E_x, dx))
        + intensity * _spectral_xx(state.E_x, dx)
    )
    assert np.linalg.norm(E_tau - spectral_a7) / np.linalg.norm(spectral_a7) < 1e-11
    # The unchanged centered-difference A7 path remains a convergent, separate model.
    centered = hopping_rhs(
        state.E_x,
        intensity,
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=dx,
        xp=np,
    )
    assert np.linalg.norm(centered - E_tau) / np.linalg.norm(E_tau) < 3.1e-3


def test_uniform_equilibrium_is_exact_and_carrier_is_conserved():
    psi = np.zeros((3, 16, 14), dtype=np.float64)
    intensity = np.full_like(psi, 0.31)
    assert np.array_equal(
        potential_rhs(psi, intensity, dx_normalized=0.5, dy_normalized=0.7),
        psi,
    )
    candidate = explicit_euler_step(
        psi, intensity, dt_normalized=0.01,
        dx_normalized=0.5, dy_normalized=0.7,
    )
    assert np.array_equal(candidate, psi)

    rng = np.random.default_rng(2)
    perturbed = rng.normal(scale=1e-4, size=psi.shape)
    initial = state_from_potential(
        perturbed, dx_normalized=0.5, dy_normalized=0.7
    )
    final_psi = explicit_euler_step(
        perturbed, 0.2 + rng.random(psi.shape), dt_normalized=1e-5,
        dx_normalized=0.5, dy_normalized=0.7,
    )
    final = state_from_potential(
        final_psi, dx_normalized=0.5, dy_normalized=0.7
    )
    np.testing.assert_allclose(
        np.sum(final.carrier_density, axis=(-2, -1)),
        np.sum(initial.carrier_density, axis=(-2, -1)),
        rtol=0,
        atol=2e-13,
    )


def test_curl_gauss_and_gauge_are_at_roundoff():
    rng = np.random.default_rng(3)
    psi = rng.normal(scale=1e-3, size=(2, 19, 17))
    state = state_from_potential(
        psi, dx_normalized=0.5, dy_normalized=0.6
    )
    diagnostics = state_diagnostics(
        state, dx_normalized=0.5, dy_normalized=0.6
    )
    assert diagnostics["curl_max"] < 2e-16
    assert diagnostics["gauss_max"] < 3e-15
    assert diagnostics["potential_mean_max_abs"] < 1e-18
    assert diagnostics["finite_material_state"] is True


def test_transport_is_periodic_translation_equivariant():
    rng = np.random.default_rng(4)
    psi = rng.normal(scale=2e-4, size=(2, 15, 13))
    intensity = 0.2 + rng.random(psi.shape)
    shift = (0, 4, -3)
    expected = np.roll(
        potential_rhs(psi, intensity, dx_normalized=0.4, dy_normalized=0.5),
        shift,
        axis=(0, 1, 2),
    )
    actual = potential_rhs(
        np.roll(psi, shift, axis=(0, 1, 2)),
        np.roll(intensity, shift, axis=(0, 1, 2)),
        dx_normalized=0.4,
        dy_normalized=0.5,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_isotropic_transport_preserves_xy_exchange_symmetry():
    rng = np.random.default_rng(41)
    psi = rng.normal(scale=2e-4, size=(17, 17))
    intensity = 0.2 + rng.random(psi.shape)
    direct = potential_rhs(
        psi, intensity, dx_normalized=0.4, dy_normalized=0.4
    )
    transposed = potential_rhs(
        psi.T, intensity.T, dx_normalized=0.4, dy_normalized=0.4
    ).T
    np.testing.assert_allclose(direct, transposed, rtol=0, atol=2e-15)


def test_complete_workflow_is_periodic_translation_equivariant():
    request = _request(steps=1)
    rng = np.random.default_rng(42)
    shape = (1, request.grid.Nx, request.grid.Ny)
    A0 = (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ).astype(np.complex128)
    psi0 = rng.normal(
        scale=2e-5,
        size=(2, request.grid.Nx, request.grid.Ny),
    )
    shift = (3, -2)
    direct = run_pr_transverse_timedependent(
        PRTransverseRunRequest(**{
            **request.__dict__, "initial_A": A0, "initial_psi": psi0,
        })
    )
    shifted = run_pr_transverse_timedependent(
        PRTransverseRunRequest(**{
            **request.__dict__,
            "initial_A": np.roll(A0, shift, axis=(1, 2)),
            "initial_psi": np.roll(psi0, shift, axis=(1, 2)),
        })
    )
    np.testing.assert_allclose(
        shifted.psi_final,
        np.roll(direct.psi_final, shift, axis=(1, 2)),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        shifted.A_final,
        np.roll(direct.A_final, shift, axis=(1, 2)),
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_workflow_is_deterministic_finite_and_power_conserving(precision):
    request = _request(steps=2, precision=precision)
    first = run_pr_transverse_timedependent(request)
    second = run_pr_transverse_timedependent(request)
    assert first.status == "completed"
    assert first.completed_steps == 2
    assert first.resolved_profile["physics_profile_id"] == PR_FULL_TRANSVERSE_PROFILE_V1
    assert first.resolved_profile["material"]["dark_intensity"] == 0.2
    assert first.resolved_profile["beam_request"]["channels"][0]["waist_y_um"] == 8.0
    np.testing.assert_array_equal(first.psi_final, second.psi_final)
    np.testing.assert_array_equal(first.A_final, second.A_final)
    assert first.diagnostics["finite_material_state"] is True
    assert first.diagnostics["finite_optical_state"] is True
    tolerance = 2e-6 if precision == "float32" else 2e-13
    assert abs(first.diagnostics["optical_power_relative_drift"]) < tolerance
    assert first.diagnostics["carrier_relative_drift_max"] < tolerance


def test_cancellation_returns_only_last_accepted_state():
    request = _request(steps=3)
    token = CancellationToken()
    token.cancel()
    result = run_pr_transverse_timedependent(request, cancellation_token=token)
    assert result.status == "cancelled"
    assert result.completed_steps == 0
    np.testing.assert_array_equal(result.psi_final, result.psi_initial)
    assert result.diagnostics["complete_final_optical_replay"] is True


def test_cancellation_after_progress_preserves_one_complete_accepted_step():
    request = _request(steps=3)
    token = CancellationToken()
    accepted = []

    def stop_after_first(progress):
        accepted.append(progress.latest_field_state["psi_current"].copy())
        token.cancel()

    result = run_pr_transverse_timedependent(
        request,
        cancellation_token=token,
        progress_callback=stop_after_first,
    )
    assert result.status == "cancelled"
    assert result.completed_steps == 1
    assert len(accepted) == 1
    np.testing.assert_array_equal(result.psi_final, accepted[0])


def test_products_reconstruct_derived_fields_and_operation_is_composable():
    result = run_pr_transverse_timedependent(_request(steps=1))
    data = pr_transverse_result_to_run_data(result)
    assert data.workflow == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    assert {"psi", "P", "E_x", "E_y", "E_active"}.issubset(data.fields.keys())
    np.testing.assert_array_equal(data.fields["E_active"].data, data.fields["E_x"].data)
    assert PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.key == (
        "pr", PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    )
    through_operation = PR_TRANSVERSE_TIMEDEPENDENT_OPERATION.run(_request(steps=0))
    assert through_operation.completed_steps == 0
    runner = LocalRunner(operations=(PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,))
    runner_result = runner.run_registered(
        "pr", PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW, _request(steps=0)
    )
    assert runner_result.result.completed_steps == 0
    assert runner_result.run_data.workflow == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW


def test_profile_rejects_unapproved_backend_and_biased_material():
    request = _request(steps=0)
    with pytest.raises(ValueError, match="backend='numpy'"):
        run_pr_transverse_timedependent(
            PRTransverseRunRequest(
                **{**request.__dict__, "backend": BackendSpec("auto", "float64", False)}
            )
        )
    with pytest.raises(ValueError, match="applied_field=0"):
        run_pr_transverse_timedependent(
            PRTransverseRunRequest(
                **{
                    **request.__dict__,
                    "material": PRMaterialSpec(applied_field=0.2),
                }
            )
        )
