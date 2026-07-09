import numpy as np
from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions
from lcprop.workflows.soliton import SolitonRequest, run_soliton
from lcprop.workflows.soliton_existence import (
    SolitonExistenceRequest,
    make_static_request_for_power,
    run_soliton_existence,
)


def make_base_request():
    return StaticRunRequest(
        grid=GridSpec(
            Nx=24,
            Ny=24,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=10.0,
        ),
        material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(power_mW=1.0, waist_x_um=3.0, waist_y_um=3.0),
            )
        ),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )


def test_make_static_request_for_power_scales_beams():
    base = make_base_request()
    req = make_static_request_for_power(base, 0.25)

    assert req.beams.channels[0].power_mW == 0.25
    assert base.beams.channels[0].power_mW == 1.0


def test_run_soliton_existence_smoke():
    request = SolitonExistenceRequest(
        base=make_base_request(),
        powers_mW=(0.05, 0.1),
        continuation=True,
        soliton_max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )

    result = run_soliton_existence(request)

    assert result.kind == "SolitonExistenceResult"
    assert result.metrics["num_points"] == 2
    assert len(result.samples) == 2
    assert len(result.results) == 2
    assert result.samples[0]["requested_power_mW"] == 0.05
    assert result.samples[1]["continuation_used"] is True


def _as_numpy(a):
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _relative_symmetry_error(a, transformed):
    a = np.asarray(a)
    transformed = np.asarray(transformed)
    denom = np.linalg.norm(a.ravel()) + 1e-300
    return float(np.linalg.norm((a - transformed).ravel()) / denom)


def test_soliton_mode_10_enforces_field_and_theta_parity():
    request = SolitonRequest(
        base=make_base_request(),
        mode="10",
        max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )

    result = run_soliton(request)
    A = _as_numpy(result.A)
    theta = _as_numpy(result.theta)

    # A is odd in x and even in y for the 10 family.
    assert _relative_symmetry_error(A, -A[..., ::-1, :]) < 1e-12
    assert _relative_symmetry_error(A, A[..., :, ::-1]) < 1e-12

    # Theta is driven by intensity, so it remains even in both directions.
    assert _relative_symmetry_error(theta, theta[::-1, :]) < 1e-12
    assert _relative_symmetry_error(theta, theta[:, ::-1]) < 1e-12


def test_soliton_mode_01_enforces_field_and_theta_parity():
    request = SolitonRequest(
        base=make_base_request(),
        mode="01",
        max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )

    result = run_soliton(request)
    A = _as_numpy(result.A)
    theta = _as_numpy(result.theta)

    # A is even in x and odd in y for the 01 family.
    assert _relative_symmetry_error(A, A[..., ::-1, :]) < 1e-12
    assert _relative_symmetry_error(A, -A[..., :, ::-1]) < 1e-12

    # Theta is driven by intensity, so it remains even in both directions.
    assert _relative_symmetry_error(theta, theta[::-1, :]) < 1e-12
    assert _relative_symmetry_error(theta, theta[:, ::-1]) < 1e-12
