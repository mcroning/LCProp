import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import (
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
    OutputOptions,
)
from lcprop.workflows import (
    run_static,
    run_timedependent,
    run_soliton,
    run_soliton_existence,
)
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.soliton_existence import SolitonExistenceRequest


def make_base_static_request(power_mW=0.05):
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
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=power_mW,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )


def assert_finite_array(a):
    assert np.isfinite(np.asarray(a)).all()


def test_static_fixed_theta_workflow_public_api():
    req = make_base_static_request()
    result = run_static(req)

    assert result.A_final.shape == (1, 24, 24)
    assert result.theta_final.shape == (24, 24)
    assert result.power_final > 0.0
    assert_finite_array(result.A_final)
    assert_finite_array(result.theta_final)


def test_static_local_self_consistent_workflow_public_api():
    req = make_base_static_request()
    req = StaticRunRequest(
        grid=req.grid,
        material=req.material,
        bias=req.bias,
        beams=req.beams,
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            max_iterations=2,
        ),
        output=req.output,
    )

    result = run_static(req)

    assert result.A_final.shape == (1, 24, 24)
    assert result.theta_final.shape == (24, 24)
    assert result.method == "local_self_consistent"
    assert_finite_array(result.A_final)
    assert_finite_array(result.theta_final)


def test_timedependent_workflow_public_api():
    base = make_base_static_request()

    req = TimeDependentRunRequest(
        grid=base.grid,
        material=base.material,
        bias=base.bias,
        beams=base.beams,
        solver=TimeDependentSolverOptions(Nt=2, dt=7.5e-4, gamma_z=0.0),
        output=base.output,
    )

    result = run_timedependent(req)

    assert result.A_final.shape == (1, 24, 24)
    assert result.theta_final.shape == (2, 24, 24)
    assert result.Nt == 2
    assert_finite_array(result.A_final)
    assert_finite_array(result.theta_final)


def test_soliton_workflow_public_api():
    base = make_base_static_request()

    req = SolitonRequest(
        base=base,
        max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )

    result = run_soliton(req)

    assert result.kind == "SolitonResult"
    assert result.A.shape == (1, 24, 24)
    assert result.theta.shape == (24, 24)
    assert "beta" in result.metrics
    assert "theta_max" in result.metrics
    assert_finite_array(result.A)
    assert_finite_array(result.theta)


def test_soliton_existence_workflow_public_api():
    base = make_base_static_request()

    req = SolitonExistenceRequest(
        base=base,
        powers_mW=(0.03, 0.05),
        continuation=True,
        soliton_max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )

    result = run_soliton_existence(req)

    assert result.kind == "SolitonExistenceResult"
    assert result.metrics["num_points"] == 2
    assert len(result.samples) == 2
    assert len(result.results) == 2
    assert result.samples[1]["continuation_used"] is True
