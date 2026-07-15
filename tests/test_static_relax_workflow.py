import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, StaticWorkflowOptions, OutputOptions
from lcprop.workflows.static import run_static


def test_run_static_relax_workflow_smoke():
    request = StaticRunRequest(
        grid=GridSpec(
            Nx=32,
            Ny=32,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=20.0,
        ),
        material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=0.1,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            max_iterations=2,
            static_max_coupled_passes=2,
            static_max_relax_iterations=2,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        ),
        output=OutputOptions(),
    )

    result = run_static(request)

    assert result.A_final.shape == (1, 32, 32)
    assert result.theta_final.shape == (4, 32, 32)
    assert result.n_steps == 8
    assert result.method == "local_self_consistent"
    assert np.isfinite(np.asarray(result.A_final)).all()
    assert np.isfinite(np.asarray(result.theta_final)).all()
    assert np.isclose(result.power_initial, 1.0, rtol=1e-6)
    assert np.isclose(result.power_final, 1.0, rtol=1e-5)
    assert np.isclose(result.physical_power_initial_mW, 0.1, rtol=1e-6)
    assert result.launch_summary["field_normalization"] == (
        "sum_channel_integrals_equals_one"
    )
