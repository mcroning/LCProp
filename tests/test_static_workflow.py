import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions
from lcprop.workflows.static import run_static


def test_run_static_fixed_theta_workflow():
    request = StaticRunRequest(
        grid=GridSpec(
            Nx=64,
            Ny=64,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=50.0,
        ),
        material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )

    result = run_static(request)

    assert result.A_final.shape == (1, 64, 64)
    assert result.theta_final.shape == (64, 64)
    assert result.n_steps == 10
    assert np.isclose(result.power_final, result.power_initial, rtol=1e-5)
    assert result.grid_summary["Nz"] == 10
    assert "fixed prepared theta" in result.warnings[0]
