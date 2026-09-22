from tests.physical_launch_oracles import scalar_lineage_flux, normal_gaussian_norm
import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions
from lcprop.optics.launch import OpticalLaunchContext, build_launch
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
                BeamChannel(wavelength_um=0.633, power_mW=1.0, w1_um=3.0, w2_um=3.0),
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
    assert np.isclose(result.power_initial, normal_gaussian_norm(request.grid, request.beams), rtol=1e-12)
    assert np.isclose(result.physical_power_initial_mW, scalar_lineage_flux(result.A_initial, request.grid, request.beams, request.material.no), rtol=1e-12)
    assert np.isclose(result.physical_power_final_mW, scalar_lineage_flux(result.A_final, request.grid, request.beams, request.material.no), rtol=1e-12)
    assert result.grid_summary["Nz"] == 10
    assert result.grid_summary["du"] == 2.0 / 63.0
    assert result.grid_summary["dv"] == (2.0 / 63.0) * (1.0 / 0.75)
    assert "fixed prepared theta" in result.warnings[0]


def test_lc_workflow_supplies_physical_material_context():
    request = StaticRunRequest(
        grid=GridSpec(
            Nx=96,
            Ny=80,
            dz_um=10.0,
            x_aperture_um=96.0,
            y_aperture_um=80.0,
            z_length_um=20.0,
        ),
        material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    theta_ext_rad=.03, w1_um=8.0, w2_um=10.0,
                ),
            )
        ),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )
    result = run_static(request)
    grid = make_grid(request.grid, real_dtype=np.float64)
    expected = build_launch(
        request.beams,
        grid,
        complex_dtype=np.complex128,
        context=OpticalLaunchContext(
            grid=grid,
            n_ref=request.material.no,
            interaction_length_um=request.grid.z_length_um,
        ),
    )

    np.testing.assert_array_equal(result.A_initial, expected.A0)
