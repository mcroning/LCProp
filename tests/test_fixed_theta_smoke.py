import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions

from lcprop.lc.bias import build_bias
from lcprop.optics.launch import build_launch, total_power
from lcprop.optics.splitstep import linear_kernel, advance_slice


def test_one_slice_fixed_theta_smoke():
    request = StaticRunRequest(
        grid=GridSpec(
            Nx=64,
            Ny=64,
            dz_um=5.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=50.0,
        ),
        material=LCMaterial(
            ne=1.7,
            no=1.5,
            K=7e-12,
            delta_epsilon=13.0,
        ),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=1.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            ),
        ),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )

    grid = make_grid(request.grid)
    material = LCMaterial()
    bias = build_bias(request.bias, grid, material)
    launch = build_launch(request.beams, grid)

    A = launch.A0.copy()
    p0 = total_power(A, grid)

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=request.beams.channels[0].wavelength_um,
        n_ref=request.material.no,
    )

    advance_slice(
        A,
        bias.theta_2d,
        kernel=kernel,
        dz=grid.dz_um,
        wavelength=request.beams.channels[0].wavelength_um,
        n_ref=request.material.no,
        ne=request.material.ne,
        no=request.material.no,
    )

    p1 = total_power(A, grid)

    assert A.shape == (1, 64, 64)
    assert bias.theta_stack.shape == (10, 64, 64)
    assert np.isfinite(np.asarray(A)).all()
    assert np.isclose(p1, p0, rtol=1e-5)
