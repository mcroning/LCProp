from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions


def test_static_request_construction():
    grid = GridSpec(
        Nx=256,
        Ny=256,
        dz_um=5.0,
        x_aperture_um=75.0,
        y_aperture_um=1000.0,
        z_length_um=500.0,
    )

    material = LCMaterial(
        ne=1.7,
        no=1.5,
        K=1.2e-11,
        delta_epsilon=10.3,
    )

    bias = BiasSpec(
        V_bias=1.0,
        theta_bc=0.0,
    )

    beam = BeamChannel(
        wavelength_um=0.633,
        power_mW=1.0,
        waist_x_um=3.0,
        waist_y_um=3.0,
        x0_um=0.0,
        y0_um=0.0,
        tilt_x_rad_per_um=0.0,
        tilt_y_rad_per_um=0.0,
        phase_rad=0.0,
    )

    request = StaticRunRequest(
        grid=grid,
        material=material,
        bias=bias,
        beams=BeamStack(channels=(beam,)),
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )

    assert request.grid.Nx == 256
    assert request.material.ne == 1.7
    assert len(request.beams.channels) == 1
    assert request.beams.channels[0].power_mW == 1.0
