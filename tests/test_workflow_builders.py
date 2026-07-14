from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, OutputOptions
from lcprop.workflows.runtime import (
    build_runtime_components,
    initial_theta_intensity,
    make_picard_theta_relax,
    make_global_uniform_theta_iteration,
    make_td_optics_step,
    make_zcoupled_theta_step,
)


def make_request():
    return StaticRunRequest(
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
        solver=StaticSolverOptions(),
        output=OutputOptions(),
    )


def test_build_runtime_components_smoke():
    components = build_runtime_components(make_request())

    assert components.grid.Nx == 32
    assert components.grid.Nz == 4
    assert components.launch.A0.shape == (1, 32, 32)
    assert components.coherence_groups == components.launch.coherence_groups
    assert components.bias.theta_stack.shape == (4, 32, 32)
    assert components.b > 0.0
    assert components.bi > 0.0


def test_builder_callbacks_smoke():
    components = build_runtime_components(make_request())

    I0 = initial_theta_intensity(components)
    assert I0.shape == (32, 32)

    theta_relax = make_picard_theta_relax(components)
    theta1 = theta_relax(components.bias.theta_2d, I0, 1)
    assert theta1.shape == (32, 32)

    optics_update = make_global_uniform_theta_iteration(components)
    A2, I2 = optics_update(components.launch.A0, theta1, 1)
    assert A2.shape == (1, 32, 32)
    assert I2.shape == (32, 32)

    optics_step = make_td_optics_step(components)
    A3, I3 = optics_step(components.launch.A0.copy(), components.bias.theta_2d, 0)
    assert A3.shape == (1, 32, 32)
    assert I3.shape == (32, 32)

    theta_step = make_zcoupled_theta_step(components, gamma_z=0.0)
    theta4 = theta_step(
        components.bias.theta_2d,
        I3,
        components.bias.theta_2d,
        components.bias.theta_2d,
        0,
    )
    assert theta4.shape == (32, 32)
