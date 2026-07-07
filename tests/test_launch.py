import numpy as np

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.optics.launch import build_launch, total_power


def test_single_gaussian_launch_power_normalization():
    grid = make_grid(GridSpec(Nx=128, Ny=128, z_length_um=50.0))

    beams = BeamStack(
        channels=(
            BeamChannel(power_mW=2.0, waist_x_um=3.0, waist_y_um=3.0),
        )
    )

    launch = build_launch(beams, grid)

    assert launch.A0.shape == (1, 128, 128)
    assert np.isclose(total_power(launch.A0, grid), 2.0, rtol=1e-5)


def test_two_channel_launch_power_normalization():
    grid = make_grid(GridSpec(Nx=128, Ny=128, z_length_um=50.0))

    beams = BeamStack(
        channels=(
            BeamChannel(name="a", power_mW=1.0, y0_um=-5.0),
            BeamChannel(name="b", power_mW=3.0, y0_um=5.0),
        ),
        coherence="coherent",
    )

    launch = build_launch(beams, grid)

    assert launch.A0.shape == (2, 128, 128)
    assert launch.coherence == "coherent"
    assert np.isclose(total_power(launch.A0, grid), 4.0, rtol=1e-5)
    assert np.allclose(np.asarray(launch.wavelengths_um), [0.633, 0.633])


def test_zero_power_channel_is_zero():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(channels=(BeamChannel(power_mW=0.0),))

    launch = build_launch(beams, grid)

    assert np.allclose(launch.A0, 0.0)
    assert np.isclose(total_power(launch.A0, grid), 0.0)


def test_tilt_adds_phase_variation():
    grid = make_grid(GridSpec(Nx=64, Ny=64))

    no_tilt = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.0),))
    tilted = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.1),))

    A0 = build_launch(no_tilt, grid).A0[0]
    A1 = build_launch(tilted, grid).A0[0]

    assert np.allclose(np.abs(A0), np.abs(A1), rtol=1e-5, atol=1e-7)
    assert not np.allclose(A0, A1)
