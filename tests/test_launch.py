import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.optics.launch import (
    build_launch,
    channel_power_integrals,
    reconstructed_physical_powers_mW,
    total_power,
)
from lcprop.optics.splitstep import total_intensity


@pytest.mark.parametrize("resolution", [64, 128])
def test_single_gaussian_launch_power_normalization(resolution):
    grid = make_grid(GridSpec(Nx=resolution, Ny=resolution, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=2.0),))
    launch = build_launch(beams, grid)

    assert launch.A0.shape == (1, resolution, resolution)
    assert np.isclose(total_power(launch.A0, grid), 1.0, rtol=1e-6)
    assert np.allclose(launch.power_fractions, [1.0])
    assert np.allclose(launch.physical_powers_mW, [2.0])


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
    assert launch.coherence_groups == ("__lcprop_coherent__", "__lcprop_coherent__")
    assert np.allclose(channel_power_integrals(launch.A0, grid), [0.25, 0.75])
    assert np.isclose(total_power(launch.A0, grid), 1.0, rtol=1e-6)
    assert np.allclose(launch.power_fractions, [0.25, 0.75])
    assert np.allclose(
        reconstructed_physical_powers_mW(launch.A0, grid, launch),
        [1.0, 3.0],
    )
    assert launch.summary()["physical_channel_powers_mW"] == [1.0, 3.0]
    assert launch.summary()["physical_total_power_mW"] == 4.0
    assert launch.summary()["power_fractions"] == [0.25, 0.75]
    assert launch.summary()["field_normalization"] == (
        "sum_channel_integrals_equals_one"
    )
    assert np.allclose(np.asarray(launch.wavelengths_um), [0.633, 0.633])


def test_legacy_incoherent_stack_migrates_to_distinct_groups():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(channels=(BeamChannel(name="a"), BeamChannel(name="b")))

    launch = build_launch(beams, grid)

    assert len(set(launch.coherence_groups)) == 2
    assert launch.summary()["coherence_groups"] == list(launch.coherence_groups)


def test_explicit_coherence_groups_are_preserved():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(
            BeamChannel(name="a", coherence_group="A"),
            BeamChannel(name="b", coherence_group="A"),
            BeamChannel(name="c", coherence_group="B"),
        )
    )

    launch = build_launch(beams, grid)

    assert launch.coherence_groups == ("A", "A", "B")


def test_partially_migrated_coherence_groups_are_rejected():
    beams = BeamStack(
        channels=(
            BeamChannel(name="legacy"),
            BeamChannel(name="explicit", coherence_group="laser"),
        )
    )

    with pytest.raises(ValueError, match="coherence_group migration is incomplete"):
        beams.validate()


def test_zero_power_channel_is_zero():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(BeamChannel(power_mW=0.0), BeamChannel(power_mW=1.0))
    )

    launch = build_launch(beams, grid)

    assert np.allclose(launch.A0[0], 0.0)
    assert np.isclose(total_power(launch.A0, grid), 1.0)


def test_all_zero_power_channels_are_rejected():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(channels=(BeamChannel(power_mW=0.0),))

    with pytest.raises(ValueError, match="total physical power must be positive"):
        build_launch(beams, grid)


def test_tilt_adds_phase_variation():
    grid = make_grid(GridSpec(Nx=64, Ny=64))

    no_tilt = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.0),))
    tilted = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.1),))

    A0 = build_launch(no_tilt, grid).A0[0]
    A1 = build_launch(tilted, grid).A0[0]

    assert np.allclose(np.abs(A0), np.abs(A1), rtol=1e-5, atol=1e-7)
    assert not np.allclose(A0, A1)


def test_coherent_group_interference_does_not_renormalize_channel_fractions():
    grid = make_grid(GridSpec(Nx=128, Ny=128))
    in_phase = BeamStack(
        channels=(
            BeamChannel(power_mW=1.0, coherence_group="A"),
            BeamChannel(power_mW=1.0, coherence_group="A"),
        )
    )
    out_of_phase = BeamStack(
        channels=(
            BeamChannel(power_mW=1.0, coherence_group="A"),
            BeamChannel(power_mW=1.0, phase_rad=np.pi, coherence_group="A"),
        )
    )

    launch_in = build_launch(in_phase, grid)
    launch_out = build_launch(out_of_phase, grid)
    area = grid.dx_um * grid.dy_um
    grouped_in = total_intensity(
        launch_in.A0,
        coherence_groups=launch_in.coherence_groups,
    )
    grouped_out = total_intensity(
        launch_out.A0,
        coherence_groups=launch_out.coherence_groups,
    )

    assert np.allclose(channel_power_integrals(launch_in.A0, grid), [0.5, 0.5])
    assert np.allclose(channel_power_integrals(launch_out.A0, grid), [0.5, 0.5])
    assert np.isclose(np.sum(grouped_in) * area, 2.0, rtol=1e-6)
    assert np.sum(grouped_out) * area < 1e-12
