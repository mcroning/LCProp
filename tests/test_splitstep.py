import numpy as np

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.optics.launch import build_launch, total_power
from lcprop.optics.splitstep import (
    as_channel_stack,
    total_intensity,
    weighted_theta_intensity,
    linear_kernel,
    hop_linear,
    nonlinear_phase,
    advance_slice_with_midintensity,
)


def test_as_channel_stack_promotes_2d():
    A = np.ones((8, 10), dtype=np.complex64)
    B = as_channel_stack(A)
    assert B.shape == (1, 8, 10)


def test_total_intensity_incoherent_vs_coherent():
    A = np.ones((2, 4, 5), dtype=np.complex64)

    I_incoh = total_intensity(A, coherent=False)
    I_coh = total_intensity(A, coherent=True)

    assert np.allclose(I_incoh, 2.0)
    assert np.allclose(I_coh, 4.0)


def test_weighted_theta_intensity():
    A = np.ones((2, 4, 5), dtype=np.complex64)
    weights = np.asarray([1.0, 3.0], dtype=np.float32)

    I = weighted_theta_intensity(A, weights)

    assert np.allclose(I, 4.0)


def test_linear_hop_preserves_power():
    grid = make_grid(GridSpec(Nx=64, Ny=64, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=1.0),))
    launch = build_launch(beams, grid)

    A0 = launch.A0.copy()
    p0 = total_power(A0, grid)

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
    )

    A1 = hop_linear(A0, kernel)
    p1 = total_power(A1, grid)

    assert np.isclose(p1, p0, rtol=1e-5)


def test_nonlinear_phase_unit_magnitude():
    theta = np.ones((16, 16), dtype=np.float32) * 0.1

    phase = nonlinear_phase(
        theta,
        dz=5.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
    )

    assert np.allclose(np.abs(phase), 1.0)


def test_advance_slice_with_midintensity_shapes():
    grid = make_grid(GridSpec(Nx=32, Ny=32, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=1.0),))
    launch = build_launch(beams, grid)

    theta = np.zeros((32, 32), dtype=np.float32)

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
    )

    A, I_before, I_after, I_mid = advance_slice_with_midintensity(
        launch.A0.copy(),
        theta,
        kernel=kernel,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
    )

    assert A.shape == (1, 32, 32)
    assert I_before.shape == (32, 32)
    assert I_after.shape == (32, 32)
    assert I_mid.shape == (32, 32)
