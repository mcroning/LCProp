import numpy as np

from lcprop.core.context import GridSpec, BiasSpec, LCMaterial
from lcprop.core.grid import make_grid
from lcprop.lc.bias_cosine import build_cosine_bias_2d
from lcprop.lc.bias import stack_theta, build_bias


def test_cosine_bias_shape_and_boundary_rows():
    grid = make_grid(GridSpec(Nx=64, Ny=32, z_length_um=50.0))
    bias = BiasSpec(theta_bc=0.0)

    theta = build_cosine_bias_2d(bias, grid)

    assert theta.shape == (64, 32)
    assert np.allclose(theta[0, :], 0.0)
    assert np.allclose(theta[-1, :], 0.0)
    assert theta.max() > 0.5


def test_cosine_bias_uses_theta_center():
    grid = make_grid(GridSpec(Nx=65, Ny=16, z_length_um=50.0))
    bias = BiasSpec(theta_bc=0.0, theta_center=0.3)

    theta = build_cosine_bias_2d(bias, grid)

    assert np.isclose(theta[grid.Nx // 2, grid.Ny // 2], 0.3, atol=1e-2)


def test_stack_theta_shape():
    grid = make_grid(GridSpec(Nx=16, Ny=8, dz_um=5.0, z_length_um=50.0))
    theta_2d = np.ones((16, 8), dtype=np.float32)

    theta_stack = stack_theta(theta_2d, grid)

    assert theta_stack.shape == (10, 16, 8)


def test_build_bias_result():
    grid = make_grid(GridSpec(Nx=16, Ny=8, dz_um=5.0, z_length_um=50.0))
    bias = BiasSpec(theta_bc=0.1)
    material = LCMaterial()
    result = build_bias(bias, grid, material)

    assert result.theta_2d.shape == (16, 8)
    assert result.theta_stack.shape == (10, 16, 8)
    assert result.theta_bc == 0.1
    assert result.theta_clamp == (bias.theta_min, bias.theta_max)
