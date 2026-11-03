import numpy as np

from lcprop.core.context import BiasSpec, LCMaterial, GridSpec
from lcprop.core.grid import make_grid
from lcprop.lc.bias import build_bias


def test_exact_bias_nonzero_theta_bc_edges_and_finite():
    grid = make_grid(GridSpec(Nx=128, Ny=64))
    material = LCMaterial()
    bias = BiasSpec(theta_bc=0.1)

    result = build_bias(bias, grid, material)
    theta = np.asarray(result.theta_2d)

    assert np.allclose(theta[0, :], 0.1)
    assert np.allclose(theta[-1, :], 0.1)
    assert np.isfinite(theta).all()
    assert theta.shape == (128, 64)
