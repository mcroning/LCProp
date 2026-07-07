import numpy as np

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid, round_nz, from_cell


def test_round_nz():
    assert round_nz(500.0, 5.0) == 100
    assert round_nz(1.0, 5.0) == 1


def test_make_grid_shapes_and_spacing():
    spec = GridSpec(
        Nx=8,
        Ny=10,
        dz_um=5.0,
        x_aperture_um=80.0,
        y_aperture_um=100.0,
        z_length_um=50.0,
    )

    grid = make_grid(spec)

    assert grid.Nx == 8
    assert grid.Ny == 10
    assert grid.Nz == 10
    assert np.isclose(grid.dx_um, 10.0)
    assert np.isclose(grid.dy_um, 10.0)
    assert grid.x_um.shape == (8,)
    assert grid.y_um.shape == (10,)
    assert grid.fxy2_um.shape == (8, 10)


def test_from_cell():
    spec = from_cell(
        Nx=16,
        Ny=32,
        dz_um=2.0,
        thickness_um=75.0,
        y_aperture_um=1000.0,
        interaction_length_um=3000.0,
    )

    assert spec.x_aperture_um == 75.0
    assert spec.y_aperture_um == 1000.0
    assert spec.z_length_um == 3000.0
