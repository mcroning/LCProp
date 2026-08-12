import numpy as np

from lcprop.core.context import GridSpec, TimeSpec as LegacyTimeSpec
from lcprop.core.grid import make_grid, round_nz, from_cell
from lcprop.lc.normalization import (
    TimeSpec,
    from_cell as lc_from_cell,
    lc_grid_summary,
    make_lc_spatial_normalization,
)


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
    assert not hasattr(grid, "du")
    assert not hasattr(grid, "dv")
    assert "du" not in grid.summary()
    assert "dv" not in grid.summary()


def test_lc_spatial_normalization_preserves_trusted_theta_convention():
    spec = GridSpec(
        Nx=8,
        Ny=10,
        dz_um=5.0,
        x_aperture_um=80.0,
        y_aperture_um=100.0,
        z_length_um=50.0,
    )
    grid = make_grid(spec)

    normalization = make_lc_spatial_normalization(grid)

    np.testing.assert_allclose(normalization.u, grid.x_um / 40.0)
    np.testing.assert_allclose(normalization.v, grid.y_um / 40.0)
    assert normalization.du == 2.0 / 7.0
    assert normalization.dv == (2.0 / 7.0) * (grid.dy_um / grid.dx_um)
    assert lc_grid_summary(grid)["du"] == normalization.du
    assert lc_grid_summary(grid)["dv"] == normalization.dv


def test_from_cell():
    assert from_cell is lc_from_cell
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


def test_legacy_time_spec_resolves_to_lc_owner():
    assert LegacyTimeSpec is TimeSpec
    assert TimeSpec.__module__ == "lcprop.lc.normalization"
