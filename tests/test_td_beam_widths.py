import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.products.diagnostics import rms_widths


@pytest.mark.parametrize(
    ("sigma_x", "sigma_y", "x0", "y0"),
    (
        (4.0, 4.0, 0.0, 0.0),
        (3.0, 7.0, 0.0, 0.0),
        (3.0, 7.0, 8.0, -6.0),
    ),
)
def test_gaussian_rms_widths_use_centroid_subtracted_grid_coordinates(
    sigma_x,
    sigma_y,
    x0,
    y0,
):
    grid = make_grid(
        GridSpec(
            Nx=256,
            Ny=256,
            x_aperture_um=100.0,
            y_aperture_um=100.0,
            z_length_um=1.0,
            dz_um=1.0,
        ),
        real_dtype=np.float64,
    )
    intensity = np.exp(
        -0.5
        * (
            ((grid.x_um[:, None] - x0) / sigma_x) ** 2
            + ((grid.y_um[None, :] - y0) / sigma_y) ** 2
        )
    )

    measured_x, measured_y = rms_widths(intensity, grid)

    assert measured_x == pytest.approx(sigma_x, rel=1e-6)
    assert measured_y == pytest.approx(sigma_y, rel=1e-6)
