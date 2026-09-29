from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.grid import grid_metadata, make_grid, round_nz


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_metadata_and_runtime_share_exact_small_grid_conventions(dtype):
    spec = GridSpec(Nx=12, Ny=8, x_aperture_um=37, y_aperture_um=21,
                    z_length_um=27, dz_um=4)
    meta = grid_metadata(spec)
    runtime = make_grid(spec, real_dtype=dtype)
    for name in ("spec", "Nx", "Ny", "Nz", "dx_um", "dy_um", "dz_um"):
        assert getattr(meta, name) == getattr(runtime, name)
    assert meta.Nz == round_nz(27, 4)
    np.testing.assert_array_equal(runtime.x_um, ((np.arange(12, dtype=dtype)-6)*meta.dx_um+0.5*meta.dx_um).astype(dtype))
    fx = np.fft.fftfreq(12, meta.dx_um).astype(dtype)
    fy = np.fft.fftfreq(8, meta.dy_um).astype(dtype)
    np.testing.assert_array_equal(runtime.fxy2_um, (fx[:, None]**2 + fy[None, :]**2).astype(dtype))
    assert not any(isinstance(v, np.ndarray) for v in vars(meta).values())
    with pytest.raises(FrozenInstanceError):
        meta.Nx = 100


@pytest.mark.parametrize("field,value", [
    ("Nx", 2.5), ("Nx", True), ("Ny", 1), ("Ny", float("nan")),
    ("dz_um", float("inf")), ("x_aperture_um", float("nan")),
    ("y_aperture_um", 0), ("z_length_um", -1),
])
def test_structurally_invalid_grid_rejected(field, value):
    with pytest.raises(ValueError):
        grid_metadata(replace(GridSpec(), **{field: value}))


def test_no_resource_or_gui_limit_in_structural_metadata():
    assert grid_metadata(GridSpec(Nx=2**40, Ny=np.int64(8192))).Nx == 2**40
