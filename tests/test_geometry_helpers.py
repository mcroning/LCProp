import numpy as np

from lcprop.products.data_model import Geometry


def test_geometry_spacing_and_values():
    geom = Geometry(
        x=np.array([-1.0, 0.0, 1.0]),
        y=np.array([-2.0, 0.0, 2.0]),
        z=np.array([0.0, 5.0, 10.0]),
    )

    assert geom.dx() == 1.0
    assert geom.dy() == 2.0
    assert geom.dz() == 5.0
    assert geom.value("x", 2) == 1.0
    assert geom.value("z", 1) == 5.0


def test_geometry_nearest_index_and_extents():
    geom = Geometry(
        x=np.array([-1.0, 0.0, 1.0]),
        y=np.array([-2.0, 0.0, 2.0]),
        z=np.array([0.0, 5.0, 10.0]),
    )

    assert geom.nearest_index("x", 0.2) == 1
    assert geom.nearest_index("y", 1.8) == 2
    assert geom.extent_xy() == [-1.0, 1.0, -2.0, 2.0]
    assert geom.extent_zx() == [0.0, 10.0, -1.0, 1.0]
    assert geom.extent_zy() == [0.0, 10.0, -2.0, 2.0]
