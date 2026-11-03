from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_static
from tests.test_all_workflows import make_base_static_request


def test_static_run_data_has_physical_geometry():
    run_data = to_run_data(run_static(make_base_static_request()))

    assert run_data.geometry.x is not None
    assert run_data.geometry.y is not None
    assert run_data.geometry.z is not None
    assert run_data.geometry.units == "um"

    assert run_data.geometry.extent_xy() is not None
    assert len(run_data.geometry.extent_xy()) == 4
