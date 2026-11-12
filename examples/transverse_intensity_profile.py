"""Build a transverse mean intensity profile from a retained XY field."""

import numpy as np

from lcprop.products.data_model import CurveData, FieldData, Geometry, RunData


def mean_intensity_over_y(field: FieldData, geometry: Geometry) -> CurveData:
    """Average samples over y while retaining x and the intensity units."""
    is_intensity = field.quantity in {"intensity", "normalized_intensity"}
    if not is_intensity or field.axes not in (("y", "x"), ("x", "y")):
        raise ValueError("Expected an XY intensity field")

    intensity = np.asarray(field.data)
    x_axis = field.axes.index("x")
    x = np.asarray(field.coordinates["x"] if "x" in field.coordinates else geometry.coord("x"))
    if intensity.ndim != 2 or x.ndim != 1 or x.size != intensity.shape[x_axis]:
        raise ValueError("Expected one x coordinate per sample along the x axis")

    x_unit = field.units.get("x", geometry.units)
    return CurveData(
        key=f"{field.key}_mean_over_y",
        display_name="Mean intensity over y",
        x=x.copy(),
        y=intensity.mean(axis=field.axes.index("y")),
        x_label=f"x ({x_unit})",
        y_label=f"Mean intensity ({field.value_unit})",
        units={"x": x_unit, "y": field.value_unit},
    )


if __name__ == "__main__":
    run = RunData(workflow="synthetic", geometry=Geometry(x=np.array([-2, 0, 2])))
    intensity = np.array([[1., 2., 3.], [3., 4., 5.]])
    field = FieldData(
        key="sample_intensity", display_name="Sample intensity",
        data=intensity, axes=("y", "x"), kind="plane",
        quantity="intensity", value_unit="W/cm²",
    )
    run.fields.add(field.key, field)
    profile = mean_intensity_over_y(run.fields[field.key], run.geometry)
    np.testing.assert_array_equal(profile.x, [-2, 0, 2])
    np.testing.assert_array_equal(profile.y, [2, 3, 4])
    assert profile.units == {"x": "um", "y": "W/cm²"}
    np.testing.assert_array_equal(field.data, [[1, 2, 3], [3, 4, 5]])
    assert not np.shares_memory(profile.x, run.geometry.x)
    assert not np.shares_memory(profile.y, field.data)
    assert len(run.curves) == 0
    print(list(zip(profile.x.tolist(), profile.y.tolist())))
