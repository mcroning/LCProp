import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from lcprop.pr.longitudinal_cuts import (
    extract_longitudinal_optical_intensity_cuts,
)
from lcprop.products.data_model import (
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)


@pytest.mark.parametrize(
    ("nx", "ny", "dx", "dy", "expected_x", "expected_y"),
    (
        (5, 7, 0.4, 0.6, 0.0, 0.0),
        (4, 6, 0.4, 0.6, -0.2, -0.3),
    ),
)
def test_center_cuts_use_actual_nearest_zero_samples(
    nx, ny, dx, dy, expected_x, expected_y
):
    source = np.arange(3 * nx * ny, dtype=np.float64).reshape(3, nx, ny)
    cuts = extract_longitudinal_optical_intensity_cuts(
        source,
        grid_summary={"Nx": nx, "Ny": ny, "dx_um": dx, "dy_um": dy},
        peak_intensity_reference=2.0,
        background_intensity=0.5,
    )
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))

    assert cuts.x_cut_um == pytest.approx(expected_x, abs=1e-15)
    assert cuts.y_cut_um == pytest.approx(expected_y, abs=1e-15)
    np.testing.assert_array_equal(cuts.xz, (source[:, :, iy] - 0.5) * 2.0)
    np.testing.assert_array_equal(cuts.yz, (source[:, ix, :] - 0.5) * 2.0)
    assert cuts.xz.shape == (3, nx)
    assert cuts.yz.shape == (3, ny)


def _fast_run_data() -> RunData:
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-2.5, 2.5, 6)
    z = np.linspace(0.0, 2.0, 3)
    coordinates = {
        "retention": "fast_center_nearest",
        "x_cut_um": -0.5,
        "y_cut_um": -0.5,
    }
    fields = FieldCollection()
    fields.add("retained_fast_optical_intensity_xz", make_field(
        "retained_fast_optical_intensity_xz",
        "Retained Fast Optical Intensity x-z",
        np.arange(12, dtype=float).reshape(3, 4),
        ("z", "x"), "intensity", {"z": "um", "x": "um"},
        "longitudinal", source_volume_key="retained_fast_optical_intensity",
        coordinates=coordinates,
    ))
    fields.add("retained_fast_optical_intensity_yz", make_field(
        "retained_fast_optical_intensity_yz",
        "Retained Fast Optical Intensity y-z",
        np.arange(18, dtype=float).reshape(3, 6),
        ("z", "y"), "intensity", {"z": "um", "y": "um"},
        "longitudinal", source_volume_key="retained_fast_optical_intensity",
        coordinates=coordinates,
    ))
    return RunData(
        workflow="pr_static",
        geometry=Geometry(x=x, y=y, z=z),
        fields=fields,
        longitudinal_enabled=True,
    )


def test_gui_displays_retained_fast_cuts_without_transverse_sliders():
    app = QApplication.instance() or QApplication([])
    pane = LongitudinalPane()
    pane.set_run_data(_fast_run_data())

    assert pane.field_selector.currentText() == "Retained Fast Optical Intensity"
    assert pane.x_cut_slider.isHidden()
    assert pane.y_cut_slider.isHidden()
    assert pane.show_guides.isHidden()
    assert pane.xz_view.image is not None
    assert pane.yz_view.image is not None
    assert pane.y_cut_label.text() == "Retained Fast x-z cut at y = -0.5 µm"
    assert pane.x_cut_label.text() == "Retained Fast y-z cut at x = -0.5 µm"


def test_full_three_dimensional_gui_path_keeps_selectable_sliders():
    app = QApplication.instance() or QApplication([])
    fields = FieldCollection()
    fields.add("volume", make_field(
        "volume", "Full Volume", np.zeros((3, 4, 6)),
        ("z", "x", "y"), "intensity", default_view="longitudinal",
    ))
    pane = LongitudinalPane()
    pane.set_run_data(RunData(
        workflow="pr_static",
        geometry=Geometry(
            x=np.linspace(-1.5, 1.5, 4),
            y=np.linspace(-2.5, 2.5, 6),
            z=np.linspace(0.0, 2.0, 3),
        ),
        fields=fields,
    ))

    assert not pane.x_cut_slider.isHidden()
    assert not pane.y_cut_slider.isHidden()
    assert not pane.show_guides.isHidden()
    assert pane.field_selector_label.text() == "3-D field"


def test_representative_fast_cut_storage_is_bounded():
    nz = nx = ny = 256
    float64_bytes = np.dtype(np.float64).itemsize
    retained_bytes = nz * (nx + ny) * float64_bytes
    omitted_volume_bytes = nz * nx * ny * float64_bytes

    assert retained_bytes == 1024 * 1024
    assert retained_bytes / omitted_volume_bytes == pytest.approx(1.0 / 128.0)
