"""Physical [x, y] parity across Qt previews, PR launch and Results."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.core.backend import BackendSpec
from lcprop.gui.image_sources import decode_user_raster
from lcprop.gui.panels.beam_panel import BeamPanel
from lcprop.gui.panels import input_screen_editor as preview_module
from lcprop.gui.views.image_pane import ImagePane
from lcprop.optics.launch import build_launch
from lcprop.optics.screens import prepare_intensity_raster_screen
from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.workflow import run_pr_timedependent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _gray_pixels(pixmap):
    image = pixmap.toImage()
    return np.array([
        [image.pixelColor(column, row).red() for column in range(image.width())]
        for row in range(image.height())
    ])


@pytest.mark.parametrize("xy_axes", [True, False])
def test_qt_adapter_has_explicit_axis_and_vertical_direction(app, xy_axes):
    # Unequal dimensions and unique samples distinguish every reflection,
    # transpose and quarter-turn, including the old y-inverted preview.
    values = np.arange(24, dtype=float).reshape(6, 4)
    original = values.copy()
    size = QSize(6, 4) if xy_axes else QSize(4, 6)
    pixels = _gray_pixels(preview_module._array_pixmap(
        values, xy_axes=xy_axes, size=size,
    ))
    for i in range(6):
        for j in range(4):
            row, column = (3 - j, i) if xy_axes else (i, j)
            assert pixels[row, column] == round(255 * values[i, j] / 23)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("profile", ["uniform", "legacy_gaussian"])
@pytest.mark.parametrize("invert", [False, True])
def test_asymmetric_raster_preview_launch_results_and_volume_parity(
    app, tmp_path, monkeypatch, invert, profile,
):
    # Non-square source and physical footprint. The four unequal markers and
    # asymmetric interior distinguish every orientation, unlike a checkerboard.
    raster = np.zeros((6, 10), dtype=np.uint8)
    markers = {(1, 1): 51, (1, 8): 102, (4, 1): 153, (4, 8): 204,
               (2, 3): 255, (3, 3): 230, (3, 4): 179}
    image = QImage(10, 6, QImage.Format.Format_RGB32)
    for (row, column), value in markers.items():
        raster[row, column] = value
    for row in range(6):
        for column in range(10):
            value = int(raster[row, column])
            image.setPixelColor(column, row, QColor(value, value, value))
    path = tmp_path / "asymmetric.png"
    assert image.save(str(path))
    decoded = decode_user_raster(path)
    np.testing.assert_array_equal(decoded.source.grayscale, raster)

    panel = BeamPanel(x_aperture_um=24.0, y_aperture_um=32.0,
                      preview_Nx=24, preview_Ny=16, input_screens_enabled=True)
    pane = ImagePane()
    try:
        panel.set_beam_stack_definition(BeamStackDefinition(beams=(
            BeamDefinition(name="calibration", profile=profile, power_mW=3.0,
                           x_um=2.0, y_um=1.5,
                           waist_x_um=3.0, waist_y_um=5.0),
        )))
        editor = panel.input_screen_editor
        editor.set_source(decoded.source)
        editor.width_um.setValue(10.0)
        editor.height_um.setValue(20.0)
        editor.center_x_um.setValue(0.5)
        editor.center_y_um.setValue(1.0)
        # Exercise production decoding, screen placement and preview dispatch.
        elements = panel.launch_elements()
        screen = replace(elements[0].elements[0], invert=invert)
        editor.set_launch_elements((replace(elements[0], elements=(screen,)),))
        beams, assignments = panel.launch_configuration()
        grid = panel._screen_preview_grid()
        transmission = prepare_intensity_raster_screen(screen, grid)
        launch = build_launch(beams, grid, complex_dtype=np.complex128,
                              launch_elements=assignments)
        intensity = np.abs(launch.A0[0]) ** 2
        incident = build_launch(beams, grid, complex_dtype=np.complex128)
        incident_intensity = np.abs(incident.A0[0]) ** 2
        # Independent index-by-index oracle: pad the raster, then place its
        # columns left-to-right and rows top-to-bottom in physical coordinates.
        padded = np.ones((10, 10))
        padded[2:8, :] = 1 - raster / 255 if invert else raster / 255
        expected = np.ones((24, 16))
        old_transmission = np.ones((24, 16))
        for row in range(10):
            for column in range(10):
                expected[7 + column, 12 - row] = padded[row, column]
                old_transmission[16 - column, 3 + row] = padded[row, column]
        np.testing.assert_array_equal(transmission, expected)
        np.testing.assert_allclose(launch.A0[0], incident.A0[0] * np.sqrt(expected),
                                   rtol=0, atol=0)

        # Original source top-left stays physical top-left, without an
        # additional beam-frame transformation or a compensating display turn.
        for (row, column), value in markers.items():
            ix, iy = 7 + column, 10 - row
            expected_t = 1.0 - value / 255 if invert else value / 255
            assert grid.x_um[ix] == pytest.approx(column - 4.5)
            assert grid.y_um[iy] == pytest.approx(5 - 2 * row)
            assert transmission[ix, iy] == pytest.approx(expected_t)
            assert intensity[ix, iy] == pytest.approx(incident_intensity[ix, iy] * expected_t)

        # Capture actual preview inputs while retaining the real Qt adapter.
        captured = []
        adapter = preview_module._array_pixmap
        def capture(values, *, xy_axes, size):
            captured.append(np.array(values, copy=True))
            # Native pixel dimensions prevent smoothing from hiding a flip.
            return adapter(values, xy_axes=xy_axes, size=QSize(grid.Nx, grid.Ny))
        monkeypatch.setattr(preview_module, "_array_pixmap", capture)
        editor.refresh_preview()
        np.testing.assert_array_equal(captured[0], transmission)
        np.testing.assert_array_equal(captured[1], intensity)

        # Real tiny CPU workflow, not a synthetic result or mocked propagator.
        # Its first retained optical plane is the slice's endpoint-intensity
        # average (the material midpoint-source convention), not exactly z=0.
        # A short physical step makes diffraction changes bounded below 1e-8.
        request = PRRunRequest(
            grid=replace(grid.spec, z_length_um=1e-6, dz_um=1e-6),
            beams=beams, launch_elements=assignments,
            material=PRMaterialSpec(applied_field=0.0, gain_length_product=0.0,
                                    characteristic_wavenumber_per_um_override=0.1),
            solver=PRSolverOptions(Nt=0, dt_normalized=1e-5),
            backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
        )
        result = run_pr_timedependent(request)
        np.testing.assert_array_equal(result.A_initial, launch.A0)
        data = pr_result_to_run_data(result)
        field = data.fields["input_intensity"]
        assert field.axes == ("x", "y")
        np.testing.assert_array_equal(field.data, intensity)
        volume = data.fields["optical_intensity_stack"]
        assert volume.axes == ("z", "x", "y")
        np.testing.assert_allclose(
            volume.data[0], (intensity + np.abs(result.A_final[0]) ** 2) / 2,
            rtol=1e-12, atol=1e-16,
        )
        np.testing.assert_allclose(volume.data[0], intensity, rtol=0, atol=1e-8)

        pane.set_run_data(data)
        pane.field_selector.setCurrentIndex(pane.field_selector.findData("input_intensity"))
        artist = pane.image_view.image
        assert artist.origin == "lower"
        np.testing.assert_array_equal(artist.get_array(), intensity.T)
        assert pane.image_view.ax.get_ylim()[0] < pane.image_view.ax.get_ylim()[1]
        pixels = _gray_pixels(editor.transformed_preview.pixmap())
        scaled = np.rint(255 * (intensity - intensity.min()) /
                         (intensity.max() - intensity.min())).astype(np.uint8)
        for ix in range(grid.Nx):
            for iy in range(grid.Ny):
                assert pixels[grid.Ny - 1 - iy, ix] == scaled[ix, iy]
        # Preview updates and display adaptation cannot mutate scientific data
        # or alter incident/transmitted power and throughput.
        np.testing.assert_array_equal(np.abs(launch.A0[0]) ** 2, intensity)
        # Compare corrected and former physical overlaps with exactly the
        # same incident field. The entire power difference must equal the
        # independently integrated change in transmission, without renormalizing.
        area = grid.dx_um * grid.dy_um
        expected_throughput = float(np.sum(incident_intensity * expected) * area)
        old_throughput = float(np.sum(incident_intensity * old_transmission) * area)
        overlap_delta = float(np.sum(incident_intensity *
                                     (expected - old_transmission)) * area)
        measured = float(launch.post_element_physical_powers_mW[0])
        assert measured - 3 * old_throughput == pytest.approx(3 * overlap_delta, abs=1e-14)
        np.testing.assert_array_equal(launch.physical_powers_mW, incident.physical_powers_mW)
        assert np.sum(incident_intensity) * area == pytest.approx(1.0)
        if profile == "legacy_gaussian":
            assert abs(overlap_delta) > 1e-3
        else:
            assert overlap_delta == pytest.approx(0.0, abs=1e-15)
        assert float(editor.incident_power.text()) == pytest.approx(3.0)
        assert float(editor.transmitted_power.text()) == pytest.approx(
            3 * expected_throughput, rel=2e-8)
        assert float(editor.throughput.text()) == pytest.approx(expected_throughput, rel=2e-8)
    finally:
        pane.close()
        panel.close()
        pane.deleteLater()
        panel.deleteLater()
        app.processEvents()
