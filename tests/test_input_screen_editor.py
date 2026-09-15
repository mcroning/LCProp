from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.gui.image_sources import decode_user_raster
from lcprop.gui.panels.beam_panel import BeamPanel
from lcprop.gui.panels.input_screen_editor import (
    INTENSITY_IMAGE_SCREEN,
    NO_SCREEN,
)
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.optics.launch import build_launch
from lcprop.optics.screens import RasterSource, prepare_intensity_raster_screen
from lcprop.pr.gui.main_window import PRMainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _source(values, *, name="screen.png") -> RasterSource:
    return RasterSource.from_array(np.asarray(values, dtype=float), display_name=name)


def _write_image(path, *, width=11, height=5):
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    for y in range(height):
        for x in range(width):
            image.setPixelColor(
                x,
                y,
                QColor((23 * x) % 256, (41 * y) % 256, (17 * (x + y)) % 256, 3),
            )
    assert image.save(str(path), "PNG")


def _enabled_panel(*, Nx=24, Ny=12, x_aperture=24.0, y_aperture=12.0):
    return BeamPanel(
        x_aperture_um=x_aperture,
        y_aperture_um=y_aperture,
        preview_Nx=Nx,
        preview_Ny=Ny,
        input_screens_enabled=True,
    )


def test_lc_remains_disabled_while_ordinary_pr_host_is_enabled(app):
    lc_window = LCPropMainWindow()
    pr_window = PRMainWindow()

    assert not lc_window.beam_panel.input_screen_editor.editor_enabled
    assert "LC requests do not yet carry" in (
        lc_window.beam_panel.input_screen_editor.availability.text()
    )
    assert pr_window.beam_panel.input_screen_editor.editor_enabled
    with pytest.raises(ValueError, match="LC requests do not yet carry"):
        lc_window.beam_panel.launch_elements()
    assert pr_window.beam_panel.launch_elements() == ()

    lc_window.close()
    pr_window.close()


def test_none_and_absorbing_image_emit_declarative_plan_and_actual_power(app):
    panel = _enabled_panel()
    panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(name="first", power_mW=3.0),
                BeamDefinition(name="disabled", power_mW=8.0, enabled=False),
                BeamDefinition(name="selected", power_mW=1.0),
            )
        )
    )
    editor = panel.input_screen_editor

    editor.channel.setCurrentIndex(1)
    assert editor.channel.currentText() == "2: selected"
    assert panel.launch_elements() == ()
    assert float(editor.incident_power.text()) == pytest.approx(1.0)
    assert float(editor.transmitted_power.text()) == pytest.approx(1.0)
    assert float(editor.throughput.text()) == pytest.approx(1.0)

    source = np.ones((6, 10), dtype=float)
    source[1:5, 3:7] = 0.0
    editor.set_source(_source(source))
    editor.width_um.setValue(8.0)
    editor.height_um.setValue(4.0)
    assignments = panel.launch_elements()
    beams, configuration = panel.launch_configuration()
    grid = panel._screen_preview_grid()
    incident = build_launch(beams, grid, complex_dtype=np.complex128)
    transformed = build_launch(
        beams,
        grid,
        complex_dtype=np.complex128,
        launch_elements=configuration,
    )

    assert assignments == configuration
    assert len(assignments) == 1 and assignments[0].channel_index == 1
    assert np.array_equal(transformed.A0[0], incident.A0[0])
    assert not np.array_equal(transformed.A0[1], incident.A0[1])
    expected = float(transformed.channel_throughput_fractions[1])
    assert 0.0 < expected < 1.0
    assert float(editor.throughput.text()) == pytest.approx(expected, rel=2e-8)
    assert not editor.transmission_preview.pixmap().isNull()
    assert not editor.transformed_preview.pixmap().isNull()
    panel.close()


def test_add_replace_remove_and_explicit_rectangular_placement(app):
    panel = _enabled_panel(Nx=30, Ny=14, x_aperture=30.0, y_aperture=14.0)
    editor = panel.input_screen_editor
    editor.set_source(_source([[0.0, 1.0], [1.0, 0.0]], name="first"))
    editor.width_um.setValue(10.0)
    editor.height_um.setValue(4.0)
    editor.center_x_um.setValue(-2.0)
    editor.center_y_um.setValue(1.0)
    first = panel.launch_elements()[0].elements[0]

    assert first.placement.width_um == 10.0
    assert first.placement.height_um == 4.0
    assert first.placement.center_x_um == -2.0
    assert first.placement.center_y_um == 1.0
    assert first.placement.boundary_policy == "reject"
    assert first.placement.outside_intensity_transmission == 1.0

    editor.set_source(_source([[1.0, 0.2], [0.4, 1.0]], name="replacement"))
    replacement = panel.launch_elements()[0].elements[0]
    assert replacement.source.sha256 != first.source.sha256

    editor.screen_type.setCurrentIndex(editor.screen_type.findData(NO_SCREEN))
    assert panel.launch_elements() == ()
    assert float(editor.throughput.text()) == pytest.approx(1.0)
    panel.close()


def test_binding_tracks_reordered_enabled_beam_and_drops_disabled_target(app):
    panel = _enabled_panel()
    first = BeamDefinition(name="first", power_mW=2.0)
    target = BeamDefinition(name="target", power_mW=1.0)
    disabled = BeamDefinition(name="off", enabled=False)
    panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(first, disabled, target))
    )
    editor = panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.set_source(_source([[0.0, 1.0], [1.0, 0.0]]))
    assert panel.launch_elements()[0].channel_index == 1

    panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(target, disabled, first))
    )
    assert panel.launch_elements()[0].channel_index == 0
    assert editor.channel.currentText() == "1: target"

    edited_target = replace(target, power_mW=1.5, waist_x_um=4.0)
    edited_stack = BeamStackDefinition(beams=(edited_target, disabled, first))
    panel.launch_plane_widget.set_beam_stack(edited_stack, selected_index=0)
    panel.launch_plane_widget.beamStackChanged.emit(edited_stack)
    assert panel.launch_elements()[0].channel_index == 0
    assert float(editor.incident_power.text()) == pytest.approx(1.5)

    panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(replace(edited_target, enabled=False), first))
    )
    assert panel.launch_elements() == ()
    panel.close()


def test_user_image_decode_preview_and_large_source_do_not_select_grid(
    app,
    tmp_path,
):
    path = tmp_path / "user.png"
    _write_image(path)
    decoded = decode_user_raster(path)
    assert decoded.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert decoded.source.grayscale.shape == (5, 11)

    panel = _enabled_panel(Nx=20, Ny=8, x_aperture=20.0, y_aperture=8.0)
    editor = panel.input_screen_editor
    loaded = editor.load_user_image(path)
    assert loaded == decoded.source
    assert editor.selected_source.text() == path.name
    assert panel._screen_preview_grid().Nx == 20
    assert panel._screen_preview_grid().Ny == 8

    large = RasterSource(
        source_kind="array",
        display_name="large",
        basename="<array>",
        sha256="a" * 64,
        width=1200,
        height=400,
        encoded_format="array",
        decoded_mode="L",
        grayscale=np.ones((400, 1200), dtype=np.uint8),
    )
    editor.set_source(large)
    screen = panel.launch_elements()[0].elements[0]
    prepared = prepare_intensity_raster_screen(screen, panel._screen_preview_grid())
    assert screen.source.grayscale.shape == (400, 1200)
    assert prepared.shape == (20, 8)
    assert panel._screen_preview_grid().spec.x_aperture_um == 20.0
    assert panel._screen_preview_grid().spec.y_aperture_um == 8.0
    panel.close()


def test_empty_standard_catalog_and_out_of_aperture_error_are_actionable(app):
    panel = _enabled_panel()
    editor = panel.input_screen_editor

    assert editor.standard_source.count() == 0
    assert not editor.standard_source.isEnabled()
    assert "No approved packaged images" in editor.standard_status.text()

    editor.screen_type.setCurrentIndex(
        editor.screen_type.findData(INTENSITY_IMAGE_SCREEN)
    )
    with pytest.raises(ValueError, match="choose an image source"):
        panel.launch_elements()
    assert "choose an image source" in editor.status.text()

    editor.set_source(_source(np.eye(4)))
    editor.width_um.setValue(20.0)
    editor.center_x_um.setValue(11.0)
    with pytest.raises(ValueError, match="footprint extends outside"):
        panel.launch_elements()
    assert "footprint extends outside" in editor.status.text()
    panel.close()


def test_preview_uses_optical_launch_only_and_never_material_propagation(
    app,
    monkeypatch,
):
    import lcprop.gui.panels.input_screen_editor as editor_module

    calls = []
    original = editor_module.build_launch

    def observed(*args, **kwargs):
        calls.append(
            (kwargs.get("launch_elements", ()), kwargs.get("context"))
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(editor_module, "build_launch", observed)
    panel = _enabled_panel()
    panel.input_screen_editor.set_source(_source(np.eye(4)))
    panel.input_screen_editor.refresh_preview()

    assert calls
    assert all(isinstance(assignments, tuple) for assignments, _ in calls)
    assert all(context is not None for _, context in calls)
    panel.close()


def test_focused_beam_input_screen_preview_uses_material_neutral_context(app):
    panel = _enabled_panel()
    panel.set_optical_context(n_ref=2.3, interaction_length_um=80.0)
    panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="focused",
                    profile="focused_gaussian",
                    waist_x_at_focus_um=4.0,
                    waist_y_at_focus_um=7.0,
                    focus_z_um=-20.0,
                ),
            )
        )
    )
    editor = panel.input_screen_editor
    editor.set_source(_source(np.eye(4)))

    editor.refresh_preview()

    assert editor.status.text() == ""
    assert not editor.transmission_preview.pixmap().isNull()
    assert not editor.transformed_preview.pixmap().isNull()
    panel.close()


def test_new_screen_centers_on_selected_beam_and_remains_independent(app):
    first = BeamDefinition(name="first", x_um=4.0, y_um=-2.0)
    second = BeamDefinition(name="second", x_um=-5.0, y_um=3.0)
    panel = _enabled_panel(x_aperture=30.0, y_aperture=20.0)
    panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(first, second))
    )
    editor = panel.input_screen_editor

    editor.channel.setCurrentIndex(0)
    first_source = _source(np.eye(4), name="first.png")
    editor.set_source(first_source)
    assert editor.center_x_um.value() == pytest.approx(4.0)
    assert editor.center_y_um.value() == pytest.approx(-2.0)

    editor.center_x_um.setValue(1.25)
    editor.center_y_um.setValue(-0.75)
    moved_first = replace(first, x_um=7.0, y_um=1.5)
    panel.set_beam_stack_definition(
        BeamStackDefinition(beams=(moved_first, second))
    )
    assert editor.center_x_um.value() == pytest.approx(1.25)
    assert editor.center_y_um.value() == pytest.approx(-0.75)

    width_before = editor.width_um.value()
    height_before = editor.height_um.value()
    editor.center_on_beam.click()
    assert editor.center_x_um.value() == pytest.approx(7.0)
    assert editor.center_y_um.value() == pytest.approx(1.5)
    assert editor.width_um.value() == width_before
    assert editor.height_um.value() == height_before
    assert panel.launch_elements()[0].elements[0].source == first_source

    editor.channel.setCurrentIndex(1)
    editor.set_source(_source(np.fliplr(np.eye(4)), name="second.png"))
    assert editor.center_x_um.value() == pytest.approx(-5.0)
    assert editor.center_y_um.value() == pytest.approx(3.0)
    assignments = panel.launch_elements()
    assert tuple(item.channel_index for item in assignments) == (0, 1)
    panel.close()


def test_compact_preview_selector_and_none_state_preserve_screen_evidence(app):
    panel = _enabled_panel()
    editor = panel.input_screen_editor

    assert editor.screen_type.currentData() == NO_SCREEN
    assert editor.preview_stack.isHidden()
    assert editor.selected_source.isHidden()

    source = _source(np.eye(4), name="visible-source.png")
    editor.set_source(source)
    assert editor.preview_mode.count() == 2
    assert editor.preview_mode.itemData(0) == "transmission"
    assert editor.preview_mode.itemData(1) == "post_screen"
    assert not editor.preview_stack.isHidden()
    assert not editor.selected_source.isHidden()
    assert editor.selected_source.text() == "<array>"
    assert not editor.transmission_preview.pixmap().isNull()
    assert not editor.transformed_preview.pixmap().isNull()
    assert editor.incident_power.text() != "—"
    assert editor.transmitted_power.text() != "—"
    assert editor.throughput.text() != "—"

    editor.preview_mode.setCurrentIndex(1)
    assert editor.preview_stack.currentWidget() is editor.transformed_preview
    editor.preview_mode.setCurrentIndex(0)
    assert editor.preview_stack.currentWidget() is editor.transmission_preview

    editor.screen_type.setCurrentIndex(editor.screen_type.findData(NO_SCREEN))
    assert editor.preview_stack.isHidden()
    assert editor.selected_source.isHidden()
    panel.close()
