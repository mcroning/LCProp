from __future__ import annotations

import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PySide6.QtWidgets import QApplication
from launchplane.launchpane import LaunchPlaneWidget
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.gui.panels.beam_panel import BeamPanel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_beam_panel_embeds_launchplane_with_lcprop_defaults(app):
    panel = BeamPanel()

    assert isinstance(panel.launch_plane_widget, LaunchPlaneWidget)
    assert panel.layout().count() == 1
    beam = panel.beam_stack_definition.beams[0]
    assert beam.name == "beam"
    assert beam.wavelength_um == 0.633
    assert beam.power_mW == 1.0
    assert (beam.x_um, beam.y_um) == (0.0, 0.0)
    assert (beam.waist_x_um, beam.waist_y_um) == (3.0, 3.0)
    assert (beam.tilt_x_rad_per_um, beam.tilt_y_rad_per_um) == (0.0, 0.0)
    assert beam.phase_rad == 0.0
    assert beam.coherence_group == "laser_A"
    assert beam.enabled is True
    panel.close()


def test_beam_panel_beams_adapts_fields_without_axis_swap(app):
    panel = BeamPanel()
    panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="probe",
                    x_um=7.0,
                    y_um=-11.0,
                    waist_x_um=4.0,
                    waist_y_um=6.0,
                    coherence_group="probe-laser",
                ),
            )
        )
    )

    channel = panel.beams().channels[0]

    assert channel.name == "probe"
    assert (channel.x0_um, channel.y0_um) == (7.0, -11.0)
    assert (channel.waist_x_um, channel.waist_y_um) == (4.0, 6.0)
    assert channel.coherence_group == "probe-laser"
    panel.close()


def test_beam_panel_preserves_enabled_order_and_coherence_groups(app):
    panel = BeamPanel()
    panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(name="first", coherence_group="A"),
                BeamDefinition(
                    name="disabled",
                    coherence_group="unused",
                    enabled=False,
                ),
                BeamDefinition(name="second", coherence_group="A"),
                BeamDefinition(name="third", coherence_group="B"),
            )
        )
    )

    converted = panel.beams()

    assert tuple(channel.name for channel in converted.channels) == (
        "first",
        "second",
        "third",
    )
    assert converted.coherence_groups == ("A", "A", "B")
    panel.close()


def test_beam_panel_rejects_all_disabled_state(app):
    panel = BeamPanel()
    panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(BeamDefinition(name="off", enabled=False),)
        )
    )

    with pytest.raises(
        ValueError,
        match="^LaunchPane beam stack contains no enabled beams$",
    ):
        panel.beams()
    panel.close()


def test_grid_aperture_changes_propagate_but_resolution_does_not(app):
    window = LCPropMainWindow()

    assert window.beam_panel.launch_plane_definition.x_aperture_um == 75.0
    assert window.beam_panel.launch_plane_definition.y_aperture_um == 100.0

    window.grid_panel.x_aperture_um.setValue(48.0)
    window.grid_panel.y_aperture_um.setValue(62.0)
    app.processEvents()

    assert window.beam_panel.launch_plane_definition.x_aperture_um == 48.0
    assert window.beam_panel.launch_plane_definition.y_aperture_um == 62.0

    before = window.beam_panel.launch_plane_definition
    window.grid_panel.Nx.setValue(128)
    window.grid_panel.Ny.setValue(256)
    app.processEvents()

    assert window.beam_panel.launch_plane_definition == before
    window.close()


def test_aperture_update_preserves_beam_coordinates_waists_and_tilts(app):
    panel = BeamPanel()
    stack = BeamStackDefinition(
        beams=(
            BeamDefinition(
                name="outside",
                x_um=30.0,
                y_um=-40.0,
                waist_x_um=8.0,
                waist_y_um=9.0,
                tilt_x_rad_per_um=0.01,
                tilt_y_rad_per_um=-0.02,
            ),
        )
    )
    panel.set_beam_stack_definition(stack)

    panel.set_aperture(20.0, 24.0)

    assert panel.beam_stack_definition == stack
    assert panel.launch_plane_definition.x_aperture_um == 20.0
    assert panel.launch_plane_definition.y_aperture_um == 24.0
    assert panel.launch_plane_widget.x_spin.maximum() >= 30.0
    assert panel.launch_plane_widget.y_spin.minimum() <= -40.0
    panel.close()


def test_request_construction_uses_adapted_beam_stack(app):
    window = LCPropMainWindow()
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(name="one", power_mW=1.5, coherence_group="A"),
                BeamDefinition(name="off", enabled=False),
                BeamDefinition(name="two", power_mW=2.5, coherence_group="B"),
            )
        )
    )

    request = window.build_request()

    assert tuple(channel.name for channel in request.beams.channels) == (
        "one",
        "two",
    )
    assert request.beams.coherence_groups == ("A", "B")
    window.close()


def test_describe_request_reports_multibeam_summary(app):
    window = LCPropMainWindow()
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="one",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    coherence_group="A",
                ),
                BeamDefinition(
                    name="two",
                    wavelength_um=0.532,
                    power_mW=2.0,
                    coherence_group="A",
                ),
                BeamDefinition(
                    name="three",
                    wavelength_um=0.633,
                    power_mW=3.0,
                    coherence_group="B",
                ),
            )
        )
    )

    text = window.describe_request(window.build_request())

    assert "Beams: 3 enabled, total P=6 mW" in text
    assert "Wavelengths: 0.633 µm, 0.532 µm" in text
    assert "Lasers/coherence groups: A, B" in text
    assert "First enabled beam: one" in text
    window.close()


def test_gui_import_without_launchplane_has_actionable_error():
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), env.get("PYTHONPATH", ""))
    )
    script = """
import builtins

original_import = builtins.__import__

def reject_launchplane(name, *args, **kwargs):
    if name == "launchplane" or name.startswith("launchplane."):
        raise ModuleNotFoundError("No module named 'launchplane'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_launchplane

try:
    import lcprop.gui.main_window
except ImportError as exc:
    message = str(exc)
    assert "separate 'launchplane' package" in message
    assert "pip install -e" in message
else:
    raise AssertionError("LCProp GUI import succeeded without LaunchPane")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
