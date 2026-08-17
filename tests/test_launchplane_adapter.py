from __future__ import annotations

import builtins
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lcprop.adapters import (
    beam_definition_to_channel,
    beam_stack_definition_to_lcprop,
    beam_stack_to_launchplane,
)
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.optics.splitstep import hop_linear, linear_kernel, total_intensity


@pytest.fixture
def launchplane_model():
    return pytest.importorskip(
        "launchplane.model",
        reason="the optional launchplane package is not installed",
    )


def test_one_enabled_beam_maps_field_for_field(launchplane_model):
    beam = launchplane_model.BeamDefinition(
        name="probe",
        wavelength_um=0.532,
        power_mW=2.5,
        x_um=1.25,
        y_um=-3.5,
        waist_x_um=4.0,
        waist_y_um=5.0,
        tilt_x_rad_per_um=0.01,
        tilt_y_rad_per_um=-0.02,
        phase_rad=0.75,
        coherence_group="laser-green",
        enabled=True,
    )

    channel = beam_definition_to_channel(beam)

    assert channel.name == beam.name
    assert channel.wavelength_um == beam.wavelength_um
    assert channel.power_mW == beam.power_mW
    assert channel.x0_um == beam.x_um
    assert channel.y0_um == beam.y_um
    assert channel.waist_x_um == beam.waist_x_um
    assert channel.waist_y_um == beam.waist_y_um
    assert channel.tilt_x_rad_per_um == beam.tilt_x_rad_per_um
    assert channel.tilt_y_rad_per_um == beam.tilt_y_rad_per_um
    assert channel.phase_rad == beam.phase_rad
    assert channel.coherence_group == beam.coherence_group


def test_external_air_angle_transfers_resolved_wavevector_exactly(
    launchplane_model,
):
    beam = launchplane_model.BeamDefinition.from_launch_angles(
        wavelength_um=0.633,
        angle_x_rad=0.1,
        angle_y_rad=0.0,
        launch_medium_index=1.0,
    )

    channel = beam_definition_to_channel(beam)

    assert beam.tilt_x_rad_per_um == pytest.approx(
        0.990950800381,
        abs=5e-13,
    )
    assert channel.tilt_x_rad_per_um == beam.transverse_wavevector_x_rad_per_um
    assert channel.tilt_y_rad_per_um == beam.transverse_wavevector_y_rad_per_um


def test_legacy_phase_slope_remains_phase_slope(launchplane_model):
    beam = launchplane_model.BeamDefinition(tilt_x_rad_per_um=0.1)

    channel = beam_definition_to_channel(beam)

    assert beam.launch_input_mode == "transverse_wavevector"
    assert beam.launch_medium_index is None
    assert channel.tilt_x_rad_per_um == 0.1


def test_lcprop_round_trip_preserves_wavevector_without_air_provenance():
    stack = BeamStack(
        channels=(
            BeamChannel(
                name="off-axis",
                tilt_x_rad_per_um=0.73,
                tilt_y_rad_per_um=-0.41,
                coherence_group="laser",
            ),
        )
    )

    definition = beam_stack_to_launchplane(stack)
    rebuilt = beam_stack_definition_to_lcprop(definition)
    beam = definition.beams[0]

    assert beam.launch_input_mode == "transverse_wavevector"
    assert beam.launch_medium_index is None
    assert rebuilt.channels[0].tilt_x_rad_per_um == 0.73
    assert rebuilt.channels[0].tilt_y_rad_per_um == -0.41


@pytest.mark.parametrize(
    ("angle_x_rad", "angle_y_rad", "axis"),
    ((0.04, 0.0, 0), (0.0, 0.04, 1)),
)
def test_external_angle_phase_gradient_and_paraxial_centroid_slope(
    launchplane_model,
    angle_x_rad,
    angle_y_rad,
    axis,
):
    wavelength_um = 0.633
    z_um = 20.0
    n_ref = 1.5
    beam = launchplane_model.BeamDefinition.from_launch_angles(
        wavelength_um=wavelength_um,
        waist_x_um=10.0,
        waist_y_um=10.0,
        angle_x_rad=angle_x_rad,
        angle_y_rad=angle_y_rad,
        launch_medium_index=1.0,
    )
    channel = beam_definition_to_channel(beam)
    grid = make_grid(
        GridSpec(
            Nx=256,
            Ny=256,
            x_aperture_um=128.0,
            y_aperture_um=128.0,
            z_length_um=z_um,
            dz_um=z_um,
        ),
        real_dtype=np.float64,
    )
    field = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
    ).A0
    q = (
        channel.tilt_x_rad_per_um
        if axis == 0
        else channel.tilt_y_rad_per_um
    )
    scalar_field = field[0]
    if axis == 0:
        product = scalar_field[1:, :] * np.conj(scalar_field[:-1, :])
        weight = np.minimum(
            np.abs(scalar_field[1:, :]),
            np.abs(scalar_field[:-1, :]),
        )
        spacing = grid.dx_um
        coordinate = grid.x_um
    else:
        product = scalar_field[:, 1:] * np.conj(scalar_field[:, :-1])
        weight = np.minimum(
            np.abs(scalar_field[:, 1:]),
            np.abs(scalar_field[:, :-1]),
        )
        spacing = grid.dy_um
        coordinate = grid.y_um
    mask = weight > 0.05 * np.max(np.abs(scalar_field))
    measured_q = np.median(np.angle(product[mask]) / spacing)

    def centroid(A):
        intensity = np.abs(A[0]) ** 2
        marginal = intensity.sum(axis=1 if axis == 0 else 0)
        return float(np.sum(marginal * coordinate) / np.sum(marginal))

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=z_um,
        wavelength=wavelength_um,
        n_ref=n_ref,
    )
    propagated = hop_linear(field, kernel)
    k_ref = 2.0 * np.pi * n_ref / wavelength_um

    assert q > 0.0
    assert measured_q == pytest.approx(q, abs=2e-15)
    assert centroid(propagated) - centroid(field) == pytest.approx(
        z_um * q / k_ref,
        abs=2e-8,
    )


def test_x_and_y_coordinates_are_not_swapped(launchplane_model):
    beam = launchplane_model.BeamDefinition(x_um=12.0, y_um=-7.0)

    channel = beam_definition_to_channel(beam)

    assert (channel.x0_um, channel.y0_um) == (12.0, -7.0)


def test_multiple_enabled_beams_preserve_order(launchplane_model):
    stack = launchplane_model.BeamStackDefinition(
        beams=tuple(
            launchplane_model.BeamDefinition(name=name)
            for name in ("third", "first", "second")
        )
    )

    converted = beam_stack_definition_to_lcprop(stack)

    assert tuple(channel.name for channel in converted.channels) == (
        "third",
        "first",
        "second",
    )


def test_coherence_groups_are_preserved_exactly(launchplane_model):
    groups = ("A", "A", " laser B ")
    stack = launchplane_model.BeamStackDefinition(
        beams=tuple(
            launchplane_model.BeamDefinition(
                name=f"beam-{index}",
                coherence_group=group,
            )
            for index, group in enumerate(groups)
        )
    )

    converted = beam_stack_definition_to_lcprop(stack)

    assert converted.coherence_groups == groups


def test_disabled_beams_are_filtered_without_zero_power_channels(launchplane_model):
    stack = launchplane_model.BeamStackDefinition(
        beams=(
            launchplane_model.BeamDefinition(
                name="disabled",
                power_mW=9.0,
                enabled=False,
            ),
            launchplane_model.BeamDefinition(
                name="enabled",
                power_mW=2.0,
                enabled=True,
            ),
        )
    )

    converted = beam_stack_definition_to_lcprop(stack)

    assert tuple(channel.name for channel in converted.channels) == ("enabled",)
    assert converted.channels[0].power_mW == 2.0


def test_all_disabled_stack_is_rejected(launchplane_model):
    stack = launchplane_model.BeamStackDefinition(
        beams=(
            launchplane_model.BeamDefinition(name="one", enabled=False),
            launchplane_model.BeamDefinition(name="two", enabled=False),
        )
    )

    with pytest.raises(
        ValueError,
        match="^LaunchPane beam stack contains no enabled beams$",
    ):
        beam_stack_definition_to_lcprop(stack)


def test_removed_theta_weight_unit_compatibility_is_accepted(launchplane_model):
    beam = launchplane_model.BeamDefinition()

    channel = beam_definition_to_channel(beam, theta_weight=1.0)

    assert not hasattr(channel, "theta_weight")


def test_removed_theta_weight_unit_stack_compatibility_is_accepted(
    launchplane_model,
):
    stack = launchplane_model.BeamStackDefinition(
        beams=(
            launchplane_model.BeamDefinition(name="one"),
            launchplane_model.BeamDefinition(name="two"),
        )
    )

    converted = beam_stack_definition_to_lcprop(stack, theta_weight=1.0)

    assert all(not hasattr(channel, "theta_weight") for channel in converted.channels)


@pytest.mark.parametrize("theta_weight", [0.0, -1.0, 2.75, "invalid"])
def test_nonunit_legacy_theta_weight_is_rejected_for_one_beam(
    launchplane_model,
    theta_weight,
):
    beam = launchplane_model.BeamDefinition()

    with pytest.raises(ValueError, match="legacy non-unit theta_weight"):
        beam_definition_to_channel(beam, theta_weight=theta_weight)


@pytest.mark.parametrize("theta_weight", [0.0, -1.0, 2.75, "invalid"])
def test_nonunit_legacy_theta_weight_is_rejected_for_stack(
    launchplane_model,
    theta_weight,
):
    stack = launchplane_model.BeamStackDefinition(
        beams=(launchplane_model.BeamDefinition(),)
    )

    with pytest.raises(ValueError, match="legacy non-unit theta_weight"):
        beam_stack_definition_to_lcprop(stack, theta_weight=theta_weight)


def test_mixed_grouping_uses_lcprop_grouped_total_intensity(launchplane_model):
    stack = launchplane_model.BeamStackDefinition(
        beams=tuple(
            launchplane_model.BeamDefinition(
                name=f"beam-{index}",
                coherence_group=group,
            )
            for index, group in enumerate(("A", "A", "B"))
        )
    )
    converted = beam_stack_definition_to_lcprop(stack)
    fields = np.array([1.0 + 1.0j, 2.0 - 1.0j, -0.5 + 0.25j])[:, None, None]

    actual = total_intensity(
        fields,
        coherence_groups=converted.coherence_groups,
    )
    expected = np.abs(fields[0] + fields[1]) ** 2 + np.abs(fields[2]) ** 2

    assert np.allclose(actual, expected)


def test_normal_lcprop_imports_do_not_require_launchplane():
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
        raise AssertionError("LCProp import attempted to import LaunchPane")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_launchplane

import lcprop.core.beams
import lcprop.core.context
import lcprop.adapters
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr


def test_calling_adapter_without_launchplane_has_clear_error(monkeypatch):
    original_import = builtins.__import__

    def reject_launchplane(name, *args, **kwargs):
        if name == "launchplane" or name.startswith("launchplane."):
            raise ModuleNotFoundError("No module named 'launchplane'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_launchplane)

    with pytest.raises(
        ImportError,
        match="the separate 'launchplane' package is required",
    ):
        beam_definition_to_channel(object())
