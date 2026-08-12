from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.lc.source import (
    advance_slice_with_midpoint_source,
    director_driving_intensity,
)
from lcprop.optics.launch import LaunchResult, build_launch
from lcprop.optics.splitstep import linear_kernel, total_intensity


def test_shared_beam_and_launch_models_have_no_material_source_weight():
    beam_fields = {item.name for item in fields(BeamChannel)}
    launch_fields = {item.name for item in fields(LaunchResult)}

    assert "theta_weight" not in beam_fields
    assert "theta_weights" not in launch_fields
    assert not any(
        "material_weight" in name or "source_weight" in name
        for name in beam_fields
    )


def test_lc_source_functions_have_lc_module_provenance():
    assert director_driving_intensity.__module__ == "lcprop.lc.source"
    assert advance_slice_with_midpoint_source.__module__ == "lcprop.lc.source"


def test_lc_source_matches_historical_unit_weight_intensity():
    A = np.asarray(
        [
            [[1.0 + 1.0j, 2.0 - 0.5j]],
            [[0.5 - 0.25j, -1.0 + 2.0j]],
            [[3.0 + 0.5j, 0.25 + 0.75j]],
        ]
    )
    groups = ("A", "A", "B")

    actual = director_driving_intensity(A, coherence_groups=groups)
    expected = total_intensity(A, coherence_groups=groups)

    np.testing.assert_array_equal(actual, expected)


def test_lc_midpoint_source_shapes_and_definition():
    grid = make_grid(GridSpec(Nx=32, Ny=32, z_length_um=50.0))
    launch = build_launch(
        BeamStack(channels=(BeamChannel(power_mW=1.0),)),
        grid,
    )
    theta = np.zeros((32, 32), dtype=np.float32)
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
    )

    A, I_before, I_after, I_mid = advance_slice_with_midpoint_source(
        launch.A0.copy(),
        theta,
        kernel=kernel,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
    )

    assert A.shape == (1, 32, 32)
    assert I_before.shape == I_after.shape == I_mid.shape == (32, 32)
    np.testing.assert_allclose(I_mid, 0.5 * (I_before + I_after))


def test_shared_optics_does_not_import_lc_source_logic():
    path = Path(__file__).resolve().parents[1] / "src/lcprop/optics/splitstep.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "lcprop.lc.source" not in imported
