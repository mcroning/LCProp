"""Ownership and compatibility checks for material GUI applications."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    (
        ("lcprop.gui.main_window", "lcprop.lc.gui.main_window"),
        ("lcprop.gui.retained_results", "lcprop.lc.gui.retained_results"),
        (
            "lcprop.gui.panels.experiment_panel",
            "lcprop.lc.gui.panels.experiment_panel",
        ),
        (
            "lcprop.gui.panels.physics_panel",
            "lcprop.lc.gui.panels.physics_panel",
        ),
        (
            "lcprop.gui.panels.solver_panel",
            "lcprop.lc.gui.panels.solver_panel",
        ),
        (
            "lcprop.gui.panels.sweep_panel",
            "lcprop.lc.gui.panels.sweep_panel",
        ),
    ),
)
def test_legacy_lc_gui_modules_alias_canonical_modules(
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)

    assert legacy is canonical
    assert legacy.__name__ == canonical_name


def test_material_application_objects_report_canonical_provenance() -> None:
    from lcprop.lc.gui.app import main as lc_main
    from lcprop.lc.gui.main_window import LCPropMainWindow
    from lcprop.lc.gui.panels import (
        ExperimentPanel,
        PhysicsPanel,
        SolverPanel,
        SweepPanel,
    )
    from lcprop.lc.gui.retained_results import RetainedResults
    from lcprop.pr.gui.app import main as pr_main
    from lcprop.pr.gui.main_window import PRMainWindow

    assert lc_main.__module__ == "lcprop.lc.gui.app"
    assert LCPropMainWindow.__module__ == "lcprop.lc.gui.main_window"
    assert RetainedResults.__module__ == "lcprop.lc.gui.retained_results"
    for panel in (ExperimentPanel, PhysicsPanel, SolverPanel, SweepPanel):
        assert panel.__module__.startswith("lcprop.lc.gui.panels.")

    assert pr_main.__module__ == "lcprop.pr.gui.app"
    assert PRMainWindow.__module__ == "lcprop.pr.gui.main_window"


def test_legacy_lc_app_delegates_to_canonical_entry_point() -> None:
    from lcprop.gui.app import main as legacy_main
    from lcprop.lc.gui.app import main as canonical_main

    assert legacy_main is canonical_main


def test_shared_gui_framework_import_does_not_import_material_packages() -> None:
    script = """
import sys
import lcprop.gui.workers
import lcprop.gui.workspace
import lcprop.gui.views
import lcprop.gui.panels.beam_panel
import lcprop.gui.panels.grid_panel
import lcprop.gui.panels.results_panel
assert 'lcprop.lc' not in sys.modules
assert 'lcprop.pr' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_historical_panel_package_exports_preserve_identity() -> None:
    from lcprop.gui import panels as legacy_panels
    from lcprop.lc.gui import panels as canonical_panels

    assert legacy_panels.ExperimentPanel is canonical_panels.ExperimentPanel
    assert legacy_panels.PhysicsPanel is canonical_panels.PhysicsPanel
    assert legacy_panels.SolverPanel is canonical_panels.SolverPanel
    assert legacy_panels.SweepPanel is canonical_panels.SweepPanel
