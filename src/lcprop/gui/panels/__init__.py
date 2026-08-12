"""Shared panels plus lazy compatibility exports for historical LC panels."""

from importlib import import_module

from lcprop.gui.panels.beam_panel import BeamPanel
from lcprop.gui.panels.grid_panel import GridPanel
from lcprop.gui.panels.results_panel import ResultsPanel


_LC_COMPATIBILITY_EXPORTS = {
    "ExperimentPanel": "experiment_panel",
    "PhysicsPanel": "physics_panel",
    "SolverPanel": "solver_panel",
    "SweepPanel": "sweep_panel",
}


def __getattr__(name: str):
    """Resolve historical LC panel exports from their canonical owner."""

    module_suffix = _LC_COMPATIBILITY_EXPORTS.get(name)
    if module_suffix is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"lcprop.lc.gui.panels.{module_suffix}"), name)
    globals()[name] = value
    return value

__all__ = [
    "ExperimentPanel",
    "PhysicsPanel",
    "BeamPanel",
    "GridPanel",
    "SolverPanel",
    "SweepPanel",
    "ResultsPanel",
]
