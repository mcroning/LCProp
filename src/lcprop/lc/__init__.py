"""Authoritative public ownership boundary for liquid-crystal components.

Canonical LC types and presentation products live under this package while
legacy import paths remain available as compatibility exports. Facades are
loaded lazily so historical modules can continue importing ``LC_MATERIAL_ID``
while they initialize.
"""

from __future__ import annotations

from importlib import import_module


LC_MATERIAL_ID = "lc"

_FACADE_MODULES = {
    "BiasSpec": "lcprop.lc.specs",
    "CouplingMode": "lcprop.lc.requests",
    "ExistenceSolver": "lcprop.lc.requests",
    "LCContext": "lcprop.lc.specs",
    "LCMaterial": "lcprop.lc.specs",
    "LCSpatialNormalization": "lcprop.lc.normalization",
    "LC_STATIC_OPERATION": "lcprop.lc.operations",
    "LC_TIMEDEPENDENT_OPERATION": "lcprop.lc.operations",
    "OpticsSolver": "lcprop.lc.requests",
    "OutputOptions": "lcprop.lc.requests",
    "ParameterSweepRequest": "lcprop.lc.requests",
    "ParameterSweepResult": "lcprop.lc.results",
    "Precision": "lcprop.lc.requests",
    "RuntimeOptions": "lcprop.lc.requests",
    "STATIC_CHECKPOINT_SCHEMA_VERSION": "lcprop.lc.persistence",
    "SolitonExistenceRequest": "lcprop.lc.requests",
    "SolitonExistenceResult": "lcprop.lc.results",
    "SolitonRequest": "lcprop.lc.requests",
    "SolitonResult": "lcprop.lc.results",
    "SolitonSweepMember": "lcprop.lc.results",
    "StaticCheckpoint": "lcprop.lc.persistence",
    "StaticIterationRecord": "lcprop.lc.results",
    "StaticRunRequest": "lcprop.lc.requests",
    "StaticRunResult": "lcprop.lc.results",
    "StaticSliceSummary": "lcprop.lc.results",
    "StaticSolverOptions": "lcprop.lc.requests",
    "StaticStrategy": "lcprop.lc.requests",
    "StaticTorqueBalanceData": "lcprop.lc.static_torque_balance",
    "StaticWorkflowOptions": "lcprop.lc.requests",
    "SweepExecution": "lcprop.lc.requests",
    "SweepExperiment": "lcprop.lc.requests",
    "SweepMemberStatus": "lcprop.lc.requests",
    "SweepParameter": "lcprop.lc.requests",
    "TD_CHECKPOINT_SCHEMA_VERSION": "lcprop.lc.persistence",
    "TimeDependentCheckpoint": "lcprop.lc.persistence",
    "TimeDependentRunRequest": "lcprop.lc.requests",
    "TimeDependentRunResult": "lcprop.lc.results",
    "TimeDependentSolverOptions": "lcprop.lc.requests",
    "TimeSpec": "lcprop.lc.normalization",
    "ThetaSolver": "lcprop.lc.requests",
    "build_static_torque_balance_data": "lcprop.lc.static_torque_balance",
    "continue_static": "lcprop.lc.workflows",
    "continue_timedependent": "lcprop.lc.workflows",
    "from_parameter_sweep_result": "lcprop.lc.products",
    "from_cell": "lcprop.lc.normalization",
    "from_soliton_existence_result": "lcprop.lc.products",
    "from_soliton_result": "lcprop.lc.products",
    "from_static_live_state": "lcprop.lc.products",
    "from_static_result": "lcprop.lc.products",
    "from_timedependent_live_state": "lcprop.lc.products",
    "from_timedependent_result": "lcprop.lc.products",
    "load_static_checkpoint": "lcprop.lc.persistence",
    "load_timedependent_checkpoint": "lcprop.lc.persistence",
    "lc_grid_summary": "lcprop.lc.normalization",
    "make_lc_spatial_normalization": "lcprop.lc.normalization",
    "plot_static_torque_balance": "lcprop.lc.static_torque_balance",
    "polish_soliton": "lcprop.lc.workflows",
    "residual_theta_static": "lcprop.lc.diagnostics",
    "run_parameter_sweep": "lcprop.lc.workflows",
    "run_soliton": "lcprop.lc.workflows",
    "run_soliton_existence": "lcprop.lc.workflows",
    "run_static": "lcprop.lc.workflows",
    "run_timedependent": "lcprop.lc.workflows",
    "save_static_checkpoint": "lcprop.lc.persistence",
    "save_timedependent_checkpoint": "lcprop.lc.persistence",
    "static_request_fingerprint": "lcprop.lc.persistence",
    "theta_metrics": "lcprop.lc.diagnostics",
    "theta_update_metrics": "lcprop.lc.diagnostics",
    "timedependent_state_from_static_result": "lcprop.lc.workflows",
    "to_run_data": "lcprop.lc.products",
    "validate_static_checkpoint": "lcprop.lc.persistence",
    "validate_static_continuation": "lcprop.lc.workflows",
    "validate_timedependent_continuation": "lcprop.lc.workflows",
}


def __getattr__(name: str):
    """Resolve one LC-owned alias without changing the implementation object."""

    module_name = _FACADE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "LC_MATERIAL_ID",
    *_FACADE_MODULES,
]
