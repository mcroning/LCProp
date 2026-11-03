"""Compatibility re-exports for LC request types.

Canonical definitions live in :mod:`lcprop.lc.requests`.
"""

from lcprop.lc.requests import (
    CouplingMode,
    OpticsSolver,
    OutputOptions,
    Precision,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticStrategy,
    StaticWorkflowOptions,
    ThetaSolver,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)


__all__ = [
    "CouplingMode",
    "OpticsSolver",
    "OutputOptions",
    "Precision",
    "RuntimeOptions",
    "StaticRunRequest",
    "StaticSolverOptions",
    "StaticStrategy",
    "StaticWorkflowOptions",
    "ThetaSolver",
    "TimeDependentRunRequest",
    "TimeDependentSolverOptions",
]
