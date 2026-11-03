"""Compatibility exports for canonical LC workflow entry points."""

from lcprop.lc.workflows.static import (
    continue_static,
    run_static,
    validate_static_continuation,
)
from lcprop.lc.workflows.timedependent import (
    continue_timedependent,
    run_timedependent,
    timedependent_state_from_static_result,
    validate_timedependent_continuation,
)
from lcprop.lc.workflows.soliton import run_soliton
from lcprop.lc.workflows.soliton_existence import run_soliton_existence
from lcprop.lc.workflows.sweep import run_parameter_sweep

__all__ = [
    "run_static",
    "continue_static",
    "validate_static_continuation",
    "run_timedependent",
    "continue_timedependent",
    "validate_timedependent_continuation",
    "timedependent_state_from_static_result",
    "run_soliton",
    "run_soliton_existence",
    "run_parameter_sweep",
]
