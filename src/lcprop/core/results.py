"""Compatibility re-exports for LC result types.

Canonical definitions live in :mod:`lcprop.lc.results`.
"""

from lcprop.lc.results import (
    StaticIterationRecord,
    StaticRunResult,
    StaticSliceSummary,
    TimeDependentRunResult,
)


__all__ = [
    "StaticIterationRecord",
    "StaticRunResult",
    "StaticSliceSummary",
    "TimeDependentRunResult",
]
