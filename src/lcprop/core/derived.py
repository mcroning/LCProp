"""Compatibility exports for LC-owned derived-physics helpers.

Canonical definitions live under :mod:`lcprop.lc`. New code should import
them from their LC-owned modules rather than from this historical path.
"""

from lcprop.lc.bias import (
    EPS0,
    compute_b_from_voltage,
    compute_freedericksz_voltage,
    resolved_b,
)
from lcprop.lc.bias_cosine import theta_center
from lcprop.lc.optical_response import compute_neff

__all__ = [
    "EPS0",
    "compute_b_from_voltage",
    "compute_freedericksz_voltage",
    "compute_neff",
    "resolved_b",
    "theta_center",
]
