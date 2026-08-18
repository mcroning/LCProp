"""Optical projection boundary for full-transverse PR states."""

from __future__ import annotations

import numpy as np

from lcprop.pr.transverse.specs import PRTransverseProjectionProfile


def project_active_field(
    E_x, E_y, *, profile: PRTransverseProjectionProfile
) -> np.ndarray:
    """Project vector transverse fields to the scalar optical response."""

    profile.validate()
    field_x = np.asarray(E_x)
    field_y = np.asarray(E_y)
    if field_x.shape != field_y.shape:
        raise ValueError("E_x and E_y must have identical shapes")
    return float(profile.g_x) * field_x + float(profile.g_y) * field_y


__all__ = ["project_active_field"]
