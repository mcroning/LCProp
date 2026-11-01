"""Explicit potential-space identities for transverse zero-flux equilibrium."""

DERIVATIVE_RESOLVED_V1 = "derivative_resolved_v1"
CONSTANT_GAUGE_V1 = "constant_gauge_v1"


def potential_null_mask(denominator, potential_space=DERIVATIVE_RESOLVED_V1, *, xp):
    """Return excluded potential modes; never change derivative symbols."""
    validate_potential_space(potential_space)
    if potential_space == DERIVATIVE_RESOLVED_V1:
        return denominator == 0.0
    mask = xp.zeros(denominator.shape, dtype=bool)
    mask[0, 0] = True
    return mask


def validate_potential_space(potential_space):
    if potential_space not in (DERIVATIVE_RESOLVED_V1, CONSTANT_GAUGE_V1):
        raise ValueError("Unknown transverse potential space")
