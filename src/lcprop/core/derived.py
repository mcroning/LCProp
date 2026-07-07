"""
Derived physical quantities for LCProp.
"""

import math
import numpy as np

EPS0 = 8.8541878128e-12


def compute_b_from_voltage(
    V_bias: float,
    *,
    K: float,
    delta_epsilon: float,
) -> float:
    """Return b = delta_epsilon eps0 V_bias^2 / (8K)."""
    return float(delta_epsilon) * EPS0 * float(V_bias) ** 2 / (8.0 * float(K))


def compute_freedericksz_voltage(
    *,
    K: float,
    delta_epsilon: float,
) -> float:
    """Return one-constant Freedericksz voltage."""
    return math.pi * math.sqrt(float(K) / (EPS0 * float(delta_epsilon)))


def compute_neff(theta, *, ne: float, no: float, xp=np):
    """Extraordinary-ray effective index."""
    c = xp.cos(theta)
    s = xp.sin(theta)
    return (float(ne) * float(no)) / xp.sqrt((float(ne) * c) ** 2 + (float(no) * s) ** 2)


def resolved_b(material, bias) -> float:
    """Return b_override if present, otherwise compute b from voltage."""
    material.validate()
    bias.validate()
    if bias.b_override is not None:
        return float(bias.b_override)
    return compute_b_from_voltage(
        bias.V_bias,
        K=material.K,
        delta_epsilon=material.delta_epsilon,
    )


def theta_center(bias) -> float:
    """Return requested center theta seed, defaulting to pi/4."""
    return math.pi / 4 if bias.theta_center is None else float(bias.theta_center)
