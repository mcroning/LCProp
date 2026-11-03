import numpy as np

from lcprop.core import derived as legacy_derived
from lcprop.core.derived import (
    EPS0,
    compute_b_from_voltage,
    compute_freedericksz_voltage,
    compute_neff,
)
from lcprop.lc import bias as lc_bias
from lcprop.lc import bias_cosine as lc_bias_cosine
from lcprop.lc import optical_response as lc_optical_response
from lcprop.lc.coupling import compute_bi_from_power


def test_legacy_derived_exports_preserve_lc_owned_identity_and_provenance():
    assert legacy_derived.EPS0 is lc_bias.EPS0
    assert legacy_derived.compute_b_from_voltage is lc_bias.compute_b_from_voltage
    assert (
        legacy_derived.compute_freedericksz_voltage
        is lc_bias.compute_freedericksz_voltage
    )
    assert legacy_derived.resolved_b is lc_bias.resolved_b
    assert legacy_derived.theta_center is lc_bias_cosine.theta_center
    assert legacy_derived.compute_neff is lc_optical_response.compute_neff

    assert lc_bias.compute_b_from_voltage.__module__ == "lcprop.lc.bias"
    assert lc_bias.compute_freedericksz_voltage.__module__ == "lcprop.lc.bias"
    assert lc_bias.resolved_b.__module__ == "lcprop.lc.bias"
    assert lc_bias_cosine.theta_center.__module__ == "lcprop.lc.bias_cosine"
    assert lc_optical_response.compute_neff.__module__ == (
        "lcprop.lc.optical_response"
    )

def test_compute_b_from_voltage():
    b = compute_b_from_voltage(
        1.0,
        K=1.2e-11,
        delta_epsilon=10.3,
    )

    expected = 10.3 * EPS0 / (8.0 * 1.2e-11)
    assert np.isclose(b, expected)


def test_freedericksz_voltage_matches_bc():
    K = 1.2e-11
    delta_epsilon = 10.3

    VF = compute_freedericksz_voltage(K=K, delta_epsilon=delta_epsilon)

    b = compute_b_from_voltage(
        VF,
        K=K,
        delta_epsilon=delta_epsilon,
    )

    assert np.isclose(b, np.pi**2 / 8.0)


def test_compute_neff_limits():
    ne = 1.7
    no = 1.5

    assert np.isclose(compute_neff(0.0, ne=ne, no=no), no)
    assert np.isclose(compute_neff(np.pi / 2, ne=ne, no=no), ne)


from scipy.constants import c as C0

def test_compute_bi_from_power():
    P_mW = 2.0
    d_um = 75.0
    K = 7e-12
    ne = 1.7
    no = 1.5

    d_m = d_um * 1e-6
    P_W = P_mW * 1e-3
    na2 = ne**2 - no**2

    expected = na2 * d_m**2 * 1e12 * P_W / (8.0 * C0 * K)

    actual = compute_bi_from_power(
        P_mW,
        d_um=d_um,
        K=K,
        ne=ne,
        no=no,
    )

    assert np.isclose(actual, expected)
