import numpy as np

from lcprop.lc.coupling import compute_bi_from_power as new_bi
from lc_reference.physics.coupling import compute_bi_from_power as ref_bi


def test_compute_bi_from_power_matches_reference():
    cases = [
        dict(P_mW=0.1, d_um=75.0, K=7e-12, ne=1.7, no=1.5),
        dict(P_mW=1.0, d_um=75.0, K=7e-12, ne=1.7, no=1.5),
        dict(P_mW=2.0, d_um=75.0, K=1.2e-11, ne=1.7, no=1.5),
        dict(P_mW=4.0, d_um=50.0, K=1.2e-11, ne=1.8, no=1.5),
    ]

    for case in cases:
        assert np.isclose(new_bi(**case), ref_bi(**case))
