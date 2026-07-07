import numpy as np

from lcprop.lc.coupling import compute_bi_from_power


def test_compute_bi_from_power_known_value():
    bi = compute_bi_from_power(
        1.0,
        d_um=75.0,
        K=7e-12,
        ne=1.7,
        no=1.5,
    )

    # Regression value from the validated LC implementation.
    assert np.isclose(bi, 214.43406119881186, rtol=1e-12)
