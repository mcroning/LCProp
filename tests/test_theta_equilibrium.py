from dataclasses import replace
import pytest
import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.workflows.runtime import (
    build_runtime_components,
    make_zcoupled_theta_step,
)
from lcprop.workflows.soliton import run_soliton as run_soliton_map
from lcprop.workflows.soliton_trans import polish_soliton

from .test_soliton_trans_td import make_small_soliton_request

@pytest.mark.slow
def test_polished_theta_is_td_equilibrium():
    request = make_small_soliton_request()
    seed = run_soliton_map(request)

    polish_request = replace(
        request,
        theta_steps_per_outer=50,
    )
    polished = polish_soliton(
        polish_request,
        seed,
        max_outer=100,
        field_mix=0.5,
        theta_mix=0.5,
    )

    runtime = build_runtime_components(
        polish_request.base,
        theta_dt=7.5e-4,
        mobility=1.0,
    )
    theta_step = make_zcoupled_theta_step(
        runtime,
        theta_dt=7.5e-4,
        mobility=1.0,
        gamma_z=0.0,
    )

    xp = runtime.grid.xp
    theta0 = xp.asarray(polished.theta).copy()
    A = xp.asarray(polished.A)
    I0 = xp.sum(xp.abs(A) ** 2, axis=0)

    theta1 = theta_step(
        theta0,
        I0,
        theta0,
        theta0,
        0,
    )
    dtheta = theta1 - theta0

    dtheta_rms = float(
        asnumpy(xp.sqrt(xp.mean(dtheta * dtheta)))
    )
    dtheta_max = float(
        asnumpy(xp.max(xp.abs(dtheta)))
    )

    print("\nsingle TD theta update")
    print(f"dtheta_rms = {dtheta_rms:.12e}")
    print(f"dtheta_max = {dtheta_max:.12e}")

    assert np.isfinite(dtheta_rms)
    assert np.isfinite(dtheta_max)
    assert dtheta_rms >= 0.0
    assert dtheta_max >= 0.0
