import pytest

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.workflows.runtime import build_runtime_components, make_td_optics_step

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import OutputOptions, StaticRunRequest, StaticSolverOptions
from lcprop.workflows.soliton import SolitonRequest, run_soliton as run_soliton_map
from lcprop.workflows.soliton_trans import polish_soliton

from dataclasses import replace


def make_small_soliton_request() -> SolitonRequest:
    base = StaticRunRequest(
        grid=GridSpec(
            Nx=32,
            Ny=32,
            dz_um=20.0,
            x_aperture_um=32.0,
            y_aperture_um=32.0,
            z_length_um=160.0,
        ),
        material=LCMaterial(),
        bias=BiasSpec(V_bias=0.9144),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    power_mW=0.05,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(max_iterations=1),
        output=OutputOptions(),
    )

    return SolitonRequest(
        base=base,
        mode="00",
        max_outer=50,
        theta_steps_per_outer=20,
        field_mix=0.5,
        theta_mix=0.5,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )


# Helper for frozen-theta breathing analysis
def frozen_theta_breathing(request: SolitonRequest, result):
    """Propagate a field through its own frozen theta and quantify breathing."""
    runtime = build_runtime_components(request.base, theta_dt=7.5e-4, mobility=1.0)
    optics_step = make_td_optics_step(runtime)

    A = runtime.grid.xp.asarray(result.A).copy()
    theta = runtime.grid.xp.asarray(result.theta)
    intensity_stack = []

    for k in range(runtime.grid.Nz):
        A, I_mid = optics_step(A, theta, k)
        intensity_stack.append(asnumpy(I_mid))

    intensity_stack = np.stack(intensity_stack, axis=0)
    peak_z = intensity_stack.max(axis=(1, 2))
    peak_breathing = float((peak_z.max() - peak_z.min()) / (peak_z.mean() + 1e-300))

    reference = intensity_stack[0]
    reference_norm = np.linalg.norm(reference) + 1e-300
    profile_rel = np.linalg.norm(
        intensity_stack - reference[None, :, :], axis=(1, 2)
    ) / reference_norm
    profile_breathing = float(profile_rel.max())

    return {
        "intensity_stack": intensity_stack,
        "peak_z": peak_z,
        "peak_breathing": peak_breathing,
        "profile_breathing": profile_breathing,
    }

def request_with_dz(request: SolitonRequest, dz_um: float) -> SolitonRequest:
    """Return the same physical problem with a different propagation step."""
    grid = replace(
        request.base.grid,
        dz_um=float(dz_um),
        z_length_um=160.0,
    )
    base = replace(request.base, grid=grid)
    return replace(request, base=base)

def test_polish_soliton_preserves_basic_properties():
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


    seed_metrics = dict(seed.metrics)
    polished_metrics = dict(polished.metrics)

    seed_optical_residual = seed_metrics.get("optical_residual", float("nan"))
    polished_optical_residual = polished_metrics["optical_residual"]

    field_change = (
        ((abs(polished.A - seed.A) ** 2).sum()) ** 0.5
        / (((abs(seed.A) ** 2).sum()) ** 0.5 + 1e-300)
    )
    theta_change = (((polished.theta - seed.theta) ** 2).mean()) ** 0.5

    print("\ntransverse polish diagnostic")
    print(f"seed beta:              {seed_metrics.get('beta')}")
    print(f"polished beta:          {polished_metrics.get('beta')}")
    print(f"seed optical residual:  {seed_optical_residual}")
    print(f"polished optical resid: {polished_optical_residual}")
    print(f"seed LC residual rms:   {seed_metrics.get('residual_rms')}")
    print(f"polished LC resid rms:  {polished_metrics.get('residual_rms')}")
    print(f"relative field change:  {field_change}")
    print(f"theta RMS change:       {theta_change}")

    print("\nouter history")
    print("outer beta optical_residual residual_rms field_rel dtheta_rms")
    for row in polished.history:
        print(
            f"{row.get('outer')} "
            f"{row.get('beta')} "
            f"{row.get('optical_residual')} "
            f"{row.get('residual_rms')} "
            f"{row.get('field_rel')} "
            f"{row.get('dtheta_rms')}"
        )




    breathing_by_dz = []

    print("\nfrozen-theta propagation vs dz")
    print("dz_um seed_peak polished_peak seed_profile polished_profile")

    for dz_um in (20.0, 10.0, 5.0, 2.5):
        validation_request = request_with_dz(request, dz_um)

        seed_breathing = frozen_theta_breathing(validation_request, seed)
        polished_breathing = frozen_theta_breathing(validation_request, polished)

        breathing_by_dz.append(
            (dz_um, seed_breathing, polished_breathing)
        )

        print(
            f"{dz_um} "
            f"{seed_breathing['peak_breathing']} "
            f"{polished_breathing['peak_breathing']} "
            f"{seed_breathing['profile_breathing']} "
            f"{polished_breathing['profile_breathing']}"
        )



    assert polished.A.shape == seed.A.shape
    assert polished.theta.shape == seed.theta.shape
    assert polished.intensity.shape == seed.intensity.shape
    assert polished.metrics["target_power_mW"] == pytest.approx(
        seed.metrics["target_power_mW"], rel=1e-12
    )
    assert polished.mode == seed.mode
    assert "optical_residual" in polished.metrics
    assert polished.metrics["optical_residual"] >= 0.0
    assert field_change >= 0.0
    assert theta_change >= 0.0
    assert len(breathing_by_dz) == 4

    for _, seed_breathing, polished_breathing in breathing_by_dz:
        assert seed_breathing["peak_breathing"] >= 0.0
        assert polished_breathing["peak_breathing"] >= 0.0
        assert seed_breathing["profile_breathing"] >= 0.0
        assert polished_breathing["profile_breathing"] >= 0.0
        assert polished.metrics["residual_rms"] >= 0.0
        assert polished.metrics["residual_max"] >= 0.0
