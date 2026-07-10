from dataclasses import replace
import pytest
import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import (
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.workflows.soliton import (
    SolitonRequest,
    run_soliton as run_soliton_map,
)
from lcprop.workflows.soliton_trans import polish_soliton
from lcprop.workflows.timedependent import run_timedependent


def make_small_soliton_request() -> SolitonRequest:
    base = StaticRunRequest(
        grid=GridSpec(
            Nx=32,
            Ny=32,
            dz_um=5.0,
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


def run_coupled_td(
    request: SolitonRequest,
    result,
    *,
    dz_um: float,
    Nt: int,
    dt: float,
):
    grid = replace(
        request.base.grid,
        dz_um=float(dz_um),
        z_length_um=160.0,
    )

    td_request = TimeDependentRunRequest(
        grid=grid,
        material=request.base.material,
        bias=request.base.bias,
        beams=request.base.beams,
        solver=TimeDependentSolverOptions(
            Nt=int(Nt),
            dt=float(dt),
        ),
        output=request.base.output,
        runtime=request.base.runtime,
        initial_A=result.A,
        initial_theta=result.theta,
    )

    td = run_timedependent(td_request)

    theta_final = np.asarray(
        asnumpy(td.theta_final),
        dtype=np.float64,
    )
    theta_initial = np.asarray(
        asnumpy(result.theta),
        dtype=np.float64,
    )

    theta_change_rms = float(
        np.sqrt(
            np.mean(
                (
                    theta_final
                    - theta_initial[None, :, :]
                )
                ** 2
            )
        )
    )

    A_initial = np.asarray(asnumpy(result.A))
    A_final = np.asarray(asnumpy(td.A_final))

    I_initial = np.sum(np.abs(A_initial) ** 2, axis=0)
    I_final = np.sum(np.abs(A_final) ** 2, axis=0)

    intensity_change_rel = float(
        np.linalg.norm(I_final - I_initial)
        / (np.linalg.norm(I_initial) + 1e-300)
    )

    return {
        "theta_change_rms": theta_change_rms,
        "intensity_change_rel": intensity_change_rel,
        "power_initial": float(td.power_initial),
        "power_final": float(td.power_final),
    }

@pytest.mark.slow
def test_seed_and_polished_solitons_in_coupled_td():
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

    rows = []

    print("\ncoupled TD evolution")
    print(
        "dz_um Nt "
        "seed_dtheta polished_dtheta "
        "seed_dI polished_dI"
    )
    for dz_um in (1.25,):
        for Nt in (100,200,500):
 #           seed_td = run_coupled_td(
 #               request,
 #               seed,
 #               dz_um=dz_um,
 #               Nt=Nt,
 #               dt=7.5e-4,
 #           )
            polished_td = run_coupled_td(
                request,
                polished,
                dz_um=dz_um,
                Nt=Nt,
                dt=7.5e-4,
            )

            rows.append((dz_um, Nt, polished_td))

            print(
                f"{dz_um} {Nt} "
                #f"{seed_td['theta_change_rms']} "
                f"{polished_td['theta_change_rms']} "
                #f"{seed_td['intensity_change_rel']} "
                f"{polished_td['intensity_change_rel']}"
            )

            print(
                "final coupled polish:",
                polished.metrics["final_coupled_polish_cycles"],
                polished.metrics["final_theta_polish_steps"],
                polished.metrics["final_theta_update_rms"],
            )