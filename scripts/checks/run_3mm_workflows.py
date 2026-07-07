import time
import numpy as np

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import (
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
    OutputOptions,
)
from lcprop.workflows import run_static, run_timedependent, run_soliton_existence
from lcprop.workflows.soliton_existence import SolitonExistenceRequest


def base_static_request(power_mW=1.0):
    return StaticRunRequest(
        grid=GridSpec(
            Nx=64,
            Ny=64,
            dz_um=20.0,
            x_aperture_um=75.0,
            y_aperture_um=100.0,
            z_length_um=3000.0,
        ),
        material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
        bias=BiasSpec(theta_bc=0.0),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=power_mW,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            max_iterations=3,
        ),
        output=OutputOptions(),
    )


def report(name, result, t0):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"elapsed_s      {time.perf_counter() - t0:.3f}")
    print(f"power_initial  {result.power_initial:.6g}")
    print(f"power_final    {result.power_final:.6g}")
    print(f"A shape        {result.A_final.shape}")
    print(f"theta shape    {result.theta_final.shape}")


# 1) 3 mm static request
t0 = time.perf_counter()
static_result = run_static(base_static_request(1.0))
report("3 mm static, 3 um waist, 1 mW", static_result, t0)


# 2) 3 mm TD request
base = base_static_request(1.0)

td_req = TimeDependentRunRequest(
    grid=base.grid,
    material=base.material,
    bias=base.bias,
    beams=base.beams,
    solver=TimeDependentSolverOptions(
        Nt=2,
        dt=7.5e-4,
        gamma_z=0.0,
    ),
    output=base.output,
)

t0 = time.perf_counter()
td_result = run_timedependent(td_req)
report("3 mm TD, 3 um waist, 1 mW, Nt=2", td_result, t0)


# 3) Soliton existence curve
exist_req = SolitonExistenceRequest(
    base=base_static_request(0.5),
    powers_mW=(0.5, 1.0, 2.0),
    continuation=True,
    soliton_max_outer=5,
    theta_steps_per_outer=5,
    field_mix=0.25,
    theta_mix=1.0,
    tol_residual_rms=1e9,
    tol_residual_max=1e9,
)

t0 = time.perf_counter()
exist_result = run_soliton_existence(exist_req)

print("\nSoliton existence curve: 0.5, 1.0, 2.0 mW")
print("-------------------------------------------")
print(f"elapsed_s      {time.perf_counter() - t0:.3f}")
print(f"num_points     {exist_result.metrics['num_points']}")
for row in exist_result.samples:
    print(
        f"P={row['requested_power_mW']:g} mW  "
        f"beta={row.get('beta', np.nan):.6g}  "
        f"theta_max={row.get('theta_max', np.nan):.6g}  "
        f"Imax={row.get('Imax', np.nan):.6g}  "
        f"res_rms={row.get('residual_rms', np.nan):.6g}  "
        f"cont={row['continuation_used']}"
    )
