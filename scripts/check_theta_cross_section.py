import numpy as np
import matplotlib.pyplot as plt

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions, RuntimeOptions
from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_timedependent
from tests.test_all_workflows import make_base_static_request

base = make_base_static_request()

from dataclasses import replace

grid = replace(
    base.grid,
    Nx=256,
    Ny=256,
    z_length_um=3000.0,
    dz_um=5.0,
)
beam = replace(base.beams.channels[0], power_mW=0.1)
beams = replace(base.beams, channels=(beam,))
req = TimeDependentRunRequest(
    grid=grid,
    material=base.material,
    bias=base.bias,
    beams=beams,
    solver=TimeDependentSolverOptions(Nt=10, dt=750e-6),
    output=base.output,
    runtime=RuntimeOptions(precision="float64"),
)

res = run_timedependent(req)
print("bias_summary:", res.bias_summary)
print("beam power_mW:", req.beams.channels[0].power_mW)
run_data = to_run_data(res)

theta = np.asarray(res.theta_final)
bias = np.asarray(res.theta_bias)
delta = theta - bias[None, :, :]

x = np.asarray(run_data.geometry.x)
y = np.asarray(run_data.geometry.y)
z = np.asarray(run_data.geometry.z)

iy = len(y) // 2
iz_list = [0, len(z)//4, len(z)//2, len(z)-1]

plt.figure(figsize=(7, 4))
for iz in iz_list:
    plt.plot(x, delta[iz, :, iy], label=f"z={z[iz]:.0f} µm")

plt.axhline(0, linewidth=1)
plt.xlabel("x (µm)")
plt.ylabel("Δθ (rad)")
plt.title(f"Δθ(x) at y={y[iy]:.3g} µm")
plt.legend()
plt.tight_layout()
plt.show()

print("boundary values:")
for iz in iz_list:
    print(
        f"z={z[iz]:.0f}: left={delta[iz,0,iy]:.3e}, "
        f"right={delta[iz,-1,iy]:.3e}, "
        f"min={delta[iz,:,iy].min():.3e}, "
        f"max={delta[iz,:,iy].max():.3e}"
    )
