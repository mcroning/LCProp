import numpy as np

from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions, RuntimeOptions
from lcprop.workflows import run_timedependent
from tests.test_all_workflows import make_base_static_request

base = make_base_static_request()

req = TimeDependentRunRequest(
    grid=base.grid,
    material=base.material,
    bias=base.bias,
    beams=base.beams,
    solver=TimeDependentSolverOptions(Nt=10, dt=750e-6),
    output=base.output,
    runtime=RuntimeOptions(precision="float64"),
)

res = run_timedependent(req)

theta = np.asarray(res.theta_final)
bias = np.asarray(res.theta_bias)
delta = theta - bias[None, :, :]

print("theta BC left/right:",
      np.max(np.abs(theta[:, 0, :] - req.bias.theta_bc)),
      np.max(np.abs(theta[:, -1, :] - req.bias.theta_bc)))

print("bias BC left/right:",
      np.max(np.abs(bias[0, :] - req.bias.theta_bc)),
      np.max(np.abs(bias[-1, :] - req.bias.theta_bc)))

print("delta edge left min/max/maxabs:",
      delta[:, 0, :].min(), delta[:, 0, :].max(), np.max(np.abs(delta[:, 0, :])))

print("delta edge right min/max/maxabs:",
      delta[:, -1, :].min(), delta[:, -1, :].max(), np.max(np.abs(delta[:, -1, :])))

print("delta full min/max/rms:",
      delta.min(), delta.max(), np.sqrt(np.mean(delta**2)))
