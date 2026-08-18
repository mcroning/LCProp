from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_ACTIVE_CANCELLATION_SCRIPT = r"""
import json
from threading import Thread
from time import perf_counter, sleep

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.workflow import run_pr_timedependent


grid = GridSpec(
    Nx=512,
    Ny=128,
    x_aperture_um=500.0,
    y_aperture_um=500.0,
    dz_um=10.0,
    z_length_um=1200.0,
)
request = PRRunRequest(
    grid=grid,
    beams=BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=40.0,
                waist_y_um=40.0,
            ),
        ),
    ),
    material=PRMaterialSpec(
        characteristic_wavenumber_per_um_override=0.1,
    ),
    solver=PRSolverOptions(
        Nt=20,
        dt_normalized=0.001,
        integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
    ),
    backend=BackendSpec(
        backend="numpy",
        precision="float64",
        verbose=False,
    ),
)
token = CancellationToken()
timing = {}


def request_stop():
    sleep(0.2)
    timing["requested_at"] = perf_counter()
    token.cancel()


Thread(target=request_stop, daemon=True).start()
started_at = perf_counter()
result = run_pr_timedependent(request, cancellation_token=token)
finished_at = perf_counter()
payload = {
    "status": result.status,
    "completed_steps": result.completed_steps,
    "cancel_to_return_seconds": finished_at - timing["requested_at"],
    "total_seconds": finished_at - started_at,
    "request_to_notice_seconds": (
        result.diagnostics["cancellation_observed_wall_time"]
        - (timing["requested_at"] - started_at)
    ),
    "observed_stage": result.diagnostics["cancellation_observed_stage"],
    "optical_observation": result.diagnostics["final_optical_observation"],
    "accepted_state_unchanged": bool(
        (result.E_final == result.E_initial).all()
    ),
}
print(json.dumps(payload, sort_keys=True))
"""


def test_large_numpy_td_run_cancels_inside_active_optical_pass():
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", _ACTIVE_CANCELLATION_SCRIPT],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "cancelled"
    assert payload["completed_steps"] == 0
    assert payload["accepted_state_unchanged"]
    assert payload["observed_stage"] == "material_source_optical_z_march"
    assert payload["optical_observation"] == "unpropagated_launch_fallback"
    assert payload["request_to_notice_seconds"] < 0.5
    assert payload["cancel_to_return_seconds"] < 1.5
