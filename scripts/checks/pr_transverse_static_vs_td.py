#!/usr/bin/env python3
"""Small deterministic interpretation check for transverse static versus TD."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.specs import (
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent


def _relative_l2(actual, reference) -> float:
    numerator = np.linalg.norm(np.asarray(actual) - np.asarray(reference))
    denominator = np.linalg.norm(np.asarray(reference))
    return float(numerator / max(denominator, np.finfo(float).tiny))


def _case(gain_length_product: float) -> dict[str, object]:
    grid = GridSpec(
        Nx=64,
        Ny=64,
        x_aperture_um=40.0,
        y_aperture_um=40.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    beams = BeamStack(channels=(BeamChannel(
        wavelength_um=0.633,
        waist_x_um=10.0,
        waist_y_um=10.0,
        coherence_group="static-td-check",
    ),))
    material = PRMaterialSpec(
        dark_intensity=0.4,
        uniform_background_intensity=0.1,
        applied_field=0.0,
        gain_length_product=float(gain_length_product),
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.1,
    )
    backend = BackendSpec(backend="numpy", precision="float64", verbose=False)
    static_request = PRTransverseStaticRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=20),
        backend=backend,
    )
    td_request = PRTransverseRunRequest(
        grid=grid,
        beams=beams,
        material=material,
        solver=PRTransverseSolverOptions(
            Nt=200,
            dt_normalized=0.05,
            optical_substeps=1,
        ),
        backend=backend,
    )
    started = perf_counter()
    static = run_pr_transverse_static(static_request)
    static_seconds = perf_counter() - started
    started = perf_counter()
    td = run_pr_transverse_timedependent(td_request)
    td_seconds = perf_counter() - started
    return {
        "gain_length_product": float(gain_length_product),
        "grid": asdict(grid),
        "static_status": static.status,
        "static_converged": static.converged,
        "static_coupled_iterations": static.completed_coupled_iterations,
        "static_material_newton_iterations": sum(
            record.material_newton_iterations
            for record in static.iteration_records
        ),
        "static_material_pcg_iterations": sum(
            record.material_pcg_iterations
            for record in static.iteration_records
        ),
        "static_coupled_backtracks": sum(
            record.backtracks for record in static.iteration_records
        ),
        "static_equilibrium_residual_max": static.diagnostics[
            "equilibrium_residual_max"
        ],
        "static_td_rhs_residual_max": static.diagnostics[
            "td_rhs_residual_max"
        ],
        "td_material_time_normalized": td.time_normalized,
        "psi_relative_l2_td_vs_static": _relative_l2(
            td.psi_final, static.psi_final
        ),
        "static_runtime_seconds": static_seconds,
        "td_runtime_seconds": td_seconds,
        "interpretation": (
            "This bounded no-scattering case is nearly saturated by tau=10. "
            "Agreement at GL=3 is the expected near-static control. Agreement "
            "at GL=10 does not imply that the saved production fanning history "
            "was saturated; its continuing broad-ring evolution remains a "
            "separate scientific comparison."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema": "lcprop.pr_transverse_static_vs_td.v1",
        "cases": [_case(3.0), _case(10.0)],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
