#!/usr/bin/env python3
"""Research-only diagnosis of transverse-static coherent-grating failure.

This script wraps production entry points without changing their behavior.  It
is intentionally not a public workflow or solver API.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.optics.splitstep import linear_kernel
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.static import (
    _physical_state_validity,
    project_production_resolved_modes,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.transverse.projection import project_active_field
import lcprop.pr.transverse.static_workflow as workflow


def fixture(
    *,
    n: int,
    coherent: bool,
    dz_um: float = 10.0,
    backend: str = "numpy",
    precision: str = "float64",
):
    groups = ("shared", "shared") if coherent else ("beam-a", "beam-b")
    channels = (
        BeamChannel(
            name="beam-1", power_mW=1.0, wavelength_um=0.633,
            waist_x_um=20.0, waist_y_um=20.0,
            x0_um=11.8072, y0_um=0.0,
            tilt_x_rad_per_um=-0.9909508004,
            coherence_group=groups[0],
        ),
        BeamChannel(
            name="beam-2", power_mW=1.0, wavelength_um=0.633,
            waist_x_um=20.0, waist_y_um=20.0,
            x0_um=-14.3039, y0_um=-0.2477,
            tilt_x_rad_per_um=0.9909508004,
            coherence_group=groups[1],
        ),
    )
    return workflow.PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=n, Ny=n, x_aperture_um=100.0, y_aperture_um=100.0,
            z_length_um=200.0, dz_um=dz_um,
        ),
        beams=BeamStack(channels=channels, coherence="coherent"),
        material=PRMaterialSpec(
            dark_intensity=0.01, uniform_background_intensity=0.0,
            applied_field=0.0, gain_length_product=1.0,
            refractive_index=2.4, relative_permittivity=2500.0,
            mobile_charge_density_m3=6.4e22, temperature_K=293.0,
        ),
        solver=workflow.PRTransverseStaticWorkflowOptions(),
        backend=BackendSpec(backend=backend, precision=precision, verbose=False),
    )


def portable(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v) for v in value]
    return value


def rms(value) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2, dtype=np.float64)))


def aligned_rms(left, right) -> float:
    overlap = np.vdot(right.ravel(), left.ravel())
    phase = 1.0 if overlap == 0 else np.exp(-1j * np.angle(overlap))
    return rms(phase * left - right)


def optics_context(request):
    backend = get_backend(request.backend)
    grid = make_grid(request.grid, xp=backend.xp, real_dtype=backend.real_dtype)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    wavelength = request.beams.channels[0].wavelength_um
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / request.solver.optical_substeps,
        wavelength=wavelength,
        n_ref=request.material.refractive_index,
        xp=np,
    )
    k0 = request.material.characteristic_wavenumber_per_um
    return {
        "grid": grid, "A0": launch.A0,
        "kernel": kernel, "peak": channel_peak_intensity_reference(launch.A0, xp=np),
        "wavelength": wavelength,
        "dx": k0 * grid.dx_um, "dy": k0 * grid.dy_um,
    }


def optical_pass(request, context, psi):
    return workflow._optical_pass(
        context["A0"], psi, request=request, grid=context["grid"],
        kernel=context["kernel"], peak_reference=context["peak"],
        wavelength_um=context["wavelength"],
        dx_normalized=context["dx"], dy_normalized=context["dy"],
        cancellation_token=None,
    )


def trace_case(request, *, probe_failed_direction: bool):
    original_solve = workflow.solve_pr_transverse_static_intensity
    captures = []

    def capture_solve(intensity, **kwargs):
        result = original_solve(intensity, **kwargs)
        captures.append({
            "source": np.asarray(intensity).copy(),
            "initial_psi": np.asarray(kwargs["initial_psi"]).copy(),
            "result": result,
        })
        return result

    workflow.solve_pr_transverse_static_intensity = capture_solve
    started = perf_counter()
    try:
        result = workflow._run_pr_transverse_static_at_visibility(
            request, visibility=1.0
        )
    finally:
        workflow.solve_pr_transverse_static_intensity = original_solve

    material = []
    for outer, capture in enumerate(captures, 1):
        solved = capture["result"]
        material.append({
            "outer": outer,
            "converged": solved.converged,
            "status": solved.status,
            "planes_converged": sum(s.converged for s in solved.plane_summaries),
            "planes": len(solved.plane_summaries),
            "worst_equilibrium_rms": max(s.equilibrium_rms for s in solved.plane_summaries),
            "worst_equilibrium_max": max(s.equilibrium_max for s in solved.plane_summaries),
            "newton_iterations": sum(s.newton_iterations for s in solved.plane_summaries),
            "pcg_iterations": sum(s.pcg_iterations for s in solved.plane_summaries),
            "backtracks": sum(s.backtracks for s in solved.plane_summaries),
        })

    payload = {
        "request": portable(asdict(request)),
        "status": result.status,
        "converged": result.converged,
        "completed_coupled_iterations": result.completed_coupled_iterations,
        "wall_seconds": perf_counter() - started,
        "iteration_records": [portable(asdict(r)) for r in result.iteration_records],
        "material_solves": material,
        "diagnostics": portable(result.diagnostics),
        "replay": portable(result.replay_diagnostics),
    }
    if not probe_failed_direction or result.diagnostics["termination_reason"] != "coupled_line_search_failed":
        return payload

    failed = captures[-1]
    psi = failed["initial_psi"]
    direction = np.asarray(failed["result"].psi) - psi
    context = optics_context(request)
    baseline_A, baseline_source = optical_pass(request, context, psi)
    _, _, baseline_metrics = workflow._residuals(
        psi, baseline_source, dx_normalized=context["dx"],
        dy_normalized=context["dy"], h_y=request.dielectric.h_y, xp=np,
    )
    alphas = [2.0**(-i) for i in range(request.solver.max_backtracks + 1)]
    alphas += [-(2.0**-i) for i in (12, 10, 8, 6, 4)]
    trials = []
    for alpha in alphas:
        trial = project_production_resolved_modes(
            psi + alpha * direction,
            dx_normalized=context["dx"], dy_normalized=context["dy"],
            h_y=request.dielectric.h_y, xp=np,
        )
        state = state_from_potential(
            trial, dx_normalized=context["dx"], dy_normalized=context["dy"],
            h_y=request.dielectric.h_y, xp=np,
        )
        valid, carrier_mean, carrier_minimum, potential_mean = (
            _physical_state_validity(state, xp=np)
        )
        row = {
            "alpha": alpha, "valid": valid,
            "carrier_mean": carrier_mean, "carrier_minimum": carrier_minimum,
            "potential_mean": potential_mean,
        }
        if valid:
            trial_A, trial_source = optical_pass(request, context, trial)
            _, _, metrics = workflow._residuals(
                trial, trial_source, dx_normalized=context["dx"],
                dy_normalized=context["dy"], h_y=request.dielectric.h_y, xp=np,
            )
            required_rms = (
                (1.0 - request.solver.armijo_fraction * alpha) * baseline_metrics[0]
                if alpha > 0 else None
            )
            row.update({
                "equilibrium_rms": metrics[0], "equilibrium_max": metrics[1],
                "td_rhs_rms": metrics[2], "td_rhs_max": metrics[3],
                "rms_delta": metrics[0] - baseline_metrics[0],
                "max_delta": metrics[1] - baseline_metrics[1],
                "required_rms": required_rms,
                "rms_armijo_pass": None if alpha < 0 else metrics[0] <= required_rms,
                "max_guard_pass": metrics[1] <= baseline_metrics[1],
                "source_change_rms": rms(trial_source - baseline_source),
                "optical_change_aligned_rms": aligned_rms(trial_A, baseline_A),
                "E_x_max_abs": float(np.max(np.abs(state.E_x))),
                "E_y_max_abs": float(np.max(np.abs(state.E_y))),
            })
        trials.append(row)
    payload["failed_direction"] = {
        "outer": len(captures),
        "direction_rms": rms(direction),
        "direction_max": float(np.max(np.abs(direction))),
        "baseline_metrics": list(baseline_metrics),
        "trials": trials,
    }
    return payload


def run_contrast_case(*, n: int, visibility: float, initial_psi=None):
    coherent_request = fixture(n=n, coherent=True)
    coherent_request = replace(coherent_request, initial_psi=initial_psi)
    return workflow._run_pr_transverse_static_at_visibility(
        coherent_request, visibility=visibility
    )


def contrast_case(*, n: int, visibility: float, initial_psi=None):
    result = run_contrast_case(
        n=n, visibility=visibility, initial_psi=initial_psi
    )
    return {
        "visibility": visibility, "status": result.status,
        "converged": result.converged,
        "completed_coupled_iterations": result.completed_coupled_iterations,
        "equilibrium_rms": result.diagnostics["equilibrium_residual_rms"],
        "equilibrium_max": result.diagnostics["equilibrium_residual_max"],
        "termination_reason": result.diagnostics["termination_reason"],
    }


def result_summary(result):
    return {
        "status": result.status,
        "converged": result.converged,
        "completed_coupled_iterations": result.completed_coupled_iterations,
        "diagnostics": portable(result.diagnostics),
        "replay": portable(result.replay_diagnostics),
        "timing": portable(result.timing),
        "power_initial": result.power_initial,
        "power_final": result.power_final,
        "iteration_records": [portable(asdict(row)) for row in result.iteration_records],
    }


def equivalence_case(*, n: int):
    request = fixture(n=n, coherent=True)
    direct = workflow._run_pr_transverse_static_at_visibility(
        request, visibility=1.0
    )
    if not direct.converged:
        return {"n": n, "direct": result_summary(direct), "comparison": None}
    continued = workflow._run_pr_transverse_static_visibility_continuation(
        request, direct_result=direct
    )
    profile = direct.resolved_profile
    direct_state = state_from_potential(
        direct.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
        h_y=request.dielectric.h_y,
        xp=np,
    )
    continued_state = state_from_potential(
        continued.psi_final,
        dx_normalized=profile["dx_normalized"],
        dy_normalized=profile["dy_normalized"],
        h_y=request.dielectric.h_y,
        xp=np,
    )
    direct_active = project_active_field(
        direct_state.E_x, direct_state.E_y, profile=request.projection, xp=np
    )
    continued_active = project_active_field(
        continued_state.E_x,
        continued_state.E_y,
        profile=request.projection,
        xp=np,
    )
    coherent_groups = request.beams.coherence_groups
    direct_intensity = total_intensity(
        direct.A_final, coherence_groups=coherent_groups, xp=np
    )
    continued_intensity = total_intensity(
        continued.A_final, coherence_groups=coherent_groups, xp=np
    )
    return {
        "n": n,
        "direct": result_summary(direct),
        "continuation": result_summary(continued),
        "comparison": {
            "phase_aligned_A_relative_l2": aligned_rms(
                continued.A_final, direct.A_final
            ) / max(rms(direct.A_final), np.finfo(float).tiny),
            "intensity_relative_l2": rms(continued_intensity - direct_intensity)
            / max(rms(direct_intensity), np.finfo(float).tiny),
            "psi_relative_l2": rms(continued.psi_final - direct.psi_final)
            / max(rms(direct.psi_final), np.finfo(float).tiny),
            "P_relative_l2": rms(
                continued_state.carrier_density - direct_state.carrier_density
            ) / max(rms(direct_state.carrier_density), np.finfo(float).tiny),
            "E_x_relative_l2": rms(continued_state.E_x - direct_state.E_x)
            / max(rms(direct_state.E_x), np.finfo(float).tiny),
            "E_y_relative_l2": rms(continued_state.E_y - direct_state.E_y)
            / max(rms(direct_state.E_y), np.finfo(float).tiny),
            "E_active_relative_l2": rms(continued_active - direct_active)
            / max(rms(direct_active), np.finfo(float).tiny),
            "source_relative_l2": rms(
                continued.source_intensity_stack - direct.source_intensity_stack
            ) / max(rms(direct.source_intensity_stack), np.finfo(float).tiny),
            "power_relative_difference": (
                continued.power_final - direct.power_final
            ) / direct.power_final,
        },
    }


def production_case(*, n: int, backend: str, precision: str):
    request = fixture(
        n=n,
        coherent=True,
        backend=backend,
        precision=precision,
    )
    result = workflow.run_pr_transverse_static(request)
    payload = {
        "schema": "lcprop.pr_coherent_visibility_continuation.v1",
        "request": portable(asdict(request)),
        "result": result_summary(result),
    }
    if backend == "cupy":
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        payload["gpu"] = {
            "device_id": int(device.id),
            "name": properties["name"].decode()
            if isinstance(properties["name"], bytes)
            else str(properties["name"]),
            "peak_device_memory_bytes": int(
                cp.get_default_memory_pool().total_bytes()
            ),
        }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("trace", "contrast", "continuation", "equivalence", "production"),
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--case", choices=("coherent", "incoherent"))
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "trace":
        if args.case is None:
            parser.error("trace requires --case")
        payload = trace_case(
            fixture(n=args.n, coherent=args.case == "coherent"),
            probe_failed_direction=args.case == "coherent",
        )
    elif args.mode == "contrast":
        payload = {
            "n": args.n,
            "construction": "I_incoherent + visibility*(I_coherent-I_incoherent)",
            "cases": [contrast_case(n=args.n, visibility=v) for v in (0, .25, .5, .75, 1)],
        }
    elif args.mode == "continuation":
        cases = []
        initial_psi = None
        for visibility in (0, .25, .5, .75, 1):
            result = run_contrast_case(
                n=args.n, visibility=visibility, initial_psi=initial_psi
            )
            cases.append({
                "visibility": visibility, "status": result.status,
                "converged": result.converged,
                "completed_coupled_iterations": result.completed_coupled_iterations,
                "equilibrium_rms": result.diagnostics["equilibrium_residual_rms"],
                "equilibrium_max": result.diagnostics["equilibrium_residual_max"],
                "termination_reason": result.diagnostics["termination_reason"],
            })
            if not result.converged:
                break
            initial_psi = result.psi_final
        payload = {
            "n": args.n,
            "construction": "sequential visibility homotopy with prior converged psi",
            "cases": cases,
        }
    elif args.mode == "equivalence":
        payload = equivalence_case(n=args.n)
    else:
        payload = production_case(
            n=args.n, backend=args.backend, precision=args.precision
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": payload.get("status", "complete"),
        "converged": payload.get("converged"),
    }, indent=2))
    if args.mode == "production" and not payload["result"]["converged"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
