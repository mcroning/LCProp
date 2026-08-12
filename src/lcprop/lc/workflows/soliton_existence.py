"""Soliton existence-curve workflow."""

from __future__ import annotations

from dataclasses import replace

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.lc.requests import (
    ExistenceSolver,
    ParameterSweepRequest,
    SolitonExistenceRequest,
    SolitonRequest,
    StaticRunRequest,
)
from lcprop.lc.results import SolitonExistenceResult, SolitonResult
from lcprop.lc.workflows.soliton import run_soliton
from lcprop.lc.workflows.sweep import run_parameter_sweep


def _with_total_power(beams: BeamStack, total_power_mW: float) -> BeamStack:
    beams.validate()

    old_total = sum(float(ch.power_mW) for ch in beams.channels)

    if old_total <= 0.0:
        new_powers = [float(total_power_mW) / len(beams.channels)] * len(beams.channels)
    else:
        scale = float(total_power_mW) / old_total
        new_powers = [float(ch.power_mW) * scale for ch in beams.channels]

    new_channels: list[BeamChannel] = [
        replace(ch, power_mW=float(p))
        for ch, p in zip(beams.channels, new_powers)
    ]

    return BeamStack(channels=tuple(new_channels), coherence=beams.coherence)


def make_static_request_for_power(base: StaticRunRequest, power_mW: float) -> StaticRunRequest:
    return replace(base, beams=_with_total_power(base.beams, float(power_mW)))


def _run_one_soliton(
    request: SolitonExistenceRequest,
    static_req: StaticRunRequest,
    *,
    initial_A=None,
    initial_theta=None,
) -> SolitonResult:
    return run_soliton(
        SolitonRequest(
            base=static_req,
            mode=request.mode,
            max_outer=int(request.soliton_max_outer),
            theta_steps_per_outer=int(request.theta_steps_per_outer),
            field_mix=float(request.field_mix),
            theta_mix=float(request.theta_mix),
            tol_field=float(request.tol_field),
            tol_theta=float(request.tol_theta),
            tol_residual_rms=float(request.tol_residual_rms),
            tol_residual_max=float(request.tol_residual_max),
            initial_A=initial_A,
            initial_theta=initial_theta,
        )
    )


def run_soliton_existence(request: SolitonExistenceRequest) -> SolitonExistenceResult:
    request.validate()

    base_soliton = SolitonRequest(
        base=request.base,
        mode=request.mode,
        max_outer=int(request.soliton_max_outer),
        theta_steps_per_outer=int(request.theta_steps_per_outer),
        field_mix=float(request.field_mix),
        theta_mix=float(request.theta_mix),
        tol_field=float(request.tol_field),
        tol_theta=float(request.tol_theta),
        tol_residual_rms=float(request.tol_residual_rms),
        tol_residual_max=float(request.tol_residual_max),
    )

    sweep_result = run_parameter_sweep(ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=tuple(float(p) for p in request.powers_mW),
        base=base_soliton,
        continuation=bool(request.continuation),
        execution="sequential",
    ))

    rows: list[dict] = []
    for i, row in enumerate(sweep_result.samples):
        prev_power = None if i == 0 else float(request.powers_mW[i - 1])
        requested_power = float(request.powers_mW[i])
        converted = dict(row)
        converted.update({
            "i": int(i),
            "solver": request.solver,
            "mode": request.mode,
            "requested_power_mW": requested_power,
            "seed_power_mW": prev_power,
            "continuation_used": bool(request.continuation and i > 0),
            "continuation_direction": (
                "up" if i == 0 or requested_power >= float(request.powers_mW[i - 1]) else "down"
            ),
        })
        res = sweep_result.results[i]
        converted.update(res.metrics)
        converted["output_power_mW"] = float(
            res.metrics.get("physical_power_mW", res.metrics.get("target_power_mW", 0.0))
        )
        rows.append(converted)

    metrics: dict = {
        "num_points": len(rows),
        "power_min_mW": float(min(request.powers_mW)),
        "power_max_mW": float(max(request.powers_mW)),
        "converged_count": int(sum(1 for r in sweep_result.results if r.converged)),
        "continuation": bool(request.continuation),
        "solver": request.solver,
        "mode": request.mode,
    }

    if rows:
        metrics["last_requested_power_mW"] = float(rows[-1]["requested_power_mW"])
        metrics["last_output_power_mW"] = float(rows[-1]["output_power_mW"])
        metrics["last_theta_max"] = float(rows[-1].get("theta_max", 0.0))
        metrics["last_Imax"] = float(rows[-1].get("Imax", 0.0))
        metrics["last_beta"] = float(rows[-1].get("beta", float("nan")))
        metrics["last_residual_rms"] = float(rows[-1].get("residual_rms", 0.0))
        metrics["last_residual_max"] = float(rows[-1].get("residual_max", 0.0))

    return SolitonExistenceResult(
        metrics=metrics,
        mode=request.mode,
        samples=rows,
        results=list(sweep_result.results),
    )


__all__ = [
    "ExistenceSolver",
    "SolitonExistenceRequest",
    "SolitonExistenceResult",
    "make_static_request_for_power",
    "run_soliton_existence",
]
