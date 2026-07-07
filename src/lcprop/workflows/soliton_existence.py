"""Soliton existence-curve workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from lcprop.core.requests import StaticRunRequest
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.workflows.soliton import SolitonRequest, SolitonResult, run_soliton


ExistenceSolver = Literal["soliton"]


@dataclass(frozen=True)
class SolitonExistenceRequest:
    base: StaticRunRequest
    powers_mW: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    continuation: bool = True
    solver: ExistenceSolver = "soliton"

    soliton_max_outer: int = 80
    theta_steps_per_outer: int = 50
    field_mix: float = 0.25
    theta_mix: float = 1.0
    tol_field: float = 1e-4
    tol_theta: float = 1e-5
    tol_residual_rms: float = 1e-3
    tol_residual_max: float = 1e-2

    def validate(self) -> None:
        self.base.grid.validate()
        self.base.material.validate()
        self.base.bias.validate()
        self.base.beams.validate()

        if self.solver != "soliton":
            raise ValueError("solver must be 'soliton'")
        if len(self.powers_mW) == 0:
            raise ValueError("powers_mW must be nonempty")
        for p in self.powers_mW:
            if p < 0.0:
                raise ValueError("powers_mW must be nonnegative")
        if self.soliton_max_outer < 1:
            raise ValueError("soliton_max_outer must be >= 1")
        if self.theta_steps_per_outer < 1:
            raise ValueError("theta_steps_per_outer must be >= 1")
        if not (0.0 < self.field_mix <= 1.0):
            raise ValueError("field_mix must be in (0, 1]")
        if not (0.0 < self.theta_mix <= 1.0):
            raise ValueError("theta_mix must be in (0, 1]")


@dataclass
class SolitonExistenceResult:
    kind: str = "SolitonExistenceResult"
    metrics: dict = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    results: list[SolitonResult] = field(default_factory=list)


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

    rows: list[dict] = []
    results: list[SolitonResult] = []

    previous_A = None
    previous_theta = None

    for i, requested_power in enumerate(request.powers_mW):
        static_req = make_static_request_for_power(request.base, float(requested_power))
        use_continuation = bool(request.continuation and previous_theta is not None)

        res = _run_one_soliton(
            request,
            static_req,
            initial_A=previous_A if use_continuation else None,
            initial_theta=previous_theta if use_continuation else None,
        )

        previous_A = res.A
        previous_theta = res.theta
        results.append(res)

        row = {
            "i": int(i),
            "solver": request.solver,
            "requested_power_mW": float(requested_power),
            "seed_power_mW": None if i == 0 else float(request.powers_mW[i - 1]),
            "converged": bool(res.converged),
            "continuation_used": bool(use_continuation),
            "continuation_direction": (
                "up" if i == 0 or requested_power >= request.powers_mW[i - 1] else "down"
            ),
            "used_initial_A": bool(res.metrics.get("used_initial_A", False)),
            "used_initial_theta": bool(res.metrics.get("used_initial_theta", False)),
        }

        row.update(res.metrics)
        row["output_power_mW"] = float(res.metrics.get("power", res.metrics.get("target_power", 0.0)))
        rows.append(row)

    metrics: dict = {
        "num_points": len(rows),
        "power_min_mW": float(min(request.powers_mW)),
        "power_max_mW": float(max(request.powers_mW)),
        "converged_count": int(sum(1 for r in results if r.converged)),
        "continuation": bool(request.continuation),
        "solver": request.solver,
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
        samples=rows,
        results=results,
    )


__all__ = [
    "ExistenceSolver",
    "SolitonExistenceRequest",
    "SolitonExistenceResult",
    "make_static_request_for_power",
    "run_soliton_existence",
]
