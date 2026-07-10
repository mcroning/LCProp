

"""Generic parameter sweep workflow for LCProp experiments.

This module intentionally starts small.  The first supported sweep is the
soliton power sweep, which is the traditional soliton existence curve.  The
interfaces are generic so additional experiments, parameters, and execution
backends can be added without changing the GUI contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from concurrent.futures import ProcessPoolExecutor

from lcprop.core.beams import BeamStack
from lcprop.workflows.soliton import SolitonRequest, SolitonResult, run_soliton

SweepExperiment = Literal["soliton"]
SweepParameter = Literal["power_mW"]
SweepExecution = Literal["sequential", "parallel"]


@dataclass(frozen=True)
class ParameterSweepRequest:
    """Request for a generic parameter sweep.

    The initial implementation supports the common soliton existence curve:
    run the soliton solver over a sequence of optical powers.
    """

    experiment: SweepExperiment
    parameter: SweepParameter
    values: tuple[float, ...]
    base: SolitonRequest
    continuation: bool = True
    execution: SweepExecution = "sequential"
    max_workers: int | None = None

    def validate(self) -> None:
        if self.experiment != "soliton":
            raise ValueError("Only experiment='soliton' is currently supported")
        if self.parameter != "power_mW":
            raise ValueError("Only parameter='power_mW' is currently supported")
        if self.execution not in {"sequential", "parallel"}:
            raise ValueError("execution must be 'sequential' or 'parallel'")
        if self.execution == "parallel" and self.continuation:
            raise ValueError("Continuation sweeps must be executed sequentially")
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1 or None")
        if not self.values:
            raise ValueError("values must contain at least one point")
        for value in self.values:
            if float(value) < 0.0:
                raise ValueError("sweep values must be nonnegative")
        self.base.validate()


@dataclass
class ParameterSweepResult:
    """Result from a generic parameter sweep."""

    kind: str = "ParameterSweepResult"
    experiment: str = "soliton"
    parameter: str = "power_mW"
    values: tuple[float, ...] = ()
    continuation: bool = True
    execution: str = "sequential"
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)


def _set_single_channel_power(beams: BeamStack, power_mW: float) -> BeamStack:
    if not beams.channels:
        raise ValueError("Soliton power sweep requires at least one beam channel")
    channel0 = replace(beams.channels[0], power_mW=float(power_mW))
    return replace(beams, channels=(channel0,) + tuple(beams.channels[1:]))


def _set_soliton_power(request: SolitonRequest, power_mW: float) -> SolitonRequest:
    static_base = request.base
    beams = _set_single_channel_power(static_base.beams, power_mW)
    return replace(request, base=replace(static_base, beams=beams))


def _sample_from_soliton_result(index: int, value: float, result: SolitonResult) -> dict[str, Any]:
    metrics = dict(result.metrics)
    return {
        "i": int(index),
        "parameter": "power_mW",
        "value": float(value),
        "requested_power_mW": float(value),
        "mode": result.mode,
        "converged": bool(result.converged),
        "beta": metrics.get("beta"),
        "theta_max": metrics.get("theta_max"),
        "Imax": metrics.get("Imax"),
        "residual_rms": metrics.get("residual_rms", metrics.get("final_residual_rms")),
        "residual_max": metrics.get("residual_max", metrics.get("final_residual_max")),
        "field_rel": metrics.get("field_rel"),
        "overlap_abs": metrics.get("overlap_abs"),
        "dtheta_rms": metrics.get("dtheta_rms"),
        "elapsed_s": metrics.get("elapsed_s"),
        "outer": metrics.get("outer"),
    }


# --- Parallel/Sequential helpers ---

def _run_soliton_sweep_point(args) -> tuple[int, float, SolitonResult]:
    """Run one independent soliton sweep point.

    This top-level helper is intentionally pickle-friendly for
    ProcessPoolExecutor.
    """
    index, value, base = args
    soliton_req = _set_soliton_power(base, float(value))
    result = run_soliton(soliton_req)
    return int(index), float(value), result


def _run_sequential_soliton_sweep(request: ParameterSweepRequest) -> tuple[list[SolitonResult], list[dict[str, Any]]]:
    results: list[SolitonResult] = []
    samples: list[dict[str, Any]] = []
    seed_A = request.base.initial_A
    seed_theta = request.base.initial_theta

    for i, value in enumerate(request.values):
        soliton_req = _set_soliton_power(request.base, float(value))
        if request.continuation and i > 0:
            soliton_req = replace(soliton_req, initial_A=seed_A, initial_theta=seed_theta)

        result = run_soliton(soliton_req)
        results.append(result)
        samples.append(_sample_from_soliton_result(i, float(value), result))

        if request.continuation and result.converged:
            seed_A = result.A
            seed_theta = result.theta

    return results, samples


def _run_parallel_soliton_sweep(request: ParameterSweepRequest) -> tuple[list[SolitonResult], list[dict[str, Any]]]:
    tasks = [(i, float(value), request.base) for i, value in enumerate(request.values)]
    max_workers = None if request.max_workers is None else int(request.max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        completed = list(executor.map(_run_soliton_sweep_point, tasks))

    completed.sort(key=lambda item: item[0])
    results = [result for _, _, result in completed]
    samples = [
        _sample_from_soliton_result(i, value, result)
        for i, value, result in completed
    ]
    return results, samples


def run_parameter_sweep(request: ParameterSweepRequest) -> ParameterSweepResult:
    """Run a parameter sweep.

    The first implementation supports sequential soliton power sweeps.  If
    continuation is enabled, each converged solution seeds the next power.
    """

    request.validate()

    if request.execution == "parallel":
        results, samples = _run_parallel_soliton_sweep(request)
    else:
        results, samples = _run_sequential_soliton_sweep(request)

    metrics: dict[str, Any] = {
        "experiment": request.experiment,
        "parameter": request.parameter,
        "n_points": len(results),
        "converged_count": int(sum(1 for r in results if r.converged)),
        "continuation": bool(request.continuation),
        "execution": request.execution,
        "max_workers": request.max_workers,
        "mode": request.base.mode,
    }

    return ParameterSweepResult(
        experiment=request.experiment,
        parameter=request.parameter,
        values=tuple(float(v) for v in request.values),
        continuation=bool(request.continuation),
        execution=request.execution,
        metrics=metrics,
        samples=samples,
        results=results,
    )


__all__ = [
    "ParameterSweepRequest",
    "ParameterSweepResult",
    "run_parameter_sweep",
]