

"""Generic parameter sweep workflow for LCProp experiments.

This module intentionally starts small.  The first supported sweep is the
soliton power sweep, which is the traditional soliton existence curve.  The
interfaces are generic so additional experiments, parameters, and execution
backends can be added without changing the GUI contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal
import os
import uuid

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context

from lcprop.core.beams import BeamStack
from lcprop.workflows.soliton import SolitonRequest, SolitonResult, run_soliton
from lcprop.core.execution import RunProgress

SweepExperiment = Literal["soliton"]
SweepParameter = Literal["power_mW"]
SweepExecution = Literal["sequential", "parallel"]
SweepMemberStatus = Literal["not_started", "completed", "failed", "cancelled"]
DEFAULT_SWEEP_MAX_WORKERS = 4

_PROCESS_CANCELLATION_EVENT = None
_PROCESS_ACTIVE_COUNT = None
_PROCESS_MAXIMUM_ACTIVE = None
_PROCESS_ACTIVE_LOCK = None


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
    status: str = "completed"
    completed_points: int = 0
    total_points: int = 0
    request: Any | None = None
    sweep_id: str = ""
    members: list["SolitonSweepMember"] = field(default_factory=list)


@dataclass
class SolitonSweepMember:
    requested_index: int
    requested_power_mW: float
    status: SweepMemberStatus = "not_started"
    converged: bool | None = None
    result: SolitonResult | None = None
    beta: float | None = None
    residual_rms: float | None = None
    residual_max: float | None = None
    error_text: str | None = None
    completion_order: int | None = None


def resolved_worker_count(
    number_of_requested_powers: int,
    configured_max_workers: int | None,
    *,
    available_logical_cpus: int | None = None,
) -> int:
    available = (
        int(available_logical_cpus)
        if available_logical_cpus is not None
        else int(os.cpu_count() or 1)
    )
    configured = (
        DEFAULT_SWEEP_MAX_WORKERS
        if configured_max_workers is None
        else int(configured_max_workers)
    )
    return max(1, min(int(number_of_requested_powers), configured, available))


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
        "sx_um": metrics.get("sx_um"),
        "sy_um": metrics.get("sy_um"),
        "residual_rms": metrics.get("residual_rms", metrics.get("final_residual_rms")),
        "residual_max": metrics.get("residual_max", metrics.get("final_residual_max")),
        "field_rel": metrics.get("field_rel"),
        "overlap_abs": metrics.get("overlap_abs"),
        "dtheta_rms": metrics.get("dtheta_rms"),
        "elapsed_s": metrics.get("elapsed_s"),
        "outer": metrics.get("outer"),
    }


# --- Parallel/Sequential helpers ---


class _ProcessCancellationToken:
    def is_cancelled(self) -> bool:
        return bool(
            _PROCESS_CANCELLATION_EVENT is not None
            and _PROCESS_CANCELLATION_EVENT.is_set()
        )


def _initialize_sweep_worker(
    cancellation_event,
    active_count,
    maximum_active,
    active_lock,
    startup_barrier,
) -> None:
    """Prevent nested BLAS pools from multiplying each sweep process."""

    global _PROCESS_CANCELLATION_EVENT
    global _PROCESS_ACTIVE_COUNT
    global _PROCESS_MAXIMUM_ACTIVE
    global _PROCESS_ACTIVE_LOCK
    _PROCESS_CANCELLATION_EVENT = cancellation_event
    _PROCESS_ACTIVE_COUNT = active_count
    _PROCESS_MAXIMUM_ACTIVE = maximum_active
    _PROCESS_ACTIVE_LOCK = active_lock

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        threadpool_limits(limits=1)
    startup_barrier.wait(timeout=30.0)


def _run_soliton_sweep_point(
    index: int,
    value: float,
    base: SolitonRequest,
) -> tuple[int, float, SolitonResult]:
    token = _ProcessCancellationToken()
    if _PROCESS_ACTIVE_LOCK is not None:
        with _PROCESS_ACTIVE_LOCK:
            _PROCESS_ACTIVE_COUNT.value += 1
            _PROCESS_MAXIMUM_ACTIVE.value = max(
                _PROCESS_MAXIMUM_ACTIVE.value,
                _PROCESS_ACTIVE_COUNT.value,
            )
    try:
        result = run_soliton(
            _set_soliton_power(base, float(value)),
            cancellation_token=token,
        )
    finally:
        if _PROCESS_ACTIVE_LOCK is not None:
            with _PROCESS_ACTIVE_LOCK:
                _PROCESS_ACTIVE_COUNT.value -= 1
    return int(index), float(value), result


def _completed_member(
    index: int,
    value: float,
    result: SolitonResult,
    completion_order: int,
) -> SolitonSweepMember:
    metrics = result.metrics
    return SolitonSweepMember(
        requested_index=index,
        requested_power_mW=value,
        status="completed",
        converged=bool(result.converged),
        result=result,
        beta=metrics.get("beta"),
        residual_rms=metrics.get(
            "residual_rms", metrics.get("final_residual_rms")
        ),
        residual_max=metrics.get(
            "residual_max", metrics.get("final_residual_max")
        ),
        completion_order=completion_order,
    )


def _result_from_members(
    request: ParameterSweepRequest,
    members: list[SolitonSweepMember],
    *,
    sweep_id: str,
    status: str,
    worker_count: int,
    executor_class: str,
    maximum_active_futures: int = 1,
) -> ParameterSweepResult:
    completed = [member for member in members if member.status == "completed"]
    results = [member.result for member in completed if member.result is not None]
    samples = [
        _sample_from_soliton_result(
            member.requested_index,
            member.requested_power_mW,
            member.result,
        )
        for member in completed
        if member.result is not None
    ]
    failed_count = sum(member.status == "failed" for member in members)
    metrics: dict[str, Any] = {
        "experiment": request.experiment,
        "parameter": request.parameter,
        "n_points": len(completed),
        "converged_count": sum(bool(member.converged) for member in completed),
        "continuation": bool(request.continuation),
        "execution": request.execution,
        "max_workers": request.max_workers,
        "resolved_worker_count": worker_count,
        "available_logical_cpus": int(os.cpu_count() or 1),
        "submitted_futures": sum(
            member.status != "not_started" for member in members
        ),
        "executor_class": executor_class,
        "maximum_active_futures": int(maximum_active_futures),
        "mode": request.base.mode,
        "status": status,
        "completed_points": len(completed),
        "total_points": len(request.values),
        "failed_count": failed_count,
        "sweep_id": sweep_id,
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
        status=status,
        completed_points=len(completed),
        total_points=len(request.values),
        request=request,
        sweep_id=sweep_id,
        members=list(members),
    )


def _emit_sweep_progress(
    request: ParameterSweepRequest,
    members: list[SolitonSweepMember],
    *,
    sweep_id: str,
    worker_count: int,
    executor_class: str,
    current_member: SolitonSweepMember,
    progress_callback,
) -> None:
    if progress_callback is None:
        return
    partial = _result_from_members(
        request,
        members,
        sweep_id=sweep_id,
        status="running",
        worker_count=worker_count,
        executor_class=executor_class,
        maximum_active_futures=min(
            worker_count,
            sum(member.status != "not_started" for member in members),
        ),
    )
    progress_callback(
        RunProgress(
            workflow="soliton_existence",
            status="running",
            completed_units=partial.completed_points,
            total_units=len(request.values),
            current_coordinate=current_member.requested_power_mW,
            coordinate_name=request.parameter,
            coordinate_unit="mW",
            elapsed_wall_time=float(
                sum(
                    float(member.result.metrics.get("elapsed_s", 0.0))
                    for member in members
                    if member.result is not None
                )
            ),
            latest_field_state=partial,
            checkpoint_available=True,
            diagnostics={
                "current_power_mW": current_member.requested_power_mW,
                "converged": bool(current_member.converged),
                "failed_count": partial.metrics["failed_count"],
            },
        )
    )


def _run_sequential_members(
    request: ParameterSweepRequest,
    *,
    sweep_id: str,
    cancellation_token=None,
    progress_callback=None,
) -> tuple[list[SolitonSweepMember], bool]:
    members = [
        SolitonSweepMember(i, float(value))
        for i, value in enumerate(request.values)
    ]
    seed_A = request.base.initial_A
    seed_theta = request.base.initial_theta
    completion_order = 0

    for member in members:
        if cancellation_token is not None and cancellation_token.is_cancelled():
            break
        soliton_request = _set_soliton_power(
            request.base, member.requested_power_mW
        )
        if request.continuation and member.requested_index > 0:
            soliton_request = replace(
                soliton_request,
                initial_A=seed_A,
                initial_theta=seed_theta,
            )
        try:
            result = run_soliton(
                soliton_request,
                cancellation_token=cancellation_token,
            )
        except Exception as exc:
            member.status = "failed"
            member.error_text = f"{type(exc).__name__}: {exc}"
        else:
            if result.status == "stopped":
                member.status = "cancelled"
            else:
                completion_order += 1
                replacement = _completed_member(
                    member.requested_index,
                    member.requested_power_mW,
                    result,
                    completion_order,
                )
                members[member.requested_index] = replacement
                member = replacement
                if request.continuation and result.converged:
                    seed_A = result.A
                    seed_theta = result.theta
        if member.status in {"completed", "failed"}:
            _emit_sweep_progress(
                request,
                members,
                sweep_id=sweep_id,
                worker_count=1,
                executor_class="sequential",
                current_member=member,
                progress_callback=progress_callback,
            )
        if member.status == "cancelled":
            break
    stopped = bool(
        cancellation_token is not None and cancellation_token.is_cancelled()
    )
    return members, stopped


def _run_parallel_members(
    request: ParameterSweepRequest,
    *,
    sweep_id: str,
    cancellation_token=None,
    progress_callback=None,
) -> tuple[list[SolitonSweepMember], bool, int, int]:
    members = [
        SolitonSweepMember(i, float(value))
        for i, value in enumerate(request.values)
    ]
    worker_count = resolved_worker_count(len(members), request.max_workers)
    completion_order = 0
    next_index = 0
    stopped = False

    context = get_context("spawn")
    cancellation_event = context.Event()
    active_count = context.Value("i", 0)
    maximum_active = context.Value("i", 0)
    active_lock = context.Lock()
    startup_barrier = context.Barrier(worker_count)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_sweep_worker,
        initargs=(
            cancellation_event,
            active_count,
            maximum_active,
            active_lock,
            startup_barrier,
        ),
    ) as executor:
        pending = {}

        def submit_available() -> None:
            nonlocal next_index
            while (
                next_index < len(members)
                and len(pending) < worker_count
                and not stopped
            ):
                member = members[next_index]
                future = executor.submit(
                    _run_soliton_sweep_point,
                    member.requested_index,
                    member.requested_power_mW,
                    request.base,
                )
                pending[future] = member.requested_index
                next_index += 1

        submit_available()
        while pending:
            if (
                cancellation_token is not None
                and cancellation_token.is_cancelled()
            ):
                stopped = True
                cancellation_event.set()
            done, _ = wait(
                tuple(pending),
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                index = pending.pop(future)
                member = members[index]
                try:
                    _, value, result = future.result()
                except Exception as exc:
                    member.status = "failed"
                    member.error_text = f"{type(exc).__name__}: {exc}"
                else:
                    if result.status == "stopped":
                        member.status = "cancelled"
                    else:
                        completion_order += 1
                        member = _completed_member(
                            index,
                            value,
                            result,
                            completion_order,
                        )
                        members[index] = member
                if member.status in {"completed", "failed"}:
                    _emit_sweep_progress(
                        request,
                        members,
                        sweep_id=sweep_id,
                        worker_count=worker_count,
                        executor_class="ProcessPoolExecutor",
                        current_member=member,
                        progress_callback=progress_callback,
                    )
            if (
                cancellation_token is not None
                and cancellation_token.is_cancelled()
            ):
                stopped = True
                cancellation_event.set()
            submit_available()
    observed_maximum = int(maximum_active.value)
    return members, stopped, worker_count, observed_maximum


def run_parameter_sweep(
    request: ParameterSweepRequest,
    *,
    cancellation_token=None,
    progress_callback=None,
) -> ParameterSweepResult:
    """Run a parameter sweep.

    The first implementation supports sequential soliton power sweeps.  If
    continuation is enabled, each converged solution seeds the next power.
    """

    request.validate()

    sweep_id = uuid.uuid4().hex
    if request.execution == "parallel":
        members, stopped, worker_count, maximum_active = _run_parallel_members(
            request,
            sweep_id=sweep_id,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        executor_class = "ProcessPoolExecutor"
    else:
        members, stopped = _run_sequential_members(
            request,
            sweep_id=sweep_id,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        worker_count = 1
        maximum_active = 1
        executor_class = "sequential"

    if stopped:
        for member in members:
            if member.status == "not_started":
                member.status = "not_started"
    failed = any(member.status == "failed" for member in members)
    status = "stopped" if stopped else ("completed_with_failures" if failed else "completed")
    return _result_from_members(
        request,
        members,
        sweep_id=sweep_id,
        status=status,
        worker_count=worker_count,
        executor_class=executor_class,
        maximum_active_futures=maximum_active,
    )


__all__ = [
    "ParameterSweepRequest",
    "ParameterSweepResult",
    "SolitonSweepMember",
    "resolved_worker_count",
    "run_parameter_sweep",
]
