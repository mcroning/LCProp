"""Material-owned experiment request codecs for liquid-crystal workflows."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lcprop.core.context import GridSpec
from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.persistence.experiments import (
    ExperimentPayloadError,
    ExperimentRequestCodec,
    ExperimentRuntimeStateError,
    ExperimentSchemaError,
    dataclass_values,
    decode_beam_stack,
    encode_beam_stack,
    require_exact_keys,
    require_mapping,
)


LC_EXPERIMENT_REQUEST_SCHEMA_VERSION = 1


def _reject_runtime_state(request: Any) -> None:
    populated = tuple(
        name
        for name in ("initial_A", "initial_theta")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "LC experiment requests cannot persist runtime state "
            f"({', '.join(populated)}); use checkpoint persistence for "
            "continuation state"
        )
    if request.output.run_dir is not None:
        raise ExperimentRuntimeStateError(
            "LC experiment requests cannot persist machine-local output.run_dir"
        )


def _validate_common(request: StaticRunRequest | TimeDependentRunRequest) -> None:
    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()
    request.runtime.validate()
    if type(request.output.save_slices) is not bool:
        raise ExperimentPayloadError("output.save_slices must be a boolean")
    if type(request.output.save_full) is not bool:
        raise ExperimentPayloadError("output.save_full must be a boolean")


def _validate_workflow(workflow: StaticWorkflowOptions) -> None:
    if workflow.strategy not in {"fixed_theta", "local_self_consistent"}:
        raise ExperimentPayloadError("invalid LC static workflow strategy")
    if workflow.theta_solver not in {"none", "picard_cn"}:
        raise ExperimentPayloadError("invalid LC theta solver")
    if workflow.optics_solver != "splitstep":
        raise ExperimentPayloadError("invalid LC optics solver")
    if workflow.coupling not in {"frozen", "self_consistent"}:
        raise ExperimentPayloadError("invalid LC coupling mode")


def _validate_static(request: StaticRunRequest) -> None:
    _validate_common(request)
    _validate_workflow(request.solver.workflow)
    if int(request.solver.max_iterations) < 1:
        raise ExperimentPayloadError("solver.max_iterations must be positive")
    if int(request.solver.static_max_relax_iterations) < 1:
        raise ExperimentPayloadError(
            "solver.static_max_relax_iterations must be positive"
        )
    if int(request.solver.static_max_coupled_passes) < 1:
        raise ExperimentPayloadError(
            "solver.static_max_coupled_passes must be positive"
        )


def _validate_timedependent(request: TimeDependentRunRequest) -> None:
    _validate_common(request)
    _validate_workflow(request.solver.workflow)
    if int(request.solver.Nt) < 0:
        raise ExperimentPayloadError("solver.Nt must be nonnegative")
    if float(request.solver.dt) <= 0.0:
        raise ExperimentPayloadError("solver.dt must be positive")
    if int(request.solver.max_picard_iter) < 1:
        raise ExperimentPayloadError("solver.max_picard_iter must be positive")


def _encode_common(request: StaticRunRequest | TimeDependentRunRequest) -> dict:
    _reject_runtime_state(request)
    return {
        "schema_version": LC_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "bias": asdict(request.bias),
        "beams": encode_beam_stack(request.beams),
        "solver": {
            **asdict(request.solver),
            "workflow": asdict(request.solver.workflow),
        },
        "output": {
            "run_dir": None,
            "save_slices": request.output.save_slices,
            "save_full": request.output.save_full,
        },
        "runtime": asdict(request.runtime),
    }


def encode_lc_static_request(request: StaticRunRequest) -> dict:
    if not isinstance(request, StaticRunRequest):
        raise TypeError("request must be a StaticRunRequest")
    _validate_static(request)
    return _encode_common(request)


def encode_lc_timedependent_request(request: TimeDependentRunRequest) -> dict:
    if not isinstance(request, TimeDependentRunRequest):
        raise TypeError("request must be a TimeDependentRunRequest")
    _validate_timedependent(request)
    return _encode_common(request)


def _payload(value: Any) -> dict[str, Any]:
    payload = require_mapping(value, name="LC request_payload")
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "bias",
            "beams",
            "solver",
            "output",
            "runtime",
        },
        name="LC request_payload",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != LC_EXPERIMENT_REQUEST_SCHEMA_VERSION:
        raise ExperimentSchemaError(
            f"unsupported LC experiment request schema version: {version!r}"
        )
    return payload


def _decode_common(payload: dict[str, Any]) -> dict[str, Any]:
    output_values = dataclass_values(
        OutputOptions,
        payload["output"],
        name="LC request_payload.output",
    )
    if output_values["run_dir"] is not None:
        raise ExperimentRuntimeStateError(
            "LC experiment output.run_dir must be null"
        )
    return {
        "grid": GridSpec(
            **dataclass_values(
                GridSpec,
                payload["grid"],
                name="LC request_payload.grid",
            )
        ),
        "material": LCMaterial(
            **dataclass_values(
                LCMaterial,
                payload["material"],
                name="LC request_payload.material",
            )
        ),
        "bias": BiasSpec(
            **dataclass_values(
                BiasSpec,
                payload["bias"],
                name="LC request_payload.bias",
            )
        ),
        "beams": decode_beam_stack(payload["beams"]),
        "output": OutputOptions(**output_values),
        "runtime": RuntimeOptions(
            **dataclass_values(
                RuntimeOptions,
                payload["runtime"],
                name="LC request_payload.runtime",
            )
        ),
    }


def decode_lc_static_request(value: Any) -> StaticRunRequest:
    payload = _payload(value)
    solver_values = dataclass_values(
        StaticSolverOptions,
        payload["solver"],
        name="LC request_payload.solver",
    )
    workflow = StaticWorkflowOptions(
        **dataclass_values(
            StaticWorkflowOptions,
            solver_values.pop("workflow"),
            name="LC request_payload.solver.workflow",
        )
    )
    try:
        request = StaticRunRequest(
            **_decode_common(payload),
            solver=StaticSolverOptions(workflow=workflow, **solver_values),
        )
        _validate_static(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid LC static request: {exc}") from exc
    return request


def decode_lc_timedependent_request(value: Any) -> TimeDependentRunRequest:
    payload = _payload(value)
    solver_values = dataclass_values(
        TimeDependentSolverOptions,
        payload["solver"],
        name="LC request_payload.solver",
    )
    workflow = StaticWorkflowOptions(
        **dataclass_values(
            StaticWorkflowOptions,
            solver_values.pop("workflow"),
            name="LC request_payload.solver.workflow",
        )
    )
    try:
        request = TimeDependentRunRequest(
            **_decode_common(payload),
            solver=TimeDependentSolverOptions(workflow=workflow, **solver_values),
        )
        _validate_timedependent(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid LC time-dependent request: {exc}"
        ) from exc
    return request


LC_STATIC_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=LC_MATERIAL_ID,
    workflow_id="static",
    request_type=StaticRunRequest,
    encode_request=encode_lc_static_request,
    decode_request=decode_lc_static_request,
)

LC_TIMEDEPENDENT_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=LC_MATERIAL_ID,
    workflow_id="timedependent",
    request_type=TimeDependentRunRequest,
    encode_request=encode_lc_timedependent_request,
    decode_request=decode_lc_timedependent_request,
)


__all__ = [
    "LC_EXPERIMENT_REQUEST_SCHEMA_VERSION",
    "LC_STATIC_EXPERIMENT_CODEC",
    "LC_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "decode_lc_static_request",
    "decode_lc_timedependent_request",
    "encode_lc_static_request",
    "encode_lc_timedependent_request",
]
