"""LC-owned portable transport codec for the canonical static workflow."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.core.context import GridSpec
from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.persistence.static import StaticCheckpoint, validate_static_checkpoint
from lcprop.lc.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
)
from lcprop.lc.results import StaticIterationRecord, StaticRunResult, StaticSliceSummary
from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.persistence.experiments import decode_beam_stack, encode_beam_stack
from lcprop.transport.codecs import (
    EncodedRequest,
    EncodedResult,
    PortablePayload,
    TransportCodec,
)
from lcprop.transport.envelopes import TransportCodecError


LC_STATIC_REQUEST_CODEC_ID = "lc.static.request"
LC_STATIC_RESULT_CODEC_ID = "lc.static.result"
LC_STATIC_TRANSPORT_CODEC_VERSION = 1


def _portable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_portable(item) for item in value]
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _portable(item) for key, item in value.items()}
    return value


def _validate_request(request: StaticRunRequest) -> None:
    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()
    request.runtime.validate()
    if request.output.run_dir is not None:
        raise TransportCodecError(
            "remote LC static transport requires output.run_dir=None"
        )


def _encode_request_parts(
    request: StaticRunRequest,
    *,
    prefix: str = "",
) -> PortablePayload:
    if not isinstance(request, StaticRunRequest):
        raise TypeError("request must be a StaticRunRequest")
    _validate_request(request)
    arrays: dict[str, np.ndarray] = {}
    refs: dict[str, str | None] = {}
    for name in ("initial_A", "initial_theta"):
        value = getattr(request, name)
        key = f"{prefix}{name}"
        refs[name] = None if value is None else key
        if value is not None:
            arrays[key] = np.asarray(asnumpy(value)).copy()
    metadata = {
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
            "save_slices": bool(request.output.save_slices),
            "save_full": bool(request.output.save_full),
        },
        "runtime": asdict(request.runtime),
        "initial_arrays": refs,
    }
    return PortablePayload(_portable(metadata), arrays)


def _require_fields(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    if missing:
        raise TransportCodecError(
            f"{label} is missing: {', '.join(sorted(missing))}"
        )


def _decode_request_parts(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> StaticRunRequest:
    required = {
        "grid", "material", "bias", "beams", "solver", "output",
        "runtime", "initial_arrays",
    }
    _require_fields(metadata, required, "LC static request payload")
    solver_values = dict(metadata["solver"])
    workflow = StaticWorkflowOptions(**solver_values.pop("workflow"))
    output_values = dict(metadata["output"])
    if output_values.get("run_dir") is not None:
        raise TransportCodecError("transported LC output.run_dir must be null")
    refs = dict(metadata["initial_arrays"])

    def optional_array(name: str):
        key = refs.get(name)
        if key is None:
            return None
        if key not in arrays:
            raise TransportCodecError(f"missing transported LC array {key!r}")
        return np.asarray(arrays[key]).copy()

    try:
        request = StaticRunRequest(
            grid=GridSpec(**metadata["grid"]),
            material=LCMaterial(**metadata["material"]),
            bias=BiasSpec(**metadata["bias"]),
            beams=decode_beam_stack(metadata["beams"]),
            solver=StaticSolverOptions(workflow=workflow, **solver_values),
            output=OutputOptions(**output_values),
            runtime=RuntimeOptions(**metadata["runtime"]),
            initial_A=optional_array("initial_A"),
            initial_theta=optional_array("initial_theta"),
        )
        _validate_request(request)
    except TransportCodecError:
        raise
    except Exception as exc:
        raise TransportCodecError(f"invalid LC static request payload: {exc}") from exc
    return request


def encode_lc_static_transport_request(request: StaticRunRequest) -> EncodedRequest:
    return EncodedRequest(
        payload=_encode_request_parts(request),
        scientific_backend_requested="numpy",
    )


def decode_lc_static_transport_request(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> StaticRunRequest:
    return _decode_request_parts(metadata, arrays)


def _add_array(
    arrays: dict[str, np.ndarray],
    refs: dict[str, str | None],
    name: str,
    value: Any,
) -> None:
    refs[name] = None if value is None else name
    if value is not None:
        arrays[name] = np.asarray(asnumpy(value)).copy()


def encode_lc_static_transport_result(result: StaticRunResult) -> EncodedResult:
    if not isinstance(result, StaticRunResult):
        raise TypeError("result must be a StaticRunResult")
    arrays: dict[str, np.ndarray] = {}
    refs: dict[str, str | None] = {}
    for name in (
        "A_final", "theta_final", "theta_bias", "A_initial",
        "intensity_stack", "theta_intensity_stack",
    ):
        _add_array(arrays, refs, name, getattr(result, name))

    result_request_payload = None
    if result.request is not None:
        encoded_request = _encode_request_parts(result.request, prefix="result_request__")
        result_request_payload = encoded_request.metadata
        arrays.update(encoded_request.arrays)

    checkpoint_payload = None
    if result.checkpoint is not None:
        checkpoint = result.checkpoint
        if not isinstance(checkpoint, StaticCheckpoint):
            raise TransportCodecError("LC static result checkpoint has wrong type")
        validate_static_checkpoint(checkpoint)
        checkpoint_refs: dict[str, str | None] = {}
        for name in (
            "A_next", "theta_seed", "theta_stack", "intensity_stack",
            "theta_intensity_stack",
        ):
            key = f"checkpoint__{name}"
            value = getattr(checkpoint, name)
            checkpoint_refs[name] = None if value is None else key
            if value is not None:
                arrays[key] = np.asarray(asnumpy(value)).copy()
        checkpoint_payload = {
            "next_slice_index": checkpoint.next_slice_index,
            "completed_slices": checkpoint.completed_slices,
            "z_reached_um": checkpoint.z_reached_um,
            "array_refs": checkpoint_refs,
            "slice_summaries": [asdict(item) for item in checkpoint.slice_summaries],
            "iteration_records": [asdict(item) for item in checkpoint.iteration_records],
            "relax_steps": checkpoint.relax_steps,
            "grid_summary": checkpoint.grid_summary,
            "launch_summary": checkpoint.launch_summary,
            "normalized_power_initial": checkpoint.normalized_power_initial,
            "physical_power_initial_mW": checkpoint.physical_power_initial_mW,
            "A_dtype": checkpoint.A_dtype,
            "theta_dtype": checkpoint.theta_dtype,
            "status": checkpoint.status,
            "request_fingerprint": checkpoint.request_fingerprint,
            "schema_version": checkpoint.schema_version,
            "workflow": checkpoint.workflow,
        }

    metadata = {
        "array_refs": refs,
        "power_initial": result.power_initial,
        "power_final": result.power_final,
        "grid_summary": result.grid_summary,
        "launch_summary": result.launch_summary,
        "bias_summary": result.bias_summary,
        "n_steps": result.n_steps,
        "method": result.method,
        "physical_power_initial_mW": result.physical_power_initial_mW,
        "physical_power_final_mW": result.physical_power_final_mW,
        "coupling_summary": result.coupling_summary,
        "iteration_records": [asdict(item) for item in result.iteration_records],
        "slice_summaries": [asdict(item) for item in result.slice_summaries],
        "all_slices_converged": result.all_slices_converged,
        "max_final_residual_rms": result.max_final_residual_rms,
        "median_final_residual_rms": result.median_final_residual_rms,
        "rms_over_z_final_residual": result.rms_over_z_final_residual,
        "max_final_residual_max": result.max_final_residual_max,
        "worst_slice_index": result.worst_slice_index,
        "warnings": list(result.warnings),
        "status": result.status,
        "completed_slices": result.completed_slices,
        "total_slices": result.total_slices,
        "z_reached_um": result.z_reached_um,
        "provenance": result.provenance,
        "result_request": result_request_payload,
        "checkpoint": checkpoint_payload,
    }
    converged = (
        result.all_slices_converged
        if result.all_slices_converged is not None
        else result.status == "completed"
    )
    return EncodedResult(
        payload=PortablePayload(_portable(metadata), arrays),
        scientific_status=result.status,
        converged=bool(converged),
        cancelled=result.status in {"stopped", "cancelled"},
        termination_reason=result.status,
        scientific_backend_resolved="numpy",
    )


def decode_lc_static_transport_result(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> StaticRunResult:
    required = {
        "array_refs", "power_initial", "power_final", "grid_summary",
        "launch_summary", "bias_summary", "n_steps", "method",
        "physical_power_initial_mW", "physical_power_final_mW",
        "coupling_summary", "iteration_records", "slice_summaries",
        "all_slices_converged", "max_final_residual_rms",
        "median_final_residual_rms", "rms_over_z_final_residual",
        "max_final_residual_max", "worst_slice_index", "warnings", "status",
        "completed_slices", "total_slices", "z_reached_um", "provenance",
        "result_request", "checkpoint",
    }
    _require_fields(metadata, required, "LC static result payload")
    refs = dict(metadata["array_refs"])

    def result_array(name: str, *, required_value: bool = False):
        key = refs.get(name)
        if key is None:
            if required_value:
                raise TransportCodecError(f"LC static result lacks {name}")
            return None
        if key not in arrays:
            raise TransportCodecError(f"missing transported LC array {key!r}")
        return np.asarray(arrays[key]).copy()

    result_request = None
    if metadata["result_request"] is not None:
        result_request = _decode_request_parts(metadata["result_request"], arrays)

    checkpoint = None
    checkpoint_values = metadata["checkpoint"]
    if checkpoint_values is not None:
        if result_request is None:
            raise TransportCodecError("LC checkpoint requires a canonical result request")
        checkpoint_refs = dict(checkpoint_values["array_refs"])

        def checkpoint_array(name: str, *, optional: bool = False):
            key = checkpoint_refs.get(name)
            if key is None:
                if optional:
                    return None
                raise TransportCodecError(f"LC checkpoint lacks {name}")
            if key not in arrays:
                raise TransportCodecError(f"missing transported LC checkpoint array {key!r}")
            return np.asarray(arrays[key]).copy()

        checkpoint = StaticCheckpoint(
            request=result_request,
            next_slice_index=int(checkpoint_values["next_slice_index"]),
            completed_slices=int(checkpoint_values["completed_slices"]),
            z_reached_um=float(checkpoint_values["z_reached_um"]),
            A_next=checkpoint_array("A_next"),
            theta_seed=checkpoint_array("theta_seed"),
            theta_stack=checkpoint_array("theta_stack"),
            intensity_stack=checkpoint_array("intensity_stack"),
            theta_intensity_stack=checkpoint_array(
                "theta_intensity_stack", optional=True
            ),
            slice_summaries=tuple(
                StaticSliceSummary(**item)
                for item in checkpoint_values["slice_summaries"]
            ),
            iteration_records=tuple(
                StaticIterationRecord(**item)
                for item in checkpoint_values["iteration_records"]
            ),
            relax_steps=int(checkpoint_values["relax_steps"]),
            grid_summary=dict(checkpoint_values["grid_summary"]),
            launch_summary=dict(checkpoint_values["launch_summary"]),
            normalized_power_initial=float(
                checkpoint_values["normalized_power_initial"]
            ),
            physical_power_initial_mW=float(
                checkpoint_values["physical_power_initial_mW"]
            ),
            A_dtype=str(checkpoint_values["A_dtype"]),
            theta_dtype=str(checkpoint_values["theta_dtype"]),
            status=str(checkpoint_values["status"]),
            request_fingerprint=str(checkpoint_values["request_fingerprint"]),
            schema_version=int(checkpoint_values["schema_version"]),
            workflow=str(checkpoint_values["workflow"]),
        )
        validate_static_checkpoint(checkpoint)

    try:
        result = StaticRunResult(
            A_final=result_array("A_final", required_value=True),
            theta_final=result_array("theta_final", required_value=True),
            theta_bias=result_array("theta_bias", required_value=True),
            power_initial=float(metadata["power_initial"]),
            power_final=float(metadata["power_final"]),
            grid_summary=dict(metadata["grid_summary"]),
            launch_summary=dict(metadata["launch_summary"]),
            bias_summary=dict(metadata["bias_summary"]),
            n_steps=int(metadata["n_steps"]),
            method=str(metadata["method"]),
            physical_power_initial_mW=metadata["physical_power_initial_mW"],
            physical_power_final_mW=metadata["physical_power_final_mW"],
            coupling_summary=dict(metadata["coupling_summary"]),
            A_initial=result_array("A_initial"),
            intensity_stack=result_array("intensity_stack"),
            theta_intensity_stack=result_array("theta_intensity_stack"),
            iteration_records=tuple(
                StaticIterationRecord(**item) for item in metadata["iteration_records"]
            ),
            slice_summaries=tuple(
                StaticSliceSummary(**item) for item in metadata["slice_summaries"]
            ),
            all_slices_converged=metadata["all_slices_converged"],
            max_final_residual_rms=metadata["max_final_residual_rms"],
            median_final_residual_rms=metadata["median_final_residual_rms"],
            rms_over_z_final_residual=metadata["rms_over_z_final_residual"],
            max_final_residual_max=metadata["max_final_residual_max"],
            worst_slice_index=metadata["worst_slice_index"],
            warnings=tuple(metadata["warnings"]),
            status=str(metadata["status"]),
            completed_slices=int(metadata["completed_slices"]),
            total_slices=int(metadata["total_slices"]),
            z_reached_um=float(metadata["z_reached_um"]),
            checkpoint=checkpoint,
            request=result_request,
            provenance=dict(metadata["provenance"]),
        )
    except Exception as exc:
        raise TransportCodecError(f"invalid LC static result payload: {exc}") from exc
    return result


LC_STATIC_TRANSPORT_CODEC = TransportCodec(
    material_id=LC_MATERIAL_ID,
    workflow_id="static",
    request_codec_id=LC_STATIC_REQUEST_CODEC_ID,
    request_codec_version=LC_STATIC_TRANSPORT_CODEC_VERSION,
    request_type=StaticRunRequest,
    encode_request=encode_lc_static_transport_request,
    decode_request=decode_lc_static_transport_request,
    result_codec_id=LC_STATIC_RESULT_CODEC_ID,
    result_codec_version=LC_STATIC_TRANSPORT_CODEC_VERSION,
    result_type=StaticRunResult,
    encode_result=encode_lc_static_transport_result,
    decode_result=decode_lc_static_transport_result,
)


__all__ = [
    "LC_STATIC_REQUEST_CODEC_ID",
    "LC_STATIC_RESULT_CODEC_ID",
    "LC_STATIC_TRANSPORT_CODEC",
    "LC_STATIC_TRANSPORT_CODEC_VERSION",
    "decode_lc_static_transport_request",
    "decode_lc_static_transport_result",
    "encode_lc_static_transport_request",
    "encode_lc_static_transport_result",
]
