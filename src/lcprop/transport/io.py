"""Codec-driven request/result bundle I/O without material branching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lcprop.runners.base import RunnerResult, WorkflowOperation
from lcprop.transport.artifacts import (
    array_manifest,
    verify_artifact_bundle,
    write_artifact_bundle,
)
from lcprop.transport.codecs import TransportCodec, TransportCodecRegistry
from lcprop.transport.envelopes import (
    FailureEnvelope,
    RequestEnvelope,
    ResultEnvelope,
    TransportCodecError,
    TransportVerificationError,
)


@dataclass(frozen=True)
class DecodedRequest:
    envelope: RequestEnvelope
    codec: TransportCodec
    request: Any


@dataclass(frozen=True)
class DecodedResult:
    envelope: ResultEnvelope
    codec: TransportCodec
    result: Any


def write_request_package(
    run_directory: str | Path,
    *,
    registry: TransportCodecRegistry,
    material_id: str,
    workflow_id: str,
    request: Any,
    run_id: str,
    provenance: Mapping[str, Any] | None = None,
    execution_target: str = "slurm",
    resource_profile: str | None = None,
) -> Path:
    codec = registry.codec(material_id, workflow_id)
    if not isinstance(request, codec.request_type):
        raise TypeError(
            f"codec {codec.request_codec_id!r} requires "
            f"{codec.request_type.__name__}"
        )
    encoded = codec.encode_request(request)
    arrays_name = "request_arrays.npz"
    envelope = RequestEnvelope(
        run_id=run_id,
        material_id=material_id,
        workflow_id=workflow_id,
        codec_id=codec.request_codec_id,
        codec_version=codec.request_codec_version,
        scientific_backend_requested=encoded.scientific_backend_requested,
        execution_target=execution_target,
        resource_profile=resource_profile,
        request_payload=encoded.payload.metadata,
        array_manifest=array_manifest(encoded.payload.arrays, filename=arrays_name),
        provenance={} if provenance is None else provenance,
    )
    return write_artifact_bundle(
        Path(run_directory) / "request",
        run_id=run_id,
        envelope_filename="request.json",
        envelope=envelope.to_dict(),
        arrays_filename=arrays_name,
        arrays=encoded.payload.arrays,
        ready_filename="READY.json",
    )


def read_request_package(
    run_directory: str | Path,
    *,
    registry: TransportCodecRegistry,
) -> DecodedRequest:
    value, arrays = verify_artifact_bundle(
        Path(run_directory) / "request",
        envelope_filename="request.json",
        arrays_filename="request_arrays.npz",
        ready_filename="READY.json",
    )
    envelope = RequestEnvelope.from_dict(value)
    codec = registry.codec(envelope.material_id, envelope.workflow_id)
    if (
        envelope.codec_id != codec.request_codec_id
        or envelope.codec_version != codec.request_codec_version
    ):
        raise TransportCodecError(
            "unsupported request codec "
            f"{envelope.codec_id!r} version {envelope.codec_version}"
        )
    request = codec.decode_request(envelope.request_payload, arrays)
    if not isinstance(request, codec.request_type):
        raise TransportCodecError("request codec reconstructed the wrong type")
    checked = codec.encode_request(request)
    if checked.scientific_backend_requested != envelope.scientific_backend_requested:
        raise TransportCodecError(
            "request backend disagrees with transport envelope"
        )
    return DecodedRequest(envelope=envelope, codec=codec, request=request)


def write_result_package(
    run_directory: str | Path,
    *,
    codec: TransportCodec,
    result: Any,
    request_envelope: RequestEnvelope,
    provenance: Mapping[str, Any] | None = None,
    output_directory: str | Path | None = None,
) -> Path:
    if not isinstance(result, codec.result_type):
        raise TypeError(
            f"codec {codec.result_codec_id!r} requires "
            f"{codec.result_type.__name__}"
        )
    if codec.key != (
        request_envelope.material_id,
        request_envelope.workflow_id,
    ):
        raise TransportCodecError("result codec identity disagrees with request")
    encoded = codec.encode_result(result)
    arrays_name = "result_arrays.npz"
    envelope = ResultEnvelope(
        run_id=request_envelope.run_id,
        material_id=codec.material_id,
        workflow_id=codec.workflow_id,
        codec_id=codec.result_codec_id,
        codec_version=codec.result_codec_version,
        scientific_status=encoded.scientific_status,
        converged=encoded.converged,
        cancelled=encoded.cancelled,
        termination_reason=encoded.termination_reason,
        scientific_backend_requested=(
            request_envelope.scientific_backend_requested
        ),
        scientific_backend_resolved=encoded.scientific_backend_resolved,
        device_summary=encoded.device_summary,
        result_payload=encoded.payload.metadata,
        array_manifest=array_manifest(encoded.payload.arrays, filename=arrays_name),
        provenance={} if provenance is None else provenance,
    )
    destination = (
        Path(output_directory)
        if output_directory is not None
        else Path(run_directory) / "output"
    )
    return write_artifact_bundle(
        destination,
        run_id=request_envelope.run_id,
        envelope_filename="result.json",
        envelope=envelope.to_dict(),
        arrays_filename=arrays_name,
        arrays=encoded.payload.arrays,
        ready_filename="RESULT_READY.json",
    )


def read_result_package(
    run_directory: str | Path,
    *,
    registry: TransportCodecRegistry,
) -> DecodedResult:
    request = read_request_package(run_directory, registry=registry)
    value, arrays = verify_artifact_bundle(
        Path(run_directory) / "output",
        envelope_filename="result.json",
        arrays_filename="result_arrays.npz",
        ready_filename="RESULT_READY.json",
    )
    envelope = ResultEnvelope.from_dict(value)
    if envelope.run_id != request.envelope.run_id:
        raise TransportVerificationError("request/result run_id mismatch")
    if (envelope.material_id, envelope.workflow_id) != request.codec.key:
        raise TransportVerificationError("request/result operation mismatch")
    codec = registry.codec(envelope.material_id, envelope.workflow_id)
    if (
        envelope.codec_id != codec.result_codec_id
        or envelope.codec_version != codec.result_codec_version
    ):
        raise TransportCodecError(
            "unsupported result codec "
            f"{envelope.codec_id!r} version {envelope.codec_version}"
        )
    requested_backend = request.envelope.scientific_backend_requested
    resolved_backend = envelope.scientific_backend_resolved
    if requested_backend != "auto" and resolved_backend != requested_backend:
        raise TransportVerificationError(
            "resolved scientific backend is incompatible with the request: "
            f"requested={requested_backend!r}, resolved={resolved_backend!r}"
        )
    result = codec.decode_result(envelope.result_payload, arrays)
    if not isinstance(result, codec.result_type):
        raise TransportCodecError("result codec reconstructed the wrong type")
    checked = codec.encode_result(result)
    common = (
        checked.scientific_status,
        checked.converged,
        checked.cancelled,
        checked.termination_reason,
        checked.scientific_backend_resolved,
    )
    recorded = (
        envelope.scientific_status,
        envelope.converged,
        envelope.cancelled,
        envelope.termination_reason,
        envelope.scientific_backend_resolved,
    )
    if common != recorded:
        raise TransportVerificationError(
            "result status/backend metadata disagrees with canonical result"
        )
    return DecodedResult(envelope=envelope, codec=codec, result=result)


def write_failure_package(
    run_directory: str | Path,
    failure: FailureEnvelope,
    *,
    output_directory: str | Path | None = None,
) -> Path:
    destination = (
        Path(output_directory)
        if output_directory is not None
        else Path(run_directory) / "output"
    )
    value = failure.to_dict()
    value["array_manifest"] = {}
    return write_artifact_bundle(
        destination,
        run_id=failure.run_id,
        envelope_filename="failure.json",
        envelope=value,
        arrays_filename="failure_arrays.npz",
        arrays={},
        ready_filename="RESULT_READY.json",
    )


def read_failure_package(run_directory: str | Path) -> FailureEnvelope:
    value, arrays = verify_artifact_bundle(
        Path(run_directory) / "output",
        envelope_filename="failure.json",
        arrays_filename="failure_arrays.npz",
        ready_filename="RESULT_READY.json",
    )
    if arrays:
        raise TransportVerificationError("failure package must not contain arrays")
    return FailureEnvelope.from_dict(value)


def runner_result_from_package(
    run_directory: str | Path,
    *,
    registry: TransportCodecRegistry,
    operations: Iterable[WorkflowOperation],
) -> RunnerResult:
    decoded = read_result_package(run_directory, registry=registry)
    matches = [operation for operation in operations if operation.key == decoded.codec.key]
    if len(matches) != 1:
        raise TransportCodecError(
            f"expected one registered operation for {decoded.codec.key!r}"
        )
    operation = matches[0]
    try:
        run_data = operation.to_run_data(decoded.result)
    except Exception as exc:
        raise TransportCodecError(f"product conversion failed: {exc}") from exc
    return RunnerResult(
        kind=operation.workflow_id,
        result=decoded.result,
        message="Completed remotely",
        run_data=run_data,
        material_id=operation.material_id,
    )


__all__ = [
    "DecodedRequest",
    "DecodedResult",
    "read_failure_package",
    "read_request_package",
    "read_result_package",
    "runner_result_from_package",
    "write_failure_package",
    "write_request_package",
    "write_result_package",
]
