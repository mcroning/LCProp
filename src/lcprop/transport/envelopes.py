"""Versioned JSON envelope contracts for remote scientific transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


TRANSPORT_SCHEMA_VERSION = 1
FAILURE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
REQUEST_FORMAT = "lcprop-remote-request"
RESULT_FORMAT = "lcprop-remote-result"
FAILURE_FORMAT = "lcprop-remote-failure"
MANIFEST_FORMAT = "lcprop-artifact-manifest"

FAILURE_CATEGORIES = frozenset({
    "submission_failed",
    "scheduler_failed",
    "environment_failed",
    "scientific_process_failed",
    "result_missing",
    "result_retrieval_failed",
    "checksum_verification_failed",
    "result_deserialization_failed",
    "product_conversion_failed",
    "cancelled",
    "timeout",
    "out_of_memory",
})


class TransportError(ValueError):
    """Base class for portable transport failures."""


class TransportFormatError(TransportError):
    pass


class TransportSchemaError(TransportError):
    pass


class TransportVerificationError(TransportError):
    pass


class TransportCodecError(TransportError):
    pass


def _required(value: Mapping[str, Any], names: set[str], *, label: str) -> None:
    missing = names - set(value)
    if missing:
        raise TransportFormatError(
            f"{label} is missing required field(s): {', '.join(sorted(missing))}"
        )


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TransportFormatError(f"{name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True)
class RequestEnvelope:
    run_id: str
    material_id: str
    workflow_id: str
    codec_id: str
    codec_version: int
    scientific_backend_requested: str
    request_payload: Mapping[str, Any]
    array_manifest: Mapping[str, Any] = field(default_factory=dict)
    execution_target: str = "slurm"
    resource_profile: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    transport_schema_version: int = TRANSPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REQUEST_FORMAT,
            "transport_schema_version": self.transport_schema_version,
            "run_id": self.run_id,
            "material_id": self.material_id,
            "workflow_id": self.workflow_id,
            "codec_id": self.codec_id,
            "codec_version": self.codec_version,
            "scientific_backend_requested": self.scientific_backend_requested,
            "execution_target": self.execution_target,
            "resource_profile": self.resource_profile,
            "request_payload": dict(self.request_payload),
            "array_manifest": dict(self.array_manifest),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RequestEnvelope":
        required = {
            "format", "transport_schema_version", "run_id", "material_id",
            "workflow_id", "codec_id", "codec_version",
            "scientific_backend_requested", "request_payload",
            "array_manifest", "provenance",
        }
        _required(value, required, label="request envelope")
        if value["format"] != REQUEST_FORMAT:
            raise TransportFormatError("invalid request format marker")
        if value["transport_schema_version"] != TRANSPORT_SCHEMA_VERSION:
            raise TransportSchemaError(
                "unsupported request transport schema version: "
                f"{value['transport_schema_version']!r}"
            )
        version = value["codec_version"]
        if type(version) is not int or version < 1:
            raise TransportFormatError("codec_version must be a positive integer")
        for name in ("request_payload", "array_manifest", "provenance"):
            if not isinstance(value[name], Mapping):
                raise TransportFormatError(f"{name} must be a JSON object")
        target = value.get("execution_target", "slurm")
        if target not in {"local", "slurm"}:
            raise TransportFormatError("invalid execution_target")
        backend = value["scientific_backend_requested"]
        if backend not in {"numpy", "auto", "cupy"}:
            raise TransportFormatError("invalid requested scientific backend")
        return cls(
            run_id=_identifier("run_id", value["run_id"]),
            material_id=_identifier("material_id", value["material_id"]),
            workflow_id=_identifier("workflow_id", value["workflow_id"]),
            codec_id=_identifier("codec_id", value["codec_id"]),
            codec_version=version,
            scientific_backend_requested=backend,
            execution_target=target,
            resource_profile=value.get("resource_profile"),
            request_payload=dict(value["request_payload"]),
            array_manifest=dict(value["array_manifest"]),
            provenance=dict(value["provenance"]),
        )


@dataclass(frozen=True)
class ResultEnvelope:
    run_id: str
    material_id: str
    workflow_id: str
    codec_id: str
    codec_version: int
    scientific_status: str
    converged: bool | None
    cancelled: bool
    termination_reason: str | None
    scientific_backend_requested: str
    scientific_backend_resolved: str
    result_payload: Mapping[str, Any]
    array_manifest: Mapping[str, Any] = field(default_factory=dict)
    device_summary: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    transport_schema_version: int = TRANSPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RESULT_FORMAT,
            "transport_schema_version": self.transport_schema_version,
            "run_id": self.run_id,
            "material_id": self.material_id,
            "workflow_id": self.workflow_id,
            "codec_id": self.codec_id,
            "codec_version": self.codec_version,
            "scientific_status": self.scientific_status,
            "converged": self.converged,
            "cancelled": self.cancelled,
            "termination_reason": self.termination_reason,
            "scientific_backend_requested": self.scientific_backend_requested,
            "scientific_backend_resolved": self.scientific_backend_resolved,
            "device_summary": None if self.device_summary is None else dict(self.device_summary),
            "result_payload": dict(self.result_payload),
            "array_manifest": dict(self.array_manifest),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResultEnvelope":
        required = {
            "format", "transport_schema_version", "run_id", "material_id",
            "workflow_id", "codec_id", "codec_version", "scientific_status",
            "converged", "cancelled", "termination_reason",
            "scientific_backend_requested", "scientific_backend_resolved",
            "result_payload", "array_manifest", "provenance",
        }
        _required(value, required, label="result envelope")
        if value["format"] != RESULT_FORMAT:
            raise TransportFormatError("invalid result format marker")
        if value["transport_schema_version"] != TRANSPORT_SCHEMA_VERSION:
            raise TransportSchemaError(
                "unsupported result transport schema version: "
                f"{value['transport_schema_version']!r}"
            )
        version = value["codec_version"]
        if type(version) is not int or version < 1:
            raise TransportFormatError("codec_version must be a positive integer")
        for name in ("result_payload", "array_manifest", "provenance"):
            if not isinstance(value[name], Mapping):
                raise TransportFormatError(f"{name} must be a JSON object")
        requested = value["scientific_backend_requested"]
        resolved = value["scientific_backend_resolved"]
        if requested not in {"numpy", "auto", "cupy"}:
            raise TransportFormatError("invalid requested scientific backend")
        if resolved not in {"numpy", "cupy"}:
            raise TransportFormatError("invalid resolved scientific backend")
        if value["converged"] is not None and type(value["converged"]) is not bool:
            raise TransportFormatError("converged must be boolean or null")
        if type(value["cancelled"]) is not bool:
            raise TransportFormatError("cancelled must be boolean")
        return cls(
            run_id=_identifier("run_id", value["run_id"]),
            material_id=_identifier("material_id", value["material_id"]),
            workflow_id=_identifier("workflow_id", value["workflow_id"]),
            codec_id=_identifier("codec_id", value["codec_id"]),
            codec_version=version,
            scientific_status=_identifier("scientific_status", value["scientific_status"]),
            converged=value["converged"],
            cancelled=value["cancelled"],
            termination_reason=value["termination_reason"],
            scientific_backend_requested=requested,
            scientific_backend_resolved=resolved,
            device_summary=value.get("device_summary"),
            result_payload=dict(value["result_payload"]),
            array_manifest=dict(value["array_manifest"]),
            provenance=dict(value["provenance"]),
        )


@dataclass(frozen=True)
class FailureEnvelope:
    run_id: str
    material_id: str
    workflow_id: str
    failure_category: str
    message: str
    exception_type: str | None = None
    traceback: str | None = None
    remote_job_id: str | None = None
    scheduler_state: str | None = None
    exit_code: str | None = None
    stderr_reference: str | None = None
    remote_artifact_location: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    failure_schema_version: int = FAILURE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FAILURE_FORMAT,
            "failure_schema_version": self.failure_schema_version,
            "run_id": self.run_id,
            "material_id": self.material_id,
            "workflow_id": self.workflow_id,
            "remote_job_id": self.remote_job_id,
            "scheduler_state": self.scheduler_state,
            "exit_code": self.exit_code,
            "failure_category": self.failure_category,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "stderr_reference": self.stderr_reference,
            "remote_artifact_location": self.remote_artifact_location,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureEnvelope":
        required = {
            "format", "failure_schema_version", "run_id", "material_id",
            "workflow_id", "failure_category", "message", "provenance",
        }
        _required(value, required, label="failure envelope")
        if value["format"] != FAILURE_FORMAT:
            raise TransportFormatError("invalid failure format marker")
        if value["failure_schema_version"] != FAILURE_SCHEMA_VERSION:
            raise TransportSchemaError(
                "unsupported failure schema version: "
                f"{value['failure_schema_version']!r}"
            )
        category = value["failure_category"]
        if category not in FAILURE_CATEGORIES:
            raise TransportFormatError(f"invalid failure category {category!r}")
        if not isinstance(value["provenance"], Mapping):
            raise TransportFormatError("provenance must be a JSON object")
        return cls(
            run_id=_identifier("run_id", value["run_id"]),
            material_id=_identifier("material_id", value["material_id"]),
            workflow_id=_identifier("workflow_id", value["workflow_id"]),
            failure_category=category,
            message=str(value["message"]),
            exception_type=value.get("exception_type"),
            traceback=value.get("traceback"),
            remote_job_id=value.get("remote_job_id"),
            scheduler_state=value.get("scheduler_state"),
            exit_code=value.get("exit_code"),
            stderr_reference=value.get("stderr_reference"),
            remote_artifact_location=value.get("remote_artifact_location"),
            provenance=dict(value["provenance"]),
        )


__all__ = [
    "FAILURE_CATEGORIES",
    "FAILURE_FORMAT",
    "FAILURE_SCHEMA_VERSION",
    "FailureEnvelope",
    "MANIFEST_FORMAT",
    "MANIFEST_SCHEMA_VERSION",
    "REQUEST_FORMAT",
    "RESULT_FORMAT",
    "RequestEnvelope",
    "ResultEnvelope",
    "TRANSPORT_SCHEMA_VERSION",
    "TransportCodecError",
    "TransportError",
    "TransportFormatError",
    "TransportSchemaError",
    "TransportVerificationError",
]
