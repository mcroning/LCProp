"""PR-owned portable transport codec for the reduced TD workflow."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.core.grid import round_nz
from lcprop.optics.launch_configuration import reject_prepared_launch_conflict
from lcprop.optics.screens import validate_channel_launch_elements
from lcprop.persistence.experiments import decode_beam_stack, encode_beam_stack
from lcprop.pr.checkpoint import PRTimeDependentCheckpoint, validate_pr_checkpoint
from lcprop.pr.portable_launch import (
    PortableLaunchPayloadError,
    decode_launch_elements,
    encode_launch_elements,
)
from lcprop.pr.longitudinal_cuts import (
    extract_longitudinal_optical_intensity_cuts,
    fast_retention_summary,
    retained_longitudinal_intensity_cuts,
    validate_longitudinal_cut_coordinates,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
    PR_MATERIAL_ID,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transport_common import pack_portable, unpack_portable
from lcprop.transport.codecs import (
    EncodedRequest,
    EncodedResult,
    PortablePayload,
    TransportCodec,
)
from lcprop.transport.envelopes import TransportCodecError
from lcprop.transport.result_policy import (
    FAST_RESULT_POLICY,
    FULL_RESULT_POLICY,
    normalize_result_policy,
)


PR_TIMEDEPENDENT_REQUEST_CODEC_ID = "pr.timedependent.request"
PR_TIMEDEPENDENT_RESULT_CODEC_ID = "pr.timedependent.result"
PR_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION = 1


def _validate_request(request: PRRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    validate_channel_launch_elements(
        request.launch_elements, n_channels=len(request.beams.channels)
    )
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    expected_A = (len(request.beams.channels), request.grid.Nx, request.grid.Ny)
    expected_E = (
        round_nz(request.grid.z_length_um, request.grid.dz_um),
        request.grid.Nx,
        request.grid.Ny,
    )
    if request.initial_A is not None and request.initial_A.shape != expected_A:
        raise ValueError(f"initial_A shape must be {expected_A}")
    if request.initial_E is not None and request.initial_E.shape != expected_E:
        raise ValueError(f"initial_E shape must be {expected_E}")
    if request.scattering is not None:
        request.scattering.validate()


def _encode_request_metadata(
    request: PRRunRequest,
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
) -> dict[str, Any]:
    _validate_request(request)
    return {
        "grid": asdict(request.grid),
        "beams": encode_beam_stack(request.beams),
        "material": asdict(request.material),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "launch_elements": encode_launch_elements(request.launch_elements),
        "initial_A": pack_portable(
            request.initial_A, arrays, f"{prefix}.initial_A"
        ),
        "initial_E": pack_portable(
            request.initial_E, arrays, f"{prefix}.initial_E"
        ),
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
    }


def _decode_request_metadata(
    value: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRRunRequest:
    values = unpack_portable(dict(value), arrays)
    beams = decode_beam_stack(values["beams"])
    scattering = values["scattering"]
    request = PRRunRequest(
        grid=GridSpec(**values["grid"]),
        beams=beams,
        material=PRMaterialSpec(**values["material"]),
        solver=PRSolverOptions(**values["solver"]),
        backend=BackendSpec(**values["backend"]),
        launch_elements=decode_launch_elements(
            values.get("launch_elements", []),
            n_channels=len(beams.channels),
        ),
        initial_A=values["initial_A"],
        initial_E=values["initial_E"],
        scattering=(
            None
            if scattering is None
            else PRCanonicalScatteringSpec(**scattering)
        ),
    )
    _validate_request(request)
    return request


def encode_pr_timedependent_transport_request(
    request: PRRunRequest,
) -> EncodedRequest:
    """Encode one canonical reduced time-dependent request."""

    if not isinstance(request, PRRunRequest):
        raise TypeError("request must be a PRRunRequest")
    arrays: dict[str, np.ndarray] = {}
    metadata = _encode_request_metadata(request, arrays, prefix="request")
    return EncodedRequest(
        PortablePayload(metadata, arrays), request.backend.backend
    )


def decode_pr_timedependent_transport_request(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRRunRequest:
    """Decode and validate one canonical reduced TD request."""

    try:
        return _decode_request_metadata(metadata, arrays)
    except TransportCodecError:
        raise
    except (PortableLaunchPayloadError, KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            f"invalid PR time-dependent request payload: {exc}"
        ) from exc


def _encode_checkpoint(
    checkpoint: PRTimeDependentCheckpoint,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    validate_pr_checkpoint(checkpoint)
    return {
        "request": _encode_request_metadata(
            checkpoint.request, arrays, prefix="checkpoint.request"
        ),
        "E_initial": pack_portable(
            checkpoint.E_initial, arrays, "checkpoint.E_initial"
        ),
        "E_current": pack_portable(
            checkpoint.E_current, arrays, "checkpoint.E_current"
        ),
        "A0": pack_portable(checkpoint.A0, arrays, "checkpoint.A0"),
        "completed_steps": int(checkpoint.completed_steps),
        "requested_steps": int(checkpoint.requested_steps),
        "time_normalized": float(checkpoint.time_normalized),
        "grid_summary": pack_portable(
            checkpoint.grid_summary, arrays, "checkpoint.grid_summary"
        ),
        "E_dtype": checkpoint.E_dtype,
        "A0_dtype": checkpoint.A0_dtype,
        "status": checkpoint.status,
    }


def _decode_checkpoint(
    value: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRTimeDependentCheckpoint:
    values = unpack_portable(dict(value), arrays)
    checkpoint = PRTimeDependentCheckpoint(
        request=_decode_request_metadata(values["request"], arrays),
        E_initial=values["E_initial"],
        E_current=values["E_current"],
        A0=values["A0"],
        completed_steps=values["completed_steps"],
        requested_steps=values["requested_steps"],
        time_normalized=float(values["time_normalized"]),
        grid_summary=dict(values["grid_summary"]),
        E_dtype=values["E_dtype"],
        A0_dtype=values["A0_dtype"],
        status=values["status"],
    )
    validate_pr_checkpoint(checkpoint)
    return checkpoint


_FAST_OMITTED_FIELDS = (
    "E_initial",
    "E_final",
    "source_intensity_stack",
    "checkpoint",
)


def encode_pr_timedependent_transport_result(
    result: PRRunResult,
    result_policy: str = FULL_RESULT_POLICY,
) -> EncodedResult:
    """Encode the complete canonical reduced-TD result and checkpoint."""

    if not isinstance(result, PRRunResult):
        raise TypeError("result must be a PRRunResult")
    policy = normalize_result_policy(result_policy)
    if policy == FULL_RESULT_POLICY and not isinstance(
        result.checkpoint, PRTimeDependentCheckpoint
    ):
        raise ValueError("PR time-dependent result must contain a checkpoint")
    arrays: dict[str, np.ndarray] = {}
    cuts = None
    if policy == FAST_RESULT_POLICY:
        if result.source_intensity_stack is None:
            try:
                cuts = retained_longitudinal_intensity_cuts(result)
            except ValueError:
                cuts = None
        else:
            request = getattr(result.checkpoint, "request", None)
            if request is None:
                raise ValueError(
                    "PR time-dependent Fast retention requires checkpoint request provenance"
                )
            cuts = extract_longitudinal_optical_intensity_cuts(
                result.source_intensity_stack,
                grid_summary=result.grid_summary,
                peak_intensity_reference=channel_peak_intensity_reference(
                    np.asarray(result.A_initial), xp=np
                ),
                background_intensity=float(request.material.background_intensity),
            )
    metadata = {
        "A_initial": pack_portable(result.A_initial, arrays, "result.A_initial"),
        "A_final": pack_portable(result.A_final, arrays, "result.A_final"),
        "E_initial": None,
        "E_final": None,
        "source_intensity_stack": None,
        "longitudinal_intensity_xz": (
            None if cuts is None else pack_portable(
                cuts.xz, arrays, "result.longitudinal_intensity_xz"
            )
        ),
        "longitudinal_intensity_yz": (
            None if cuts is None else pack_portable(
                cuts.yz, arrays, "result.longitudinal_intensity_yz"
            )
        ),
        "x_cut_um": None if cuts is None else cuts.x_cut_um,
        "y_cut_um": None if cuts is None else cuts.y_cut_um,
        "power_initial": float(result.power_initial),
        "power_final": float(result.power_final),
        "completed_steps": int(result.completed_steps),
        "time_normalized": float(result.time_normalized),
        "grid_summary": pack_portable(
            result.grid_summary, arrays, "result.grid_summary"
        ),
        "launch_summary": pack_portable(
            result.launch_summary, arrays, "result.launch_summary"
        ),
        "status": result.status,
        "requested_steps": int(result.requested_steps),
        "checkpoint": None,
        "diagnostics": pack_portable(
            result.diagnostics, arrays, "result.diagnostics"
        ),
        "retention_summary": (
            fast_retention_summary(_FAST_OMITTED_FIELDS, cuts)
            if cuts is not None
            else {
                "policy": policy,
                "omitted_fields": (
                    list(_FAST_OMITTED_FIELDS)
                    if policy == FAST_RESULT_POLICY else []
                ),
            }
        ),
    }
    if policy == FULL_RESULT_POLICY:
        metadata.update({
            "E_initial": pack_portable(
                result.E_initial, arrays, "result.E_initial"
            ),
            "E_final": pack_portable(result.E_final, arrays, "result.E_final"),
            "source_intensity_stack": pack_portable(
                result.source_intensity_stack,
                arrays,
                "result.source_intensity_stack",
            ),
            "checkpoint": _encode_checkpoint(result.checkpoint, arrays),
        })
    backend_summary = result.diagnostics.get("backend", {})
    backend = str(backend_summary.get("backend", "unknown"))
    cancelled = result.status == "cancelled"
    return EncodedResult(
        PortablePayload(metadata, arrays),
        scientific_status=result.status,
        converged=None,
        cancelled=cancelled,
        termination_reason=(
            "cancelled_at_accepted_material_boundary"
            if cancelled
            else "requested_material_steps_completed"
        ),
        scientific_backend_resolved=backend,
        device_summary=pack_portable(
            backend_summary, arrays, "device_summary"
        ),
        result_policy=policy,
    )


def _positive_dimension(summary: Mapping[str, Any], name: str) -> int:
    value = summary.get(name)
    if type(value) is not int or value < 1:
        raise TransportCodecError(
            f"PR time-dependent result {name} must be a positive integer"
        )
    return value


def _require_array(
    values: Mapping[str, Any], name: str, shape: tuple[int, ...], kind: str
) -> np.ndarray:
    value = values.get(name)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise TransportCodecError(
            f"PR time-dependent result {name} must be a numeric array"
        )
    if value.shape != shape:
        raise TransportCodecError(
            f"PR time-dependent result {name} shape {value.shape} "
            f"does not match {shape}"
        )
    if kind == "complex" and value.dtype.kind != "c":
        raise TransportCodecError(
            f"PR time-dependent result {name} must be complex"
        )
    if kind == "real" and value.dtype.kind != "f":
        raise TransportCodecError(
            f"PR time-dependent result {name} must be real floating point"
        )
    return value


def _validate_result(values: Mapping[str, Any]) -> None:
    grid = values.get("grid_summary")
    launch = values.get("launch_summary")
    if not isinstance(grid, Mapping) or not isinstance(launch, Mapping):
        raise TransportCodecError(
            "PR time-dependent result lacks grid or launch summary"
        )
    nx = _positive_dimension(grid, "Nx")
    ny = _positive_dimension(grid, "Ny")
    nz = _positive_dimension(grid, "Nz")
    nch = _positive_dimension(launch, "Nch")
    _require_array(values, "A_initial", (nch, nx, ny), "complex")
    _require_array(values, "A_final", (nch, nx, ny), "complex")
    retention = values.get("retention_summary", {
        "policy": FULL_RESULT_POLICY,
        "omitted_fields": [],
    })
    if not isinstance(retention, Mapping):
        raise TransportCodecError("PR time-dependent retention summary is invalid")
    try:
        policy = normalize_result_policy(retention.get("policy", FULL_RESULT_POLICY))
    except ValueError as exc:
        raise TransportCodecError(str(exc)) from exc
    expected_omitted = set(_FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ())
    omitted = retention.get("omitted_fields", [])
    if not isinstance(omitted, (tuple, list)) or set(omitted) != expected_omitted:
        raise TransportCodecError(
            "PR time-dependent omitted fields disagree with result policy"
        )
    cuts_present = validate_longitudinal_cut_coordinates(values, grid)
    if policy == FULL_RESULT_POLICY:
        _require_array(values, "E_initial", (nz, nx, ny), "real")
        _require_array(values, "E_final", (nz, nx, ny), "real")
        _require_array(
            values, "source_intensity_stack", (nz, nx, ny), "real"
        )
        if not isinstance(values.get("checkpoint"), Mapping):
            raise TransportCodecError(
                "PR time-dependent Full result lacks a checkpoint"
            )
    else:
        for name in _FAST_OMITTED_FIELDS:
            if values.get(name) is not None:
                raise TransportCodecError(
                    f"PR time-dependent Fast result unexpectedly retained {name}"
                )
        if cuts_present:
            for name, shape in (
                ("longitudinal_intensity_xz", (nz, nx)),
                ("longitudinal_intensity_yz", (nz, ny)),
            ):
                _require_array(values, name, shape, "real")
    if policy == FULL_RESULT_POLICY:
        for name in ("longitudinal_intensity_xz", "longitudinal_intensity_yz"):
            if values.get(name) is not None:
                raise TransportCodecError(
                    f"PR time-dependent Full result unexpectedly retained {name}"
                )
    status = values.get("status")
    if status not in {"completed", "cancelled"}:
        raise TransportCodecError("PR time-dependent result status is invalid")
    completed = values.get("completed_steps")
    requested = values.get("requested_steps")
    if (
        type(completed) is not int
        or type(requested) is not int
        or completed < 0
        or requested < completed
    ):
        raise TransportCodecError(
            "PR time-dependent completed/requested steps are inconsistent"
        )
    if status == "completed" and completed != requested:
        raise TransportCodecError(
            "completed PR time-dependent result did not reach requested steps"
        )
    for name in ("time_normalized", "power_initial", "power_final"):
        value = values.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TransportCodecError(
                f"PR time-dependent result {name} must be finite"
            )
    if float(values["time_normalized"]) < 0.0:
        raise TransportCodecError(
            "PR time-dependent result time_normalized must be nonnegative"
        )
    if not isinstance(values.get("diagnostics"), Mapping):
        raise TransportCodecError(
            "PR time-dependent result diagnostics must be a mapping"
        )
    backend = values["diagnostics"].get("backend")
    if not isinstance(backend, Mapping) or backend.get("backend") not in {
        "numpy",
        "cupy",
    }:
        raise TransportCodecError(
            "PR time-dependent result backend provenance is invalid"
        )


def _validate_checkpoint_consistency(
    values: Mapping[str, Any], checkpoint: PRTimeDependentCheckpoint
) -> None:
    scalar_pairs = (
        ("completed_steps", checkpoint.completed_steps),
        ("requested_steps", checkpoint.requested_steps),
        ("time_normalized", checkpoint.time_normalized),
        ("status", checkpoint.status),
    )
    for name, checkpoint_value in scalar_pairs:
        if values[name] != checkpoint_value:
            raise TransportCodecError(
                f"PR time-dependent checkpoint {name} disagrees with result"
            )
    if dict(values["grid_summary"]) != checkpoint.grid_summary:
        raise TransportCodecError(
            "PR time-dependent checkpoint grid disagrees with result"
        )
    for result_name, checkpoint_value in (
        ("A_initial", checkpoint.A0),
        ("E_initial", checkpoint.E_initial),
        ("E_final", checkpoint.E_current),
    ):
        if not np.array_equal(values[result_name], checkpoint_value):
            raise TransportCodecError(
                f"PR time-dependent checkpoint {result_name} disagrees with result"
            )


def decode_pr_timedependent_transport_result(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRRunResult:
    """Decode and strictly validate one canonical reduced-TD result."""

    try:
        values = unpack_portable(dict(metadata), arrays)
        _validate_result(values)
        retention = dict(values.get("retention_summary", {
            "policy": FULL_RESULT_POLICY,
            "omitted_fields": [],
        }))
        checkpoint = None
        if retention["policy"] == FULL_RESULT_POLICY:
            checkpoint = _decode_checkpoint(values["checkpoint"], arrays)
            _validate_checkpoint_consistency(values, checkpoint)
        return PRRunResult(
            A_initial=values["A_initial"],
            A_final=values["A_final"],
            E_initial=values["E_initial"],
            E_final=values["E_final"],
            source_intensity_stack=values["source_intensity_stack"],
            power_initial=float(values["power_initial"]),
            power_final=float(values["power_final"]),
            completed_steps=values["completed_steps"],
            time_normalized=float(values["time_normalized"]),
            grid_summary=dict(values["grid_summary"]),
            launch_summary=dict(values["launch_summary"]),
            status=values["status"],
            requested_steps=values["requested_steps"],
            checkpoint=checkpoint,
            diagnostics=dict(values["diagnostics"]),
            retention_summary=retention,
            longitudinal_intensity_xz=values.get("longitudinal_intensity_xz"),
            longitudinal_intensity_yz=values.get("longitudinal_intensity_yz"),
            x_cut_um=values.get("x_cut_um"),
            y_cut_um=values.get("y_cut_um"),
        )
    except TransportCodecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            f"invalid PR time-dependent result payload: {exc}"
        ) from exc


PR_TIMEDEPENDENT_TRANSPORT_CODEC = TransportCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    request_codec_id=PR_TIMEDEPENDENT_REQUEST_CODEC_ID,
    request_codec_version=PR_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION,
    request_type=PRRunRequest,
    encode_request=encode_pr_timedependent_transport_request,
    decode_request=decode_pr_timedependent_transport_request,
    result_codec_id=PR_TIMEDEPENDENT_RESULT_CODEC_ID,
    result_codec_version=PR_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION,
    result_type=PRRunResult,
    encode_result=encode_pr_timedependent_transport_result,
    decode_result=decode_pr_timedependent_transport_result,
    encode_result_projection=encode_pr_timedependent_transport_result,
)


__all__ = [
    "PR_TIMEDEPENDENT_TRANSPORT_CODEC",
    "decode_pr_timedependent_transport_request",
    "decode_pr_timedependent_transport_result",
    "encode_pr_timedependent_transport_request",
    "encode_pr_timedependent_transport_result",
]
