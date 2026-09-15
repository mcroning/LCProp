"""PR-owned portable transport codec for the reduced static workflow."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.core.grid import round_nz
from lcprop.optics.launch_configuration import reject_prepared_launch_conflict
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.persistence.experiments import decode_beam_stack, encode_beam_stack
from lcprop.optics.screens import validate_channel_launch_elements
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
from lcprop.pr.specs import PRMaterialSpec, PR_MATERIAL_ID
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import (
    PRCoupledStaticIterationRecord,
    PRCoupledStaticSliceSummary,
    PRStaticRunRequest,
    PRStaticRunResult,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
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


PR_STATIC_REQUEST_CODEC_ID = "pr.static.request"
PR_STATIC_RESULT_CODEC_ID = "pr.static.result"
PR_STATIC_TRANSPORT_CODEC_VERSION = 2
_PR_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION = 1


def _validate_request(request: PRStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    request.material_response.validate()
    request.optical_boundary.validate()
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


def encode_pr_static_transport_request(
    request: PRStaticRunRequest,
) -> EncodedRequest:
    """Encode one canonical reduced-static request."""

    if not isinstance(request, PRStaticRunRequest):
        raise TypeError("request must be a PRStaticRunRequest")
    _validate_request(request)
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "grid": asdict(request.grid),
        "beams": encode_beam_stack(request.beams),
        "material": asdict(request.material),
        "solver": pack_portable(request.solver, arrays, "solver"),
        "backend": asdict(request.backend),
        "material_response": asdict(request.material_response),
        "launch_elements": encode_launch_elements(request.launch_elements),
        "initial_A": pack_portable(request.initial_A, arrays, "initial_A"),
        "initial_E": pack_portable(request.initial_E, arrays, "initial_E"),
        "optical_boundary": asdict(request.optical_boundary),
    }
    return EncodedRequest(
        PortablePayload(metadata, arrays), request.backend.backend
    )


def decode_pr_static_transport_request(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRStaticRunRequest:
    """Decode and validate one canonical reduced-static request."""

    try:
        values = unpack_portable(dict(metadata), arrays)
        solver_values = dict(values["solver"])
        material_values = solver_values.pop("material_solver")
        material_solver = (
            None
            if material_values is None
            else PRStaticSolverOptions(**material_values)
        )
        beams = decode_beam_stack(values["beams"])
        request = PRStaticRunRequest(
            grid=GridSpec(**values["grid"]),
            beams=beams,
            material=PRMaterialSpec(**values["material"]),
            solver=PRStaticWorkflowOptions(
                material_solver=material_solver,
                **solver_values,
            ),
            backend=BackendSpec(**values["backend"]),
            launch_elements=decode_launch_elements(
                values.get("launch_elements", []),
                n_channels=len(beams.channels),
            ),
            initial_A=values["initial_A"],
            initial_E=values["initial_E"],
            material_response=PRTransverseMaterialResponseSpec(
                **values.get("material_response", {})
            ),
            optical_boundary=TransverseBoundarySpec(
                **values.get("optical_boundary", {})
            ),
        )
        _validate_request(request)
        return request
    except TransportCodecError:
        raise
    except (PortableLaunchPayloadError, KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            f"invalid PR static request payload: {exc}"
        ) from exc


_FAST_OMITTED_FIELDS = (
    "E_initial",
    "E_final",
    "source_intensity_stack",
    "residual_stack",
)


def encode_pr_static_transport_result(
    result: PRStaticRunResult,
    result_policy: str = FULL_RESULT_POLICY,
) -> EncodedResult:
    """Encode the complete canonical reduced-static result."""

    if not isinstance(result, PRStaticRunResult):
        raise TypeError("result must be a PRStaticRunResult")
    policy = normalize_result_policy(result_policy)
    arrays: dict[str, np.ndarray] = {}
    omitted = _FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ()
    if omitted:
        if result.source_intensity_stack is None:
            try:
                cuts = retained_longitudinal_intensity_cuts(result)
            except ValueError:
                cuts = None
        elif np.asarray(result.source_intensity_stack).shape[0] == 0:
            cuts = None
        else:
            cuts = extract_longitudinal_optical_intensity_cuts(
                result.source_intensity_stack,
                grid_summary=result.grid_summary,
                peak_intensity_reference=channel_peak_intensity_reference(
                    np.asarray(result.A_initial), xp=np
                ),
                background_intensity=float(
                    result.material_response_summary.get(
                        "background_intensity", 0.0
                    )
                ),
            )
        projected = replace(
            result,
            **{name: None for name in omitted},
            longitudinal_intensity_xz=None if cuts is None else cuts.xz,
            longitudinal_intensity_yz=None if cuts is None else cuts.yz,
            x_cut_um=None if cuts is None else cuts.x_cut_um,
            y_cut_um=None if cuts is None else cuts.y_cut_um,
        )
    else:
        cuts = None
        projected = result
    values = asdict(projected)
    values["retention_summary"] = (
        fast_retention_summary(omitted, cuts)
        if cuts is not None
        else {"policy": policy, "omitted_fields": list(omitted)}
    )
    metadata = pack_portable(values, arrays, "result")
    backend = str(result.backend_summary.get("backend", "unknown"))
    termination = (
        "cancelled_at_accepted_boundary"
        if result.status == "cancelled"
        else (
            result.slice_summaries[-1].termination_reason
            if result.slice_summaries
            else result.status
        )
    )
    return EncodedResult(
        PortablePayload(metadata, arrays),
        scientific_status=result.status,
        converged=bool(result.converged),
        cancelled=result.status == "cancelled",
        termination_reason=str(termination),
        scientific_backend_resolved=backend,
        device_summary=pack_portable(
            result.backend_summary, arrays, "device_summary"
        ),
        result_policy=policy,
    )


def _positive_dimension(summary: Mapping[str, Any], name: str) -> int:
    value = summary.get(name)
    if type(value) is not int or value < 1:
        raise TransportCodecError(
            f"PR static result {name} must be a positive integer"
        )
    return value


def _validate_result(values: Mapping[str, Any]) -> None:
    grid = values.get("grid_summary")
    launch = values.get("launch_summary")
    if not isinstance(grid, Mapping) or not isinstance(launch, Mapping):
        raise TransportCodecError(
            "PR static result lacks grid or launch summary"
        )
    nx = _positive_dimension(grid, "Nx")
    ny = _positive_dimension(grid, "Ny")
    nz = _positive_dimension(grid, "Nz")
    nch = _positive_dimension(launch, "Nch")
    completed = values.get("completed_slices")
    if type(completed) is not int or not 0 <= completed <= nz:
        raise TransportCodecError(
            "PR static completed_slices is outside the declared grid"
        )
    retention = values.get("retention_summary", {
        "policy": FULL_RESULT_POLICY,
        "omitted_fields": [],
    })
    if not isinstance(retention, Mapping):
        raise TransportCodecError("PR static retention summary must be an object")
    try:
        policy = normalize_result_policy(retention.get("policy", FULL_RESULT_POLICY))
    except ValueError as exc:
        raise TransportCodecError(str(exc)) from exc
    expected_omitted = set(_FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ())
    omitted = retention.get("omitted_fields", [])
    if not isinstance(omitted, (tuple, list)) or set(omitted) != expected_omitted:
        raise TransportCodecError("PR static omitted fields disagree with result policy")
    cuts_present = validate_longitudinal_cut_coordinates(values, grid)
    expected = {
        "A_initial": (nch, nx, ny),
        "A_final": (nch, nx, ny),
        "E_initial": (completed, nx, ny),
        "E_final": (completed, nx, ny),
        "source_intensity_stack": (completed, nx, ny),
        "residual_stack": (completed, nx, ny),
    }
    for name, shape in expected.items():
        array = values.get(name)
        if name in expected_omitted:
            if array is not None:
                raise TransportCodecError(
                    f"PR static Fast result unexpectedly retained {name}"
                )
            continue
        if not isinstance(array, np.ndarray):
            raise TransportCodecError(
                f"PR static result {name} must be a numeric array"
            )
        if array.dtype.hasobject:
            raise TransportCodecError(
                f"PR static result {name} has forbidden object dtype"
            )
        if array.shape != shape:
            raise TransportCodecError(
                f"PR static result {name} shape {array.shape} does not match {shape}"
            )
    for name, shape in (
        ("longitudinal_intensity_xz", (completed, nx)),
        ("longitudinal_intensity_yz", (completed, ny)),
    ):
        array = values.get(name)
        if policy == FAST_RESULT_POLICY:
            if not cuts_present:
                continue
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.kind != "f"
                or array.shape != shape
            ):
                raise TransportCodecError(
                    f"PR static Fast result {name} must have shape {shape}"
                )
        elif array is not None:
            raise TransportCodecError(
                f"PR static Full result unexpectedly retained {name}"
            )
    summaries = values.get("slice_summaries")
    if not isinstance(summaries, list) or len(summaries) != completed:
        raise TransportCodecError(
            "PR static slice summaries must match completed_slices"
        )
    status = values.get("status")
    converged = values.get("converged")
    if type(converged) is not bool:
        raise TransportCodecError("PR static converged must be boolean")
    if status not in {"converged", "not_converged", "cancelled"}:
        raise TransportCodecError("PR static result status is invalid")
    if converged != (status == "converged"):
        raise TransportCodecError(
            "PR static convergence flag disagrees with result status"
        )
    material_response = values.get(
        "material_response_summary",
        {"model": "nonlinear", "validation_status": "validated"},
    )
    if not isinstance(material_response, Mapping):
        raise TransportCodecError(
            "PR static material_response_summary must be a mapping"
        )


def decode_pr_static_transport_result(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRStaticRunResult:
    """Decode and validate the complete canonical reduced-static result."""

    try:
        values = unpack_portable(dict(metadata), arrays)
        _validate_result(values)
        result = PRStaticRunResult(
            A_initial=values["A_initial"],
            A_final=values["A_final"],
            E_initial=values["E_initial"],
            E_final=values["E_final"],
            source_intensity_stack=values["source_intensity_stack"],
            residual_stack=values["residual_stack"],
            power_initial=float(values["power_initial"]),
            power_final=float(values["power_final"]),
            converged=values["converged"],
            completed_slices=values["completed_slices"],
            iteration_records=tuple(
                PRCoupledStaticIterationRecord(**item)
                for item in values["iteration_records"]
            ),
            slice_summaries=tuple(
                PRCoupledStaticSliceSummary(**item)
                for item in values["slice_summaries"]
            ),
            grid_summary=dict(values["grid_summary"]),
            launch_summary=dict(values["launch_summary"]),
            backend_summary=dict(values["backend_summary"]),
            tolerance_provenance=dict(values["tolerance_provenance"]),
            replay_diagnostics=dict(values["replay_diagnostics"]),
            status=values["status"],
            retention_summary=dict(values.get("retention_summary", {
                "policy": FULL_RESULT_POLICY,
                "omitted_fields": [],
            })),
            material_response_summary=dict(
                values.get(
                    "material_response_summary",
                    {"model": "nonlinear", "validation_status": "validated"},
                )
            ),
            longitudinal_intensity_xz=values.get("longitudinal_intensity_xz"),
            longitudinal_intensity_yz=values.get("longitudinal_intensity_yz"),
            x_cut_um=values.get("x_cut_um"),
            y_cut_um=values.get("y_cut_um"),
        )
        return result
    except TransportCodecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            f"invalid PR static result payload: {exc}"
        ) from exc


PR_STATIC_TRANSPORT_CODEC = TransportCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_STATIC_WORKFLOW,
    request_codec_id=PR_STATIC_REQUEST_CODEC_ID,
    request_codec_version=PR_STATIC_TRANSPORT_CODEC_VERSION,
    request_type=PRStaticRunRequest,
    encode_request=encode_pr_static_transport_request,
    decode_request=decode_pr_static_transport_request,
    result_codec_id=PR_STATIC_RESULT_CODEC_ID,
    result_codec_version=PR_STATIC_TRANSPORT_CODEC_VERSION,
    result_type=PRStaticRunResult,
    encode_result=encode_pr_static_transport_result,
    decode_result=decode_pr_static_transport_result,
    encode_result_projection=encode_pr_static_transport_result,
    compatible_request_codec_versions=(
        _PR_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION,
    ),
    compatible_result_codec_versions=(
        _PR_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION,
    ),
)


__all__ = [
    "PR_STATIC_TRANSPORT_CODEC",
    "decode_pr_static_transport_request",
    "decode_pr_static_transport_result",
    "encode_pr_static_transport_request",
    "encode_pr_static_transport_result",
]
