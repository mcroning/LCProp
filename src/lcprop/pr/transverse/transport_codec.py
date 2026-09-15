"""PR-owned portable transport codec for the transverse static workflow."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
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
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.specs import PRMaterialSpec, PR_MATERIAL_ID
from lcprop.pr.transport_common import pack_portable, unpack_portable
from lcprop.pr.transverse.specs import (
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseDiscreteStaticNewtonRecord,
    PRTransverseStaticMaterialSolverOptions,
    PRTransverseStaticNewtonRecord,
)
from lcprop.pr.transverse.static_workflow import (
    PR_TRANSVERSE_STATIC_WORKFLOW,
    PRTransverseStaticCoupledRecord,
    PRTransverseStaticDiscreteIterationRecord,
    PRTransverseStaticMaterialIterationRecord,
    PRTransverseStaticRunRequest,
    PRTransverseStaticRunResult,
    PRTransverseStaticWorkflowOptions,
)
from lcprop.transport.codecs import EncodedRequest, EncodedResult, PortablePayload, TransportCodec
from lcprop.transport.envelopes import TransportCodecError
from lcprop.transport.result_policy import (
    FAST_RESULT_POLICY,
    FULL_RESULT_POLICY,
    normalize_result_policy,
)


PR_TRANSVERSE_STATIC_REQUEST_CODEC_ID = "pr.transverse_static.request"
PR_TRANSVERSE_STATIC_RESULT_CODEC_ID = "pr.transverse_static.result"
PR_TRANSVERSE_STATIC_TRANSPORT_CODEC_VERSION = 3
_PR_TRANSVERSE_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION = 2
_FAST_OMITTED_FIELDS = (
    "psi_initial",
    "psi_final",
    "source_intensity_stack",
    "equilibrium_residual_stack",
    "td_rhs_residual_stack",
)


def _validate_request(request: PRTransverseStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.material_response.validate_configuration(
        material_applied_field=request.material.applied_field,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
    )
    request.optical_boundary.validate()
    request.solver.validate()
    request.backend.validate()
    validate_channel_launch_elements(
        request.launch_elements, n_channels=len(request.beams.channels)
    )
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    expected_A = (len(request.beams.channels), request.grid.Nx, request.grid.Ny)
    expected_psi = (
        round_nz(request.grid.z_length_um, request.grid.dz_um),
        request.grid.Nx,
        request.grid.Ny,
    )
    if request.initial_A is not None and request.initial_A.shape != expected_A:
        raise ValueError(f"initial_A shape must be {expected_A}")
    if request.initial_psi is not None and request.initial_psi.shape != expected_psi:
        raise ValueError(f"initial_psi shape must be {expected_psi}")
    if request.scattering is not None:
        request.scattering.validate()


def encode_pr_transverse_static_transport_request(request: PRTransverseStaticRunRequest) -> EncodedRequest:
    if not isinstance(request, PRTransverseStaticRunRequest):
        raise TypeError("request must be a PRTransverseStaticRunRequest")
    _validate_request(request)
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "grid": asdict(request.grid),
        "beams": encode_beam_stack(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "material_response": asdict(request.material_response),
        "solver": pack_portable(request.solver, arrays, "solver"),
        "backend": asdict(request.backend),
        "launch_elements": encode_launch_elements(request.launch_elements),
        "initial_A": pack_portable(request.initial_A, arrays, "initial_A"),
        "initial_psi": pack_portable(request.initial_psi, arrays, "initial_psi"),
        "scattering": None if request.scattering is None else asdict(request.scattering),
        "optical_boundary": asdict(request.optical_boundary),
    }
    return EncodedRequest(PortablePayload(metadata, arrays), request.backend.backend)


def decode_pr_transverse_static_transport_request(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> PRTransverseStaticRunRequest:
    try:
        values = unpack_portable(dict(metadata), arrays)
        solver_values = dict(values["solver"])
        material_solver = PRTransverseStaticMaterialSolverOptions(**solver_values.pop("material_solver"))
        discrete_corrector = PRTransverseDiscreteStaticCorrectorOptions(**solver_values.pop("discrete_corrector"))
        scattering = values["scattering"]
        beams = decode_beam_stack(values["beams"])
        request = PRTransverseStaticRunRequest(
            grid=GridSpec(**values["grid"]),
            beams=beams,
            material=PRMaterialSpec(**values["material"]),
            transport=PRTransverseTransportProfile(**values["transport"]),
            dielectric=PRTransverseDielectricProfile(**values["dielectric"]),
            boundary=PRTransverseBoundaryProfile(**values["boundary"]),
            projection=PRTransverseProjectionProfile(**values["projection"]),
            material_response=PRTransverseMaterialResponseSpec(
                **values.get("material_response", {})
            ),
            solver=PRTransverseStaticWorkflowOptions(
                material_solver=material_solver,
                discrete_corrector=discrete_corrector,
                **solver_values,
            ),
            backend=BackendSpec(**values["backend"]),
            launch_elements=decode_launch_elements(
                values.get("launch_elements", []),
                n_channels=len(beams.channels),
            ),
            initial_A=values["initial_A"],
            initial_psi=values["initial_psi"],
            scattering=None if scattering is None else PRCanonicalScatteringSpec(**scattering),
            optical_boundary=TransverseBoundarySpec(
                **values.get("optical_boundary", {})
            ),
        )
        _validate_request(request)
        return request
    except TransportCodecError:
        raise
    except PortableLaunchPayloadError as exc:
        raise TransportCodecError(
            f"invalid PR transverse-static request payload: {exc}"
        ) from exc
    except Exception as exc:
        raise TransportCodecError(f"invalid PR transverse-static request payload: {exc}") from exc


def encode_pr_transverse_static_transport_result(
    result: PRTransverseStaticRunResult,
    result_policy: str = FULL_RESULT_POLICY,
) -> EncodedResult:
    if not isinstance(result, PRTransverseStaticRunResult):
        raise TypeError("result must be a PRTransverseStaticRunResult")
    policy = normalize_result_policy(result_policy)
    arrays: dict[str, np.ndarray] = {}
    omitted = _FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ()
    if omitted:
        if result.source_intensity_stack is None:
            try:
                cuts = retained_longitudinal_intensity_cuts(result)
            except ValueError:
                cuts = None
        else:
            material = result.resolved_profile["material"]
            cuts = extract_longitudinal_optical_intensity_cuts(
                result.source_intensity_stack,
                grid_summary=result.grid_summary,
                peak_intensity_reference=channel_peak_intensity_reference(
                    np.asarray(result.A_initial), xp=np
                ),
                background_intensity=(
                    float(material["dark_intensity"])
                    + float(material["uniform_background_intensity"])
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
    device_summary = pack_portable(
        result.backend_summary, arrays, "device_summary"
    )
    termination = result.diagnostics.get("termination_reason", result.status)
    return EncodedResult(
        PortablePayload(metadata, arrays),
        scientific_status=result.status,
        converged=bool(result.converged),
        cancelled=result.status == "cancelled",
        termination_reason=None if termination is None else str(termination),
        scientific_backend_resolved=backend,
        device_summary=device_summary,
        result_policy=policy,
    )


def _summary_dimension(summary: Mapping[str, Any], name: str) -> int:
    value = summary.get(name)
    if type(value) is not int or value < 1:
        raise TransportCodecError(f"PR result {name} must be a positive integer")
    return value


def _validate_result_shapes(values: Mapping[str, Any]) -> None:
    grid = values.get("grid_summary")
    launch = values.get("launch_summary")
    if not isinstance(grid, Mapping) or not isinstance(launch, Mapping):
        raise TransportCodecError("PR result lacks grid or launch summary")
    nx = _summary_dimension(grid, "Nx")
    ny = _summary_dimension(grid, "Ny")
    nz = _summary_dimension(grid, "Nz")
    nch = _summary_dimension(launch, "Nch")
    retention = values.get("retention_summary", {
        "policy": FULL_RESULT_POLICY,
        "omitted_fields": [],
    })
    if not isinstance(retention, Mapping):
        raise TransportCodecError("PR result retention_summary must be a mapping")
    policy = normalize_result_policy(retention.get("policy", FULL_RESULT_POLICY))
    expected_omitted = set(
        _FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ()
    )
    omitted = retention.get("omitted_fields", [])
    if not isinstance(omitted, (tuple, list)) or set(omitted) != expected_omitted:
        raise TransportCodecError("PR result omitted_fields disagree with result policy")
    cuts_present = validate_longitudinal_cut_coordinates(values, grid)
    expected = {
        "A_initial": (nch, nx, ny),
        "A_final": (nch, nx, ny),
        "psi_initial": (nz, nx, ny),
        "psi_final": (nz, nx, ny),
        "source_intensity_stack": (nz, nx, ny),
        "equilibrium_residual_stack": (nz, nx, ny),
        "td_rhs_residual_stack": (nz, nx, ny),
    }
    material_response = (
        values.get("resolved_profile", {})
        .get("material_response", {})
        .get("model", "nonlinear")
    )
    for name, shape in expected.items():
        value = values.get(name)
        if name in expected_omitted:
            if value is not None:
                raise TransportCodecError(
                    f"Fast PR result must omit {name}"
                )
            continue
        if (
            name == "td_rhs_residual_stack"
            and material_response == "linearized"
            and value is None
        ):
            continue
        if not isinstance(value, np.ndarray):
            raise TransportCodecError(f"PR result {name} must be a numeric array")
        if value.dtype.hasobject:
            raise TransportCodecError(f"PR result {name} has forbidden object dtype")
        if value.shape != shape:
            raise TransportCodecError(
                f"PR result {name} shape {value.shape} does not match {shape}"
            )
    for name, shape in (
        ("longitudinal_intensity_xz", (nz, nx)),
        ("longitudinal_intensity_yz", (nz, ny)),
    ):
        value = values.get(name)
        if policy == FAST_RESULT_POLICY:
            if not cuts_present:
                continue
            if (
                not isinstance(value, np.ndarray)
                or value.dtype.kind != "f"
                or value.shape != shape
            ):
                raise TransportCodecError(
                    f"Fast PR result {name} must have shape {shape}"
                )
        elif value is not None:
            raise TransportCodecError(f"Full PR result unexpectedly retained {name}")


def _same_visibility(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _validate_continuation(values: Mapping[str, Any]) -> None:
    diagnostics = values.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TransportCodecError("PR result diagnostics must be a mapping")
    continuation_keys = {
        "visibility_schedule", "continuation_stages", "direct_attempt_status",
        "direct_attempt_termination_reason", "requested_final_visibility",
        "last_attempted_visibility", "final_visibility",
        "continuation_succeeded", "continuation_cancelled",
        "continuation_cancellation_visibility", "continuation_failure_visibility",
        "returned_state_source",
    }
    used = diagnostics.get("continuation_used", False)
    if type(used) is not bool:
        raise TransportCodecError("continuation_used must be boolean")
    if not used:
        unexpected = continuation_keys.intersection(diagnostics)
        if unexpected:
            raise TransportCodecError(
                "continuation provenance is present while continuation_used is false"
            )
        return
    missing = continuation_keys - set(diagnostics)
    if missing:
        raise TransportCodecError(
            "PR continuation provenance is missing: " + ", ".join(sorted(missing))
        )
    succeeded = diagnostics["continuation_succeeded"]
    cancelled = diagnostics["continuation_cancelled"]
    if type(succeeded) is not bool or type(cancelled) is not bool:
        raise TransportCodecError("continuation success/cancellation flags must be boolean")
    if succeeded and cancelled:
        raise TransportCodecError("continuation cannot be successful and cancelled")
    schedule = diagnostics["visibility_schedule"]
    stages = diagnostics["continuation_stages"]
    if not isinstance(schedule, (tuple, list)) or not schedule:
        raise TransportCodecError("continuation visibility_schedule must be non-empty")
    if not isinstance(stages, (tuple, list)) or not stages:
        raise TransportCodecError("continuation_stages must be non-empty")
    if len(stages) > len(schedule):
        raise TransportCodecError("continuation stages exceed the visibility schedule")
    stage_visibilities = []
    stage_converged = []
    stage_statuses = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or not {
            "visibility", "status", "converged"
        }.issubset(stage):
            raise TransportCodecError(
                "each continuation stage requires visibility, status, and converged"
            )
        visibility = stage["visibility"]
        if not _same_visibility(visibility, schedule[index]):
            raise TransportCodecError("continuation stages are not a schedule prefix")
        if type(stage["converged"]) is not bool:
            raise TransportCodecError("continuation stage converged must be boolean")
        stage_visibilities.append(visibility)
        stage_converged.append(stage["converged"])
        stage_statuses.append(stage["status"])
    requested = diagnostics["requested_final_visibility"]
    last = diagnostics["last_attempted_visibility"]
    if not _same_visibility(requested, 1.0):
        raise TransportCodecError("requested_final_visibility must be full visibility")
    if not _same_visibility(schedule[-1], requested):
        raise TransportCodecError("visibility schedule must end at requested visibility")
    if not _same_visibility(last, stage_visibilities[-1]):
        raise TransportCodecError("last_attempted_visibility disagrees with stages")
    if not all(stage_converged[:-1]):
        raise TransportCodecError("continuation advanced after an unconverged stage")
    final = diagnostics["final_visibility"]
    cancellation_visibility = diagnostics["continuation_cancellation_visibility"]
    failure_visibility = diagnostics["continuation_failure_visibility"]
    source = diagnostics["returned_state_source"]
    if succeeded:
        if (
            cancelled
            or not all(stage_converged)
            or not _same_visibility(final, requested)
            or not _same_visibility(last, requested)
        ):
            raise TransportCodecError("successful continuation must reach requested visibility")
        if cancellation_visibility is not None or failure_visibility is not None:
            raise TransportCodecError("successful continuation cannot record failure visibility")
        if source != "final_full_visibility_stage":
            raise TransportCodecError("successful continuation has invalid returned_state_source")
    elif cancelled:
        if (
            values.get("status") != "cancelled"
            or final is not None
            or stage_converged[-1]
            or stage_statuses[-1] != "cancelled"
        ):
            raise TransportCodecError("cancelled continuation must return cancelled state")
        if not _same_visibility(cancellation_visibility, last) or failure_visibility is not None:
            raise TransportCodecError("cancelled continuation visibility is inconsistent")
        if source != "cancelled_continuation_stage":
            raise TransportCodecError("cancelled continuation has invalid returned_state_source")
    else:
        if final is not None or cancellation_visibility is not None or stage_converged[-1]:
            raise TransportCodecError("failed continuation cannot record final/cancel visibility")
        if not _same_visibility(failure_visibility, last):
            raise TransportCodecError("failed continuation visibility is inconsistent")
        if source != "direct_full_visibility_failure":
            raise TransportCodecError("failed continuation has invalid returned_state_source")


def decode_pr_transverse_static_transport_result(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> PRTransverseStaticRunResult:
    try:
        values = unpack_portable(dict(metadata), arrays)
        values.setdefault("retention_summary", {
            "policy": FULL_RESULT_POLICY,
            "omitted_fields": [],
        })
        _validate_result_shapes(values)
        _validate_continuation(values)
        material_records = []
        for item in values["material_iteration_records"]:
            item = dict(item)
            material_records.append(PRTransverseStaticMaterialIterationRecord(
                coupled_iteration=int(item["coupled_iteration"]),
                newton_record=PRTransverseStaticNewtonRecord(**item["newton_record"]),
            ))
        discrete_records = []
        for item in values["discrete_iteration_records"]:
            item = dict(item)
            discrete_records.append(PRTransverseStaticDiscreteIterationRecord(
                coupled_iteration=int(item["coupled_iteration"]),
                newton_record=PRTransverseDiscreteStaticNewtonRecord(**item["newton_record"]),
            ))
        diagnostics = dict(values["diagnostics"])
        for key in ("visibility_schedule", "continuation_stages"):
            if isinstance(diagnostics.get(key), list):
                diagnostics[key] = tuple(diagnostics[key])
        result = PRTransverseStaticRunResult(
            A_initial=values["A_initial"], A_final=values["A_final"],
            psi_initial=values["psi_initial"], psi_final=values["psi_final"],
            source_intensity_stack=values["source_intensity_stack"],
            equilibrium_residual_stack=values["equilibrium_residual_stack"],
            td_rhs_residual_stack=values["td_rhs_residual_stack"],
            power_initial=float(values["power_initial"]), power_final=float(values["power_final"]),
            converged=bool(values["converged"]),
            completed_coupled_iterations=int(values["completed_coupled_iterations"]),
            iteration_records=tuple(PRTransverseStaticCoupledRecord(**v) for v in values["iteration_records"]),
            material_iteration_records=tuple(material_records),
            discrete_iteration_records=tuple(discrete_records),
            grid_summary=dict(values["grid_summary"]), launch_summary=dict(values["launch_summary"]),
            backend_summary=dict(values["backend_summary"]), resolved_profile=dict(values["resolved_profile"]),
            replay_diagnostics=dict(values["replay_diagnostics"]), diagnostics=diagnostics,
            timing=dict(values["timing"]), status=str(values["status"]),
            retention_summary=dict(values["retention_summary"]),
            longitudinal_intensity_xz=values.get("longitudinal_intensity_xz"),
            longitudinal_intensity_yz=values.get("longitudinal_intensity_yz"),
            x_cut_um=values.get("x_cut_um"), y_cut_um=values.get("y_cut_um"),
        )
    except Exception as exc:
        raise TransportCodecError(f"invalid PR transverse-static result payload: {exc}") from exc
    if result.equilibrium_residual_stack is None:
        return result
    residual = np.asarray(result.equilibrium_residual_stack)
    rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    maximum = float(np.max(np.abs(residual)))
    expected_rms = result.diagnostics.get(
        "authoritative_equilibrium_rms",
        result.diagnostics.get("equilibrium_residual_rms"),
    )
    expected_max = result.diagnostics.get(
        "authoritative_equilibrium_max",
        result.diagnostics.get("equilibrium_residual_max"),
    )
    tolerance = 5e-6 if residual.dtype == np.float32 else 1e-11
    if expected_rms is not None and not np.isclose(rms, expected_rms, rtol=tolerance, atol=tolerance):
        raise TransportCodecError("exported PR equilibrium residual RMS disagrees with diagnostics")
    if expected_max is not None and not np.isclose(maximum, expected_max, rtol=tolerance, atol=tolerance):
        raise TransportCodecError("exported PR equilibrium residual maximum disagrees with diagnostics")
    return result


PR_TRANSVERSE_STATIC_TRANSPORT_CODEC = TransportCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
    request_codec_id=PR_TRANSVERSE_STATIC_REQUEST_CODEC_ID,
    request_codec_version=PR_TRANSVERSE_STATIC_TRANSPORT_CODEC_VERSION,
    request_type=PRTransverseStaticRunRequest,
    encode_request=encode_pr_transverse_static_transport_request,
    decode_request=decode_pr_transverse_static_transport_request,
    result_codec_id=PR_TRANSVERSE_STATIC_RESULT_CODEC_ID,
    result_codec_version=PR_TRANSVERSE_STATIC_TRANSPORT_CODEC_VERSION,
    result_type=PRTransverseStaticRunResult,
    encode_result=encode_pr_transverse_static_transport_result,
    encode_result_projection=encode_pr_transverse_static_transport_result,
    decode_result=decode_pr_transverse_static_transport_result,
    compatible_request_codec_versions=(
        _PR_TRANSVERSE_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION,
    ),
    compatible_result_codec_versions=(
        _PR_TRANSVERSE_STATIC_PREVIOUS_TRANSPORT_CODEC_VERSION,
    ),
)


__all__ = [
    "PR_TRANSVERSE_STATIC_TRANSPORT_CODEC",
    "decode_pr_transverse_static_transport_request",
    "decode_pr_transverse_static_transport_result",
    "encode_pr_transverse_static_transport_request",
    "encode_pr_transverse_static_transport_result",
]
