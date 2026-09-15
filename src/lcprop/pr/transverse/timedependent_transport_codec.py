"""PR-owned portable transport codec for the transverse TD workflow."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.core.grid import round_nz
from lcprop.optics.launch_configuration import reject_prepared_launch_conflict
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.screens import validate_channel_launch_elements
from lcprop.persistence.experiments import decode_beam_stack, encode_beam_stack
from lcprop.pr.portable_launch import (
    PortableLaunchPayloadError,
    decode_launch_elements,
    encode_launch_elements,
)
from lcprop.pr.longitudinal_cuts import (
    PRLongitudinalIntensityCuts,
    fast_retention_summary,
    validate_longitudinal_cut_coordinates,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.visualization import (
    make_fast_intensity_preview,
    validate_fast_intensity_preview,
)
from lcprop.pr.specs import PRMaterialSpec, PR_MATERIAL_ID
from lcprop.pr.transport_common import pack_portable, unpack_portable
from lcprop.pr.transverse.specs import (
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseRunResult,
    PRTransverseSolverOptions,
    PRTransverseTransportProfile,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)
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


PR_TRANSVERSE_TIMEDEPENDENT_REQUEST_CODEC_ID = (
    "pr.transverse_timedependent.request"
)
PR_TRANSVERSE_TIMEDEPENDENT_RESULT_CODEC_ID = (
    "pr.transverse_timedependent.result"
)
PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION = 2
_PR_TRANSVERSE_TIMEDEPENDENT_PREVIOUS_TRANSPORT_CODEC_VERSION = 1
PR_TRANSVERSE_TIMEDEPENDENT_RESULT_CODEC_VERSION = 3
_CANCELLATION_OBSERVED_STAGES = {
    "material_step_boundary",
    "material_source_optical_z_march",
    "linearized_material_plane",
    "after_material_candidate",
}


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], *, label: str
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise TransportCodecError(f"{label} has " + "; ".join(details))


def _validate_request(request: PRTransverseRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.solver.validate()
    request.backend.validate()
    request.material_response.validate_configuration(
        material_applied_field=request.material.applied_field,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
    )
    request.optical_boundary.validate()
    validate_channel_launch_elements(
        request.launch_elements,
        n_channels=len(request.beams.channels),
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
    if (
        request.initial_psi is not None
        and request.initial_psi.shape != expected_psi
    ):
        raise ValueError(f"initial_psi shape must be {expected_psi}")
    if request.scattering is not None:
        request.scattering.validate()


def encode_pr_transverse_timedependent_transport_request(
    request: PRTransverseRunRequest,
) -> EncodedRequest:
    """Encode one complete canonical transverse-TD request."""

    if not isinstance(request, PRTransverseRunRequest):
        raise TypeError("request must be a PRTransverseRunRequest")
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
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "initial_A": pack_portable(request.initial_A, arrays, "request.initial_A"),
        "initial_psi": pack_portable(
            request.initial_psi, arrays, "request.initial_psi"
        ),
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
        "launch_elements": encode_launch_elements(request.launch_elements),
        "optical_boundary": asdict(request.optical_boundary),
    }
    return EncodedRequest(
        PortablePayload(metadata, arrays), request.backend.backend
    )


def decode_pr_transverse_timedependent_transport_request(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRTransverseRunRequest:
    """Decode and validate one canonical transverse-TD request."""

    required = {
        "grid",
        "beams",
        "material",
        "transport",
        "dielectric",
        "boundary",
        "projection",
        "solver",
        "backend",
        "initial_A",
        "initial_psi",
        "scattering",
        "launch_elements",
    }
    try:
        extra = set(metadata) - required - {"material_response", "optical_boundary"}
        missing = required - set(metadata)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if extra:
                details.append("unexpected " + ", ".join(sorted(extra)))
            raise TransportCodecError(
                "PR transverse-TD request has " + "; ".join(details)
            )
        values = unpack_portable(dict(metadata), arrays)
        beams = decode_beam_stack(values["beams"])
        scattering = values["scattering"]
        request = PRTransverseRunRequest(
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
            solver=PRTransverseSolverOptions(**values["solver"]),
            backend=BackendSpec(**values["backend"]),
            initial_A=values["initial_A"],
            initial_psi=values["initial_psi"],
            scattering=(
                None
                if scattering is None
                else PRCanonicalScatteringSpec(**scattering)
            ),
            launch_elements=decode_launch_elements(
                values["launch_elements"],
                n_channels=len(beams.channels),
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
            f"invalid PR transverse-TD request payload: {exc}"
        ) from exc


_FAST_OMITTED_FIELDS = ("psi_initial", "psi_final", "source_intensity_stack")


def encode_pr_transverse_timedependent_transport_result(
    result: PRTransverseRunResult,
    result_policy: str = FULL_RESULT_POLICY,
) -> EncodedResult:
    """Encode the complete canonical transverse-TD result."""

    if not isinstance(result, PRTransverseRunResult):
        raise TypeError("result must be a PRTransverseRunResult")
    policy = normalize_result_policy(result_policy)
    arrays: dict[str, np.ndarray] = {}
    cuts = None
    preview_data = result.intensity_preview
    preview_metadata = result.intensity_preview_metadata
    if policy == FAST_RESULT_POLICY:
        try:
            cuts = PRLongitudinalIntensityCuts(
                xz=np.asarray(result.longitudinal_intensity_xz),
                yz=np.asarray(result.longitudinal_intensity_yz),
                x_cut_um=float(result.x_cut_um),
                y_cut_um=float(result.y_cut_um),
            )
        except (TypeError, ValueError):
            cuts = None
        if result.source_intensity_stack is not None:
            material = result.resolved_profile["material"]
            preview = make_fast_intensity_preview(
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
            preview_data = preview.intensity
            preview_metadata = preview.metadata
    metadata = {
        "A_initial": pack_portable(result.A_initial, arrays, "result.A_initial"),
        "A_final": pack_portable(result.A_final, arrays, "result.A_final"),
        "psi_initial": None,
        "psi_final": None,
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
        "intensity_preview": (
            None
            if policy != FAST_RESULT_POLICY or preview_data is None
            else pack_portable(
                preview_data, arrays, "result.intensity_preview"
            )
        ),
        "intensity_preview_metadata": (
            preview_metadata
            if policy == FAST_RESULT_POLICY else None
        ),
        "td_scalar_history": pack_portable(
            result.td_scalar_history, arrays, "result.td_scalar_history"
        ),
        "td_preview_movie": (
            None if result.td_preview_movie is None else pack_portable(
                result.td_preview_movie, arrays, "result.td_preview_movie"
            )
        ),
        "td_preview_movie_metadata": result.td_preview_movie_metadata,
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
        "backend_summary": pack_portable(
            result.backend_summary, arrays, "result.backend_summary"
        ),
        "resolved_profile": pack_portable(
            result.resolved_profile, arrays, "result.resolved_profile"
        ),
        "status": result.status,
        "requested_steps": int(result.requested_steps),
        "diagnostics": pack_portable(
            result.diagnostics, arrays, "result.diagnostics"
        ),
        "retention_summary": (
            fast_retention_summary(
                _FAST_OMITTED_FIELDS,
                cuts,
                intensity_preview_metadata=preview_metadata,
                additional_retained_fields=("td_scalar_history",) + (
                    ("td_preview_movie", "td_preview_movie_metadata")
                    if result.td_preview_movie is not None
                    else (("td_preview_movie_metadata",)
                          if result.td_preview_movie_metadata is not None else ())
                ),
            )
            if policy == FAST_RESULT_POLICY
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
            "psi_initial": pack_portable(
                result.psi_initial, arrays, "result.psi_initial"
            ),
            "psi_final": pack_portable(
                result.psi_final, arrays, "result.psi_final"
            ),
            "source_intensity_stack": pack_portable(
                result.source_intensity_stack,
                arrays,
                "result.source_intensity_stack",
            ),
        })
    _validate_result(unpack_portable(metadata, arrays))
    backend_summary = dict(result.backend_summary)
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
        device_summary=backend_summary,
        result_policy=policy,
    )


def _positive_dimension(summary: Mapping[str, Any], name: str) -> int:
    value = summary.get(name)
    if type(value) is not int or value < 1:
        raise TransportCodecError(
            f"PR transverse-TD result {name} must be a positive integer"
        )
    return value


def _require_array(
    values: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    kind: str,
) -> np.ndarray:
    value = values.get(name)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise TransportCodecError(
            f"PR transverse-TD result {name} must be a numeric array"
        )
    if value.shape != shape:
        raise TransportCodecError(
            f"PR transverse-TD result {name} shape {value.shape} "
            f"does not match {shape}"
        )
    if kind == "complex" and value.dtype.kind != "c":
        raise TransportCodecError(
            f"PR transverse-TD result {name} must be complex"
        )
    if kind == "real" and value.dtype.kind != "f":
        raise TransportCodecError(
            f"PR transverse-TD result {name} must be real floating point"
        )
    return value


def _validate_result(values: Mapping[str, Any]) -> None:
    required = {
        "A_initial",
        "A_final",
        "psi_initial",
        "psi_final",
        "source_intensity_stack",
        "power_initial",
        "power_final",
        "completed_steps",
        "time_normalized",
        "grid_summary",
        "launch_summary",
        "backend_summary",
        "resolved_profile",
        "status",
        "requested_steps",
        "diagnostics",
        "retention_summary",
        "longitudinal_intensity_xz",
        "longitudinal_intensity_yz",
        "x_cut_um",
        "y_cut_um",
        "intensity_preview",
        "intensity_preview_metadata",
        "td_scalar_history",
        "td_preview_movie",
        "td_preview_movie_metadata",
    }
    _require_exact_keys(values, required, label="PR transverse-TD result")
    grid = values["grid_summary"]
    launch = values["launch_summary"]
    if not isinstance(grid, Mapping) or not isinstance(launch, Mapping):
        raise TransportCodecError(
            "PR transverse-TD result lacks grid or launch summary"
        )
    nx = _positive_dimension(grid, "Nx")
    ny = _positive_dimension(grid, "Ny")
    nz = _positive_dimension(grid, "Nz")
    nch = _positive_dimension(launch, "Nch")
    A_initial = _require_array(
        values, "A_initial", (nch, nx, ny), "complex"
    )
    A_final = _require_array(values, "A_final", (nch, nx, ny), "complex")
    retention = values["retention_summary"]
    if not isinstance(retention, Mapping):
        raise TransportCodecError("PR transverse-TD retention summary is invalid")
    try:
        policy = normalize_result_policy(retention.get("policy", FULL_RESULT_POLICY))
    except ValueError as exc:
        raise TransportCodecError(str(exc)) from exc
    expected_omitted = set(_FAST_OMITTED_FIELDS if policy == FAST_RESULT_POLICY else ())
    omitted = retention.get("omitted_fields", [])
    legacy_omitted = {"psi_initial", "psi_final"}
    if (
        not isinstance(omitted, (tuple, list))
        or set(omitted) not in (expected_omitted, legacy_omitted)
    ):
        raise TransportCodecError(
            "PR transverse-TD omitted fields disagree with result policy"
        )
    cuts_present = validate_longitudinal_cut_coordinates(values, grid)
    try:
        preview_present = validate_fast_intensity_preview(
            values["intensity_preview"], values["intensity_preview_metadata"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(str(exc)) from exc
    history = values["td_scalar_history"]
    if not isinstance(history, (tuple, list)) or any(
        not isinstance(row, Mapping) for row in history
    ):
        raise TransportCodecError("TD scalar history must contain mapping rows")
    movie = values["td_preview_movie"]
    movie_metadata = values["td_preview_movie_metadata"]
    if movie is not None and (
        not isinstance(movie, np.ndarray) or movie.dtype != np.uint8
    ):
        raise TransportCodecError("TD preview movie must be uint8")
    if movie is not None and not isinstance(movie_metadata, Mapping):
        raise TransportCodecError("TD preview movie lacks metadata")
    psi_initial = psi_final = None
    if policy == FULL_RESULT_POLICY:
        psi_initial = _require_array(
            values, "psi_initial", (nz, nx, ny), "real"
        )
        psi_final = _require_array(values, "psi_final", (nz, nx, ny), "real")
        if values["source_intensity_stack"] is not None:
            _require_array(
                values, "source_intensity_stack", (nz, nx, ny), "real"
            )
        if preview_present:
            raise TransportCodecError("Full transverse-TD result retained preview")
    elif (
        values["psi_initial"] is not None
        or values["psi_final"] is not None
        or values["source_intensity_stack"] is not None
    ):
        raise TransportCodecError(
            "PR transverse-TD Fast result unexpectedly retained material volumes"
        )
    if policy == FAST_RESULT_POLICY:
        if cuts_present:
            _require_array(values, "longitudinal_intensity_xz", (nz, nx), "real")
            _require_array(values, "longitudinal_intensity_yz", (nz, ny), "real")
    elif (
        values["longitudinal_intensity_xz"] is not None
        or values["longitudinal_intensity_yz"] is not None
    ):
        raise TransportCodecError(
            "PR transverse-TD Full result unexpectedly retained Fast cuts"
        )
    if A_initial.dtype != A_final.dtype:
        raise TransportCodecError(
            "PR transverse-TD initial/final optical dtypes disagree"
        )
    if psi_initial is not None and psi_initial.dtype != psi_final.dtype:
        raise TransportCodecError(
            "PR transverse-TD initial/final material dtypes disagree"
        )
    status = values["status"]
    if status not in {"completed", "cancelled"}:
        raise TransportCodecError("PR transverse-TD result status is invalid")
    completed = values["completed_steps"]
    requested = values["requested_steps"]
    if (
        type(completed) is not int
        or type(requested) is not int
        or completed < 0
        or requested < completed
    ):
        raise TransportCodecError(
            "PR transverse-TD completed/requested steps are inconsistent"
        )
    for name in ("time_normalized", "power_initial", "power_final"):
        value = values[name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TransportCodecError(
                f"PR transverse-TD result {name} must be finite"
            )
    if float(values["time_normalized"]) < 0.0:
        raise TransportCodecError(
            "PR transverse-TD result time_normalized must be nonnegative"
        )
    backend = values["backend_summary"]
    if not isinstance(backend, Mapping) or backend.get("backend") not in {
        "numpy",
        "cupy",
    }:
        raise TransportCodecError(
            "PR transverse-TD result backend provenance is invalid"
        )
    expected_real = backend.get("real_dtype")
    expected_complex = backend.get("complex_dtype")
    if (
        (psi_final is not None and str(psi_final.dtype) != expected_real)
        or str(A_final.dtype) != expected_complex
    ):
        raise TransportCodecError(
            "PR transverse-TD result dtypes disagree with backend provenance"
        )
    profile = values["resolved_profile"]
    diagnostics = values["diagnostics"]
    if not isinstance(profile, Mapping):
        raise TransportCodecError(
            "PR transverse-TD result resolved_profile must be a mapping"
        )
    if not isinstance(diagnostics, Mapping):
        raise TransportCodecError(
            "PR transverse-TD result diagnostics must be a mapping"
        )
    cancellation_stage = diagnostics.get("cancellation_observed_stage")
    if status == "completed":
        if completed != requested:
            raise TransportCodecError(
                "completed PR transverse-TD result did not reach requested steps"
            )
        if cancellation_stage is not None:
            raise TransportCodecError(
                "completed PR transverse-TD result has cancellation provenance"
            )
    else:
        if completed >= requested:
            raise TransportCodecError(
                "cancelled PR transverse-TD result reached all requested steps"
            )
        if cancellation_stage not in _CANCELLATION_OBSERVED_STAGES:
            raise TransportCodecError(
                "cancelled PR transverse-TD result lacks valid cancellation provenance"
            )
    solver = profile.get("solver")
    if not isinstance(solver, Mapping):
        raise TransportCodecError(
            "PR transverse-TD result resolved solver provenance is missing"
        )
    try:
        expected_time = completed * float(solver["dt_normalized"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            "PR transverse-TD result timestep provenance is invalid"
        ) from exc
    if not math.isclose(
        float(values["time_normalized"]),
        expected_time,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise TransportCodecError(
            "PR transverse-TD normalized time disagrees with accepted steps"
        )


def decode_pr_transverse_timedependent_transport_result(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> PRTransverseRunResult:
    """Decode and strictly validate one canonical transverse-TD result."""

    try:
        values = unpack_portable(dict(metadata), arrays)
        for name in (
            "longitudinal_intensity_xz",
            "longitudinal_intensity_yz",
            "x_cut_um",
            "y_cut_um",
            "intensity_preview",
            "intensity_preview_metadata",
            "td_preview_movie",
            "td_preview_movie_metadata",
            "source_intensity_stack",
        ):
            values.setdefault(name, None)
        values.setdefault("td_scalar_history", ())
        values.setdefault("retention_summary", {
            "policy": FULL_RESULT_POLICY,
            "omitted_fields": [],
        })
        _validate_result(values)
        return PRTransverseRunResult(
            A_initial=values["A_initial"],
            A_final=values["A_final"],
            psi_initial=values["psi_initial"],
            psi_final=values["psi_final"],
            power_initial=float(values["power_initial"]),
            power_final=float(values["power_final"]),
            completed_steps=values["completed_steps"],
            time_normalized=float(values["time_normalized"]),
            grid_summary=dict(values["grid_summary"]),
            launch_summary=dict(values["launch_summary"]),
            backend_summary=dict(values["backend_summary"]),
            resolved_profile=dict(values["resolved_profile"]),
            status=values["status"],
            requested_steps=values["requested_steps"],
            diagnostics=dict(values["diagnostics"]),
            retention_summary=dict(values["retention_summary"]),
            longitudinal_intensity_xz=values["longitudinal_intensity_xz"],
            longitudinal_intensity_yz=values["longitudinal_intensity_yz"],
            x_cut_um=values["x_cut_um"],
            y_cut_um=values["y_cut_um"],
            intensity_preview=values["intensity_preview"],
            intensity_preview_metadata=values["intensity_preview_metadata"],
            source_intensity_stack=values["source_intensity_stack"],
            td_scalar_history=tuple(values["td_scalar_history"]),
            td_preview_movie=values["td_preview_movie"],
            td_preview_movie_metadata=values["td_preview_movie_metadata"],
        )
    except TransportCodecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TransportCodecError(
            f"invalid PR transverse-TD result payload: {exc}"
        ) from exc


PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC = TransportCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    request_codec_id=PR_TRANSVERSE_TIMEDEPENDENT_REQUEST_CODEC_ID,
    request_codec_version=PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION,
    request_type=PRTransverseRunRequest,
    encode_request=encode_pr_transverse_timedependent_transport_request,
    decode_request=decode_pr_transverse_timedependent_transport_request,
    result_codec_id=PR_TRANSVERSE_TIMEDEPENDENT_RESULT_CODEC_ID,
    result_codec_version=PR_TRANSVERSE_TIMEDEPENDENT_RESULT_CODEC_VERSION,
    result_type=PRTransverseRunResult,
    encode_result=encode_pr_transverse_timedependent_transport_result,
    decode_result=decode_pr_transverse_timedependent_transport_result,
    encode_result_projection=encode_pr_transverse_timedependent_transport_result,
    compatible_request_codec_versions=(
        _PR_TRANSVERSE_TIMEDEPENDENT_PREVIOUS_TRANSPORT_CODEC_VERSION,
    ),
    compatible_result_codec_versions=(
        _PR_TRANSVERSE_TIMEDEPENDENT_PREVIOUS_TRANSPORT_CODEC_VERSION,
        PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC_VERSION,
    ),
)


__all__ = [
    "PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC",
    "decode_pr_transverse_timedependent_transport_request",
    "decode_pr_transverse_timedependent_transport_result",
    "encode_pr_transverse_timedependent_transport_request",
    "encode_pr_transverse_timedependent_transport_result",
]
