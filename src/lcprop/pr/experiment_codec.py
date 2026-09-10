"""Material-owned experiment request codecs for photorefractive workflows."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import ChannelLaunchElements
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
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_MATERIAL_ID,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.image_amplification import (
    PR_IMAGE_AMPLIFICATION_WORKFLOW,
    PRImageAmplificationExperimentRequest,
)
from lcprop.pr.portable_launch import (
    PortableLaunchPayloadError,
    decode_launch_elements,
    encode_launch_elements,
)
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseMaterialResponseSpec,
    PRTransverseProjectionProfile,
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    PRTransverseTransportProfile,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseStaticMaterialSolverOptions,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)


PR_EXPERIMENT_REQUEST_SCHEMA_VERSION = 3
_PR_LEGACY_EXPERIMENT_REQUEST_SCHEMA_VERSION = 1
_PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION = 2


def _encode_launch_elements(
    assignments: tuple[ChannelLaunchElements, ...],
) -> list[dict[str, Any]]:
    try:
        return encode_launch_elements(assignments)
    except PortableLaunchPayloadError as exc:
        raise ExperimentPayloadError(str(exc)) from exc


def _decode_launch_elements(
    value: Any,
    *,
    n_channels: int,
) -> tuple[ChannelLaunchElements, ...]:
    try:
        return decode_launch_elements(value, n_channels=n_channels)
    except PortableLaunchPayloadError as exc:
        raise ExperimentPayloadError(str(exc)) from exc


def _reject_runtime_state(request: PRRunRequest | PRStaticRunRequest) -> None:
    populated = tuple(
        name
        for name in ("initial_A", "initial_E")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "PR experiment requests cannot persist runtime state "
            f"({', '.join(populated)}); use checkpoint persistence for "
            "continuation state"
        )


def _validate_common(request: PRRunRequest | PRStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()
    if isinstance(request, PRStaticRunRequest):
        request.material_response.validate()


def _encode_common(request: PRRunRequest | PRStaticRunRequest) -> dict:
    _reject_runtime_state(request)
    _validate_common(request)
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": encode_beam_stack(request.beams),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "launch_elements": _encode_launch_elements(request.launch_elements),
    }


def encode_pr_timedependent_request(request: PRRunRequest) -> dict:
    if not isinstance(request, PRRunRequest):
        raise TypeError("request must be a PRRunRequest")
    return _encode_common(request)


def encode_pr_static_request(request: PRStaticRunRequest) -> dict:
    if not isinstance(request, PRStaticRunRequest):
        raise TypeError("request must be a PRStaticRunRequest")
    payload = _encode_common(request)
    payload["material_response"] = asdict(request.material_response)
    return payload


def _validate_transverse_request(
    request: PRTransverseRunRequest | PRTransverseStaticRunRequest,
) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    if isinstance(request, PRTransverseStaticRunRequest):
        request.material_response.validate_configuration(
            material_applied_field=request.material.applied_field,
            transport=request.transport,
            dielectric=request.dielectric,
            boundary=request.boundary,
            projection=request.projection,
        )
    elif request.boundary.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
        raise ValueError(
            "time-dependent Profile v1 does not support the periodic biased "
            "electrical profile"
        )
    request.solver.validate()
    request.backend.validate()
    if request.scattering is not None:
        request.scattering.validate()


def encode_pr_transverse_static_request(
    request: PRTransverseStaticRunRequest,
) -> dict:
    """Encode canonical transverse-static inputs without runtime state."""

    if not isinstance(request, PRTransverseStaticRunRequest):
        raise TypeError("request must be a PRTransverseStaticRunRequest")
    populated = tuple(
        name
        for name in ("initial_A", "initial_psi")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "PR transverse-static experiment requests cannot persist runtime "
            f"state ({', '.join(populated)}); use result transport or an "
            "explicit continuation mechanism for runtime state"
        )
    _validate_transverse_request(request)
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": encode_beam_stack(request.beams),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "material_response": asdict(request.material_response),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
        "launch_elements": _encode_launch_elements(request.launch_elements),
    }


def encode_pr_transverse_timedependent_request(
    request: PRTransverseRunRequest,
) -> dict:
    """Encode canonical transverse-TD inputs without runtime state."""

    if not isinstance(request, PRTransverseRunRequest):
        raise TypeError("request must be a PRTransverseRunRequest")
    populated = tuple(
        name
        for name in ("initial_A", "initial_psi")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "PR transverse-TD experiment requests cannot persist runtime "
            f"state ({', '.join(populated)}); use result transport for "
            "runtime state"
        )
    _validate_transverse_request(request)
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": encode_beam_stack(request.beams),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
        "launch_elements": _encode_launch_elements(request.launch_elements),
    }


def _payload(value: Any, *, reduced_static: bool = False) -> dict[str, Any]:
    payload = require_mapping(value, name="PR request_payload")
    version = payload.get("schema_version")
    if type(version) is not int or version not in (
        _PR_LEGACY_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
    ):
        raise ExperimentSchemaError(
            f"unsupported PR experiment request schema version: {version!r}"
        )
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "beams",
            "solver",
            "backend",
        },
        optional=(
            (
                {"launch_elements"}
                if version
                >= _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION
                else set()
            )
            | (
                {"material_response"}
                if reduced_static
                and version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
                else set()
            )
        ),
        name="PR request_payload",
    )
    if version >= _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION and (
        "launch_elements" not in payload
    ):
        raise ExperimentPayloadError(
            "PR request_payload is missing required field(s): launch_elements"
        )
    return payload


def _decode_common(payload: dict[str, Any]) -> dict[str, Any]:
    beams = decode_beam_stack(payload["beams"])
    return {
        "grid": GridSpec(
            **dataclass_values(
                GridSpec,
                payload["grid"],
                name="PR request_payload.grid",
            )
        ),
        "material": PRMaterialSpec(
            **dataclass_values(
                PRMaterialSpec,
                payload["material"],
                name="PR request_payload.material",
            )
        ),
        "beams": beams,
        "backend": BackendSpec(
            **dataclass_values(
                BackendSpec,
                payload["backend"],
                name="PR request_payload.backend",
            )
        ),
        "launch_elements": _decode_launch_elements(
            payload.get("launch_elements", []),
            n_channels=len(beams.channels),
        ),
    }


def decode_pr_timedependent_request(value: Any) -> PRRunRequest:
    payload = _payload(value)
    try:
        request = PRRunRequest(
            **_decode_common(payload),
            solver=PRSolverOptions(
                **dataclass_values(
                    PRSolverOptions,
                    payload["solver"],
                    name="PR request_payload.solver",
                )
            ),
        )
        _validate_common(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR time-dependent request: {exc}"
        ) from exc
    return request


def decode_pr_static_request(value: Any) -> PRStaticRunRequest:
    payload = _payload(value, reduced_static=True)
    solver_values = dataclass_values(
        PRStaticWorkflowOptions,
        payload["solver"],
        name="PR request_payload.solver",
    )
    material_solver_values = solver_values.pop("material_solver")
    material_solver = None
    if material_solver_values is not None:
        material_solver = PRStaticSolverOptions(
            **dataclass_values(
                PRStaticSolverOptions,
                material_solver_values,
                name="PR request_payload.solver.material_solver",
            )
        )
    try:
        request = PRStaticRunRequest(
            **_decode_common(payload),
            solver=PRStaticWorkflowOptions(
                material_solver=material_solver,
                **solver_values,
            ),
            material_response=PRTransverseMaterialResponseSpec(
                **(
                    dataclass_values(
                        PRTransverseMaterialResponseSpec,
                        payload["material_response"],
                        name="PR request_payload.material_response",
                    )
                    if "material_response" in payload
                    else {}
                )
            ),
        )
        _validate_common(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid PR static request: {exc}") from exc
    return request


def decode_pr_transverse_static_request(
    value: Any,
) -> PRTransverseStaticRunRequest:
    payload = require_mapping(value, name="PR transverse-static request_payload")
    version = payload.get("schema_version")
    if type(version) is not int or version not in (
        _PR_LEGACY_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
    ):
        raise ExperimentSchemaError(
            "unsupported PR transverse-static experiment request schema "
            f"version: {version!r}"
        )
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "beams",
            "transport",
            "dielectric",
            "boundary",
            "projection",
            "solver",
            "backend",
            "scattering",
        }
        | (
            {"launch_elements"}
            if version >= _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION
            else set()
        )
        | (
            {"material_response"}
            if version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
            else set()
        ),
        name="PR transverse-static request_payload",
    )
    if version >= _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION and (
        "launch_elements" not in payload
    ):
        raise ExperimentPayloadError(
            "PR transverse-static request_payload is missing required "
            "field(s): launch_elements"
        )
    solver_values = dataclass_values(
        PRTransverseStaticWorkflowOptions,
        payload["solver"],
        name="PR transverse-static request_payload.solver",
    )
    material_solver_values = solver_values.pop("material_solver")
    discrete_corrector_values = solver_values.pop("discrete_corrector")
    scattering_values = payload["scattering"]
    try:
        beams = decode_beam_stack(payload["beams"])
        request = PRTransverseStaticRunRequest(
            grid=GridSpec(
                **dataclass_values(
                    GridSpec,
                    payload["grid"],
                    name="PR transverse-static request_payload.grid",
                )
            ),
            beams=beams,
            material=PRMaterialSpec(
                **dataclass_values(
                    PRMaterialSpec,
                    payload["material"],
                    name="PR transverse-static request_payload.material",
                )
            ),
            transport=PRTransverseTransportProfile(
                **dataclass_values(
                    PRTransverseTransportProfile,
                    payload["transport"],
                    name="PR transverse-static request_payload.transport",
                )
            ),
            dielectric=PRTransverseDielectricProfile(
                **dataclass_values(
                    PRTransverseDielectricProfile,
                    payload["dielectric"],
                    name="PR transverse-static request_payload.dielectric",
                )
            ),
            boundary=PRTransverseBoundaryProfile(
                **dataclass_values(
                    PRTransverseBoundaryProfile,
                    payload["boundary"],
                    name="PR transverse-static request_payload.boundary",
                )
            ),
            projection=PRTransverseProjectionProfile(
                **dataclass_values(
                    PRTransverseProjectionProfile,
                    payload["projection"],
                    name="PR transverse-static request_payload.projection",
                )
            ),
            material_response=PRTransverseMaterialResponseSpec(
                **(
                    dataclass_values(
                        PRTransverseMaterialResponseSpec,
                        payload["material_response"],
                        name=(
                            "PR transverse-static request_payload."
                            "material_response"
                        ),
                    )
                    if "material_response" in payload
                    else {}
                )
            ),
            solver=PRTransverseStaticWorkflowOptions(
                material_solver=PRTransverseStaticMaterialSolverOptions(
                    **dataclass_values(
                        PRTransverseStaticMaterialSolverOptions,
                        material_solver_values,
                        name=(
                            "PR transverse-static request_payload.solver."
                            "material_solver"
                        ),
                    )
                ),
                discrete_corrector=PRTransverseDiscreteStaticCorrectorOptions(
                    **dataclass_values(
                        PRTransverseDiscreteStaticCorrectorOptions,
                        discrete_corrector_values,
                        name=(
                            "PR transverse-static request_payload.solver."
                            "discrete_corrector"
                        ),
                    )
                ),
                **solver_values,
            ),
            backend=BackendSpec(
                **dataclass_values(
                    BackendSpec,
                    payload["backend"],
                    name="PR transverse-static request_payload.backend",
                )
            ),
            initial_A=None,
            initial_psi=None,
            scattering=(
                None
                if scattering_values is None
                else PRCanonicalScatteringSpec(
                    **dataclass_values(
                        PRCanonicalScatteringSpec,
                        scattering_values,
                        name="PR transverse-static request_payload.scattering",
                    )
                )
            ),
            launch_elements=_decode_launch_elements(
                payload.get("launch_elements", []),
                n_channels=len(beams.channels),
            ),
        )
        _validate_transverse_request(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR transverse-static request: {exc}"
        ) from exc
    return request


def decode_pr_transverse_timedependent_request(
    value: Any,
) -> PRTransverseRunRequest:
    """Decode one canonical transverse-TD experiment request."""

    payload = require_mapping(value, name="PR transverse-TD request_payload")
    version = payload.get("schema_version")
    if (
        type(version) is not int
        or version not in (
            _PR_LAUNCH_ELEMENTS_EXPERIMENT_REQUEST_SCHEMA_VERSION,
            PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        )
    ):
        raise ExperimentSchemaError(
            "unsupported PR transverse-TD experiment request schema version: "
            f"{version!r}"
        )
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "beams",
            "transport",
            "dielectric",
            "boundary",
            "projection",
            "solver",
            "backend",
            "scattering",
            "launch_elements",
        },
        name="PR transverse-TD request_payload",
    )
    scattering_values = payload["scattering"]
    try:
        beams = decode_beam_stack(payload["beams"])
        request = PRTransverseRunRequest(
            grid=GridSpec(
                **dataclass_values(
                    GridSpec,
                    payload["grid"],
                    name="PR transverse-TD request_payload.grid",
                )
            ),
            beams=beams,
            material=PRMaterialSpec(
                **dataclass_values(
                    PRMaterialSpec,
                    payload["material"],
                    name="PR transverse-TD request_payload.material",
                )
            ),
            transport=PRTransverseTransportProfile(
                **dataclass_values(
                    PRTransverseTransportProfile,
                    payload["transport"],
                    name="PR transverse-TD request_payload.transport",
                )
            ),
            dielectric=PRTransverseDielectricProfile(
                **dataclass_values(
                    PRTransverseDielectricProfile,
                    payload["dielectric"],
                    name="PR transverse-TD request_payload.dielectric",
                )
            ),
            boundary=PRTransverseBoundaryProfile(
                **dataclass_values(
                    PRTransverseBoundaryProfile,
                    payload["boundary"],
                    name="PR transverse-TD request_payload.boundary",
                )
            ),
            projection=PRTransverseProjectionProfile(
                **dataclass_values(
                    PRTransverseProjectionProfile,
                    payload["projection"],
                    name="PR transverse-TD request_payload.projection",
                )
            ),
            solver=PRTransverseSolverOptions(
                **dataclass_values(
                    PRTransverseSolverOptions,
                    payload["solver"],
                    name="PR transverse-TD request_payload.solver",
                )
            ),
            backend=BackendSpec(
                **dataclass_values(
                    BackendSpec,
                    payload["backend"],
                    name="PR transverse-TD request_payload.backend",
                )
            ),
            initial_A=None,
            initial_psi=None,
            scattering=(
                None
                if scattering_values is None
                else PRCanonicalScatteringSpec(
                    **dataclass_values(
                        PRCanonicalScatteringSpec,
                        scattering_values,
                        name="PR transverse-TD request_payload.scattering",
                    )
                )
            ),
            launch_elements=_decode_launch_elements(
                payload["launch_elements"],
                n_channels=len(beams.channels),
            ),
        )
        _validate_transverse_request(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR transverse-TD request: {exc}"
        ) from exc
    return request


def _encode_image_base_request(workflow_id: str, request: Any) -> dict[str, Any]:
    if workflow_id == PR_TIMEDEPENDENT_WORKFLOW:
        return encode_pr_timedependent_request(request)
    if workflow_id == PR_STATIC_WORKFLOW:
        return encode_pr_static_request(request)
    if workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
        return encode_pr_transverse_static_request(request)
    if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
        return encode_pr_transverse_timedependent_request(request)
    raise ExperimentPayloadError(
        "Image Amplification base operation is not persistable: "
        f"{workflow_id!r}"
    )


def _decode_image_base_request(workflow_id: str, value: Any) -> Any:
    if workflow_id == PR_TIMEDEPENDENT_WORKFLOW:
        return decode_pr_timedependent_request(value)
    if workflow_id == PR_STATIC_WORKFLOW:
        return decode_pr_static_request(value)
    if workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
        return decode_pr_transverse_static_request(value)
    if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
        return decode_pr_transverse_timedependent_request(value)
    raise ExperimentPayloadError(
        "Image Amplification base operation is not registered or persistable: "
        f"{workflow_id!r}"
    )


def encode_pr_image_amplification_request(
    request: PRImageAmplificationExperimentRequest,
) -> dict[str, Any]:
    """Encode the declarative composite experiment without runtime arrays."""

    if not isinstance(request, PRImageAmplificationExperimentRequest):
        raise TypeError("request must be a PRImageAmplificationExperimentRequest")
    request.validate()
    base_request = replace(request.base_request, launch_elements=())
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "base_workflow_id": request.base_workflow_id,
        "base_request": _encode_image_base_request(
            request.base_workflow_id,
            base_request,
        ),
        "launch_configuration": {
            "beams": encode_beam_stack(request.launch_configuration.beams),
            "channel_elements": _encode_launch_elements(
                request.launch_configuration.channel_elements
            ),
        },
        "pump_channel_index": request.pump_channel_index,
        "signal_channel_index": request.signal_channel_index,
    }


def decode_pr_image_amplification_request(
    value: Any,
) -> PRImageAmplificationExperimentRequest:
    payload = require_mapping(
        value,
        name="PR Image Amplification request_payload",
    )
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "base_workflow_id",
            "base_request",
            "launch_configuration",
            "pump_channel_index",
            "signal_channel_index",
        },
        name="PR Image Amplification request_payload",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != PR_EXPERIMENT_REQUEST_SCHEMA_VERSION:
        raise ExperimentSchemaError(
            "unsupported PR Image Amplification experiment request schema "
            f"version: {version!r}"
        )
    workflow_id = payload["base_workflow_id"]
    if not isinstance(workflow_id, str):
        raise ExperimentPayloadError("base_workflow_id must be a string")
    launch_payload = require_mapping(
        payload["launch_configuration"],
        name="launch_configuration",
    )
    require_exact_keys(
        launch_payload,
        required={"beams", "channel_elements"},
        name="launch_configuration",
    )
    beams = decode_beam_stack(launch_payload["beams"])
    try:
        channel_elements = _decode_launch_elements(
            launch_payload["channel_elements"],
            n_channels=len(beams.channels),
        )
        base_request = replace(
            _decode_image_base_request(
                workflow_id,
                payload["base_request"],
            ),
            launch_elements=channel_elements,
        )
        request = PRImageAmplificationExperimentRequest(
            base_workflow_id=workflow_id,
            base_request=base_request,
            launch_configuration=LaunchConfiguration(
                beams=beams,
                channel_elements=channel_elements,
            ),
            pump_channel_index=payload["pump_channel_index"],
            signal_channel_index=payload["signal_channel_index"],
        )
        request.validate()
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR Image Amplification request: {exc}"
        ) from exc
    return request


PR_STATIC_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_STATIC_WORKFLOW,
    request_type=PRStaticRunRequest,
    encode_request=encode_pr_static_request,
    decode_request=decode_pr_static_request,
)

PR_TIMEDEPENDENT_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    request_type=PRRunRequest,
    encode_request=encode_pr_timedependent_request,
    decode_request=decode_pr_timedependent_request,
)

PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
    request_type=PRTransverseStaticRunRequest,
    encode_request=encode_pr_transverse_static_request,
    decode_request=decode_pr_transverse_static_request,
)

PR_TRANSVERSE_TIMEDEPENDENT_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    request_type=PRTransverseRunRequest,
    encode_request=encode_pr_transverse_timedependent_request,
    decode_request=decode_pr_transverse_timedependent_request,
)

PR_IMAGE_AMPLIFICATION_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_IMAGE_AMPLIFICATION_WORKFLOW,
    request_type=PRImageAmplificationExperimentRequest,
    encode_request=encode_pr_image_amplification_request,
    decode_request=decode_pr_image_amplification_request,
)


__all__ = [
    "PR_EXPERIMENT_REQUEST_SCHEMA_VERSION",
    "PR_IMAGE_AMPLIFICATION_EXPERIMENT_CODEC",
    "PR_STATIC_EXPERIMENT_CODEC",
    "PR_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC",
    "PR_TRANSVERSE_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "decode_pr_static_request",
    "decode_pr_image_amplification_request",
    "decode_pr_timedependent_request",
    "decode_pr_transverse_static_request",
    "decode_pr_transverse_timedependent_request",
    "encode_pr_static_request",
    "encode_pr_image_amplification_request",
    "encode_pr_timedependent_request",
    "encode_pr_transverse_static_request",
    "encode_pr_transverse_timedependent_request",
]
