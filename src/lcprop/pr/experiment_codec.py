"""Material-owned experiment request codecs for photorefractive workflows."""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
import hashlib
from typing import Any

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
    validate_channel_launch_elements,
)
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
from lcprop.pr.image_sources import (
    PR_IMAGE_PREPROCESSING_POLICY_V1,
    PRImageSource,
)
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.transverse.specs import (
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
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


PR_EXPERIMENT_REQUEST_SCHEMA_VERSION = 2
_PR_LEGACY_EXPERIMENT_REQUEST_SCHEMA_VERSION = 1


def _encode_raster_source(source: RasterSource) -> dict[str, Any]:
    if not isinstance(source, RasterSource):
        raise ExperimentPayloadError("image source must be a RasterSource")
    pixels = np.asarray(source.grayscale)
    if pixels.dtype.kind not in "fiu" or pixels.ndim != 2:
        raise ExperimentPayloadError("image source pixels must be a real 2-D array")
    raw = pixels.tobytes(order="C")
    encoded_bytes = source.encoded_bytes
    if source.source_kind == "user" and encoded_bytes is None:
        raise ExperimentPayloadError(
            "user image source lacks embedded encoded bytes; reload the source "
            "from its original file before saving the experiment"
        )
    return {
        "source_kind": source.source_kind,
        "display_name": source.display_name,
        "basename": source.basename,
        "sha256": source.sha256,
        "width": source.width,
        "height": source.height,
        "encoded_format": source.encoded_format,
        "decoded_mode": source.decoded_mode,
        "preprocessing_policy": source.preprocessing_policy,
        "asset_id": source.asset_id,
        "alpha_policy": "discarded_not_an_optical_mask",
        "grayscale_dtype": pixels.dtype.str,
        "grayscale_shape": list(pixels.shape),
        "grayscale_sha256": hashlib.sha256(raw).hexdigest(),
        "grayscale_base64": base64.b64encode(raw).decode("ascii"),
        "encoded_bytes_base64": (
            None
            if encoded_bytes is None
            else base64.b64encode(encoded_bytes).decode("ascii")
        ),
    }


def _strict_base64(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise ExperimentPayloadError(f"{name} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ExperimentPayloadError(f"{name} is not valid base64") from exc


def _decode_raster_source(value: Any) -> RasterSource:
    payload = require_mapping(value, name="image source")
    require_exact_keys(
        payload,
        required={
            "source_kind",
            "display_name",
            "basename",
            "sha256",
            "width",
            "height",
            "encoded_format",
            "decoded_mode",
            "preprocessing_policy",
            "asset_id",
            "alpha_policy",
            "grayscale_dtype",
            "grayscale_shape",
            "grayscale_sha256",
            "grayscale_base64",
            "encoded_bytes_base64",
        },
        name="image source",
    )
    if payload["alpha_policy"] != "discarded_not_an_optical_mask":
        raise ExperimentPayloadError("unsupported image alpha policy")
    try:
        dtype = np.dtype(payload["grayscale_dtype"])
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError("invalid image grayscale dtype") from exc
    shape = payload["grayscale_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item < 1 for item in shape)
    ):
        raise ExperimentPayloadError(
            "image grayscale_shape must contain two positive integers"
        )
    if dtype.kind not in "fiu":
        raise ExperimentPayloadError("image grayscale dtype must be real numeric")
    raw = _strict_base64(payload["grayscale_base64"], name="grayscale_base64")
    if hashlib.sha256(raw).hexdigest() != payload["grayscale_sha256"]:
        raise ExperimentPayloadError("image grayscale checksum mismatch")
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_size:
        raise ExperimentPayloadError("image grayscale byte length is inconsistent")
    pixels = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
    encoded_value = payload["encoded_bytes_base64"]
    encoded = (
        None
        if encoded_value is None
        else _strict_base64(encoded_value, name="encoded_bytes_base64")
    )
    if payload["source_kind"] == "user":
        if encoded is None:
            raise ExperimentPayloadError("user image is missing embedded bytes")
        if hashlib.sha256(encoded).hexdigest() != payload["sha256"]:
            raise ExperimentPayloadError("embedded user-image checksum mismatch")
    try:
        source_type = (
            PRImageSource
            if payload["preprocessing_policy"]
            == PR_IMAGE_PREPROCESSING_POLICY_V1
            else RasterSource
        )
        return source_type(
            source_kind=payload["source_kind"],
            display_name=payload["display_name"],
            basename=payload["basename"],
            sha256=payload["sha256"],
            width=payload["width"],
            height=payload["height"],
            encoded_format=payload["encoded_format"],
            decoded_mode=payload["decoded_mode"],
            grayscale=pixels,
            preprocessing_policy=payload["preprocessing_policy"],
            asset_id=payload["asset_id"],
            encoded_bytes=encoded,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid image source: {exc}") from exc


def _encode_launch_elements(
    assignments: tuple[ChannelLaunchElements, ...],
) -> list[dict[str, Any]]:
    encoded = []
    for assignment in assignments:
        encoded.append(
            {
                "channel_index": assignment.channel_index,
                "elements": [
                    {
                        "element_type": "intensity_raster",
                        "interpretation": "intensity_transmission",
                        "source": _encode_raster_source(element.source),
                        "placement": asdict(element.placement),
                        "invert": element.invert,
                        "preprocessing_policy": element.preprocessing_policy,
                    }
                    for element in assignment.elements
                ],
            }
        )
    return encoded


def _decode_launch_elements(
    value: Any,
    *,
    n_channels: int,
) -> tuple[ChannelLaunchElements, ...]:
    if not isinstance(value, list):
        raise ExperimentPayloadError("launch_elements must be a JSON array")
    assignments = []
    for assignment_index, item in enumerate(value):
        assignment = require_mapping(
            item,
            name=f"launch_elements[{assignment_index}]",
        )
        require_exact_keys(
            assignment,
            required={"channel_index", "elements"},
            name=f"launch_elements[{assignment_index}]",
        )
        element_items = assignment["elements"]
        if not isinstance(element_items, list):
            raise ExperimentPayloadError(
                f"launch_elements[{assignment_index}].elements must be an array"
            )
        elements = []
        for element_index, item in enumerate(element_items):
            name = (
                f"launch_elements[{assignment_index}].elements[{element_index}]"
            )
            element = require_mapping(item, name=name)
            require_exact_keys(
                element,
                required={
                    "element_type",
                    "interpretation",
                    "source",
                    "placement",
                    "invert",
                    "preprocessing_policy",
                },
                name=name,
            )
            if element["element_type"] != "intensity_raster":
                raise ExperimentPayloadError(f"{name} has unsupported element_type")
            if element["interpretation"] != "intensity_transmission":
                raise ExperimentPayloadError(f"{name} has unsupported interpretation")
            elements.append(
                IntensityRasterScreen(
                    source=_decode_raster_source(element["source"]),
                    placement=ScreenPlacement(
                        **dataclass_values(
                            ScreenPlacement,
                            element["placement"],
                            name=f"{name}.placement",
                        )
                    ),
                    invert=element["invert"],
                    preprocessing_policy=element["preprocessing_policy"],
                )
            )
        assignments.append(
            ChannelLaunchElements(
                channel_index=assignment["channel_index"],
                elements=tuple(elements),
            )
        )
    result = tuple(assignments)
    try:
        validate_channel_launch_elements(result, n_channels=n_channels)
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid launch_elements: {exc}") from exc
    return result


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
    return _encode_common(request)


def _validate_transverse_static(request: PRTransverseStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
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
    _validate_transverse_static(request)
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


def _payload(value: Any) -> dict[str, Any]:
    payload = require_mapping(value, name="PR request_payload")
    version = payload.get("schema_version")
    if type(version) is not int or version not in (
        _PR_LEGACY_EXPERIMENT_REQUEST_SCHEMA_VERSION,
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
            {"launch_elements"}
            if version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
            else set()
        ),
        name="PR request_payload",
    )
    if version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION and (
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
    payload = _payload(value)
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
        },
        optional=(
            {"launch_elements"}
            if version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION
            else set()
        ),
        name="PR transverse-static request_payload",
    )
    if version == PR_EXPERIMENT_REQUEST_SCHEMA_VERSION and (
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
        _validate_transverse_static(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR transverse-static request: {exc}"
        ) from exc
    return request


def _encode_image_base_request(workflow_id: str, request: Any) -> dict[str, Any]:
    if workflow_id == PR_TIMEDEPENDENT_WORKFLOW:
        return encode_pr_timedependent_request(request)
    if workflow_id == PR_STATIC_WORKFLOW:
        return encode_pr_static_request(request)
    if workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
        return encode_pr_transverse_static_request(request)
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
    "decode_pr_static_request",
    "decode_pr_image_amplification_request",
    "decode_pr_timedependent_request",
    "decode_pr_transverse_static_request",
    "encode_pr_static_request",
    "encode_pr_image_amplification_request",
    "encode_pr_timedependent_request",
    "encode_pr_transverse_static_request",
]
