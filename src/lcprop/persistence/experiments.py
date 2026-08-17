"""Material-aware, versioned experiment request persistence.

This module owns only the shared JSON envelope, codec composition, and
material/workflow dispatch.  Material packages own request payload semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

from lcprop.core.beams import BeamChannel, BeamStack


EXPERIMENT_FORMAT = "lcprop-experiment"
EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_FILE_EXTENSION = ".lcprop.json"


class ExperimentError(ValueError):
    """Base class for experiment persistence failures."""


class ExperimentFormatError(ExperimentError):
    """The JSON document is not a valid experiment envelope."""


class ExperimentSchemaError(ExperimentError):
    """An envelope or material-owned payload schema is unsupported."""


class ExperimentMaterialError(ExperimentError):
    """The material identity is unknown or incompatible with the caller."""

    def __init__(
        self,
        message: str,
        *,
        actual_material_id: str | None = None,
        expected_material_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_material_id = actual_material_id
        self.expected_material_id = expected_material_id


class ExperimentWorkflowError(ExperimentError):
    """No request codec is registered for the named workflow."""


class ExperimentPayloadError(ExperimentError):
    """A material-owned request payload is malformed or inconsistent."""


class ExperimentRuntimeStateError(ExperimentPayloadError):
    """A request contains execution state that belongs in a checkpoint."""


class ExperimentPresentationError(ExperimentPayloadError):
    """Optional editor metadata is malformed or disagrees with the request."""


EncodeRequest = Callable[[Any], Mapping[str, Any]]
DecodeRequest = Callable[[Mapping[str, Any]], Any]


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ExperimentFormatError(f"{name} must be a non-empty, trimmed string")


def require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return one payload object as a plain dictionary."""

    if not isinstance(value, Mapping):
        raise ExperimentPayloadError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ExperimentPayloadError(f"{name} keys must be strings")
    return dict(value)


def require_exact_keys(
    values: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
    optional: set[str] | None = None,
) -> None:
    """Reject missing or unexpected keys in a versioned payload object."""

    optional = set() if optional is None else set(optional)
    actual = set(values)
    missing = required - actual
    unexpected = actual - required - optional
    if missing:
        raise ExperimentPayloadError(
            f"{name} is missing required field(s): {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise ExperimentPayloadError(
            f"{name} has unexpected field(s): {', '.join(sorted(unexpected))}"
        )


def dataclass_values(
    dataclass_type: type,
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    """Validate an explicitly encoded dataclass object's field names."""

    values = require_mapping(value, name=name)
    require_exact_keys(
        values,
        required={item.name for item in fields(dataclass_type)},
        name=name,
    )
    return values


def encode_beam_stack(stack: BeamStack) -> dict[str, Any]:
    """Encode canonical propagation beams, including exact transverse q."""

    if not isinstance(stack, BeamStack):
        raise ExperimentPayloadError("beams must be a BeamStack")
    stack.validate()
    return {
        "coherence": stack.coherence,
        "channels": [asdict(channel) for channel in stack.channels],
    }


def decode_beam_stack(value: Any) -> BeamStack:
    """Decode a strict current-schema canonical beam stack."""

    values = require_mapping(value, name="beams")
    require_exact_keys(
        values,
        required={"coherence", "channels"},
        name="beams",
    )
    channel_items = values["channels"]
    if not isinstance(channel_items, list):
        raise ExperimentPayloadError("beams.channels must be a JSON array")
    channels = []
    for index, item in enumerate(channel_items):
        channel_values = dataclass_values(
            BeamChannel,
            item,
            name=f"beams.channels[{index}]",
        )
        try:
            channels.append(BeamChannel(**channel_values))
        except (TypeError, ValueError) as exc:
            raise ExperimentPayloadError(
                f"invalid beams.channels[{index}]: {exc}"
            ) from exc
    try:
        stack = BeamStack(
            channels=tuple(channels),
            coherence=values["coherence"],
        )
        stack.validate()
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid beams: {exc}") from exc
    return stack


@dataclass(frozen=True)
class ExperimentRequestCodec:
    """Request type and callables for one material/workflow identity."""

    material_id: str
    workflow_id: str
    request_type: type
    encode_request: EncodeRequest
    decode_request: DecodeRequest

    def __post_init__(self) -> None:
        _validate_identifier("material_id", self.material_id)
        _validate_identifier("workflow_id", self.workflow_id)
        if not isinstance(self.request_type, type):
            raise TypeError("request_type must be a type")
        if not callable(self.encode_request) or not callable(self.decode_request):
            raise TypeError("request codec functions must be callable")

    @property
    def key(self) -> tuple[str, str]:
        return self.material_id, self.workflow_id


class ExperimentCodecRegistry:
    """Explicit registry extensible by material/workflow codec registration."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], ExperimentRequestCodec] = {}

    def register(self, codec: ExperimentRequestCodec) -> None:
        if not isinstance(codec, ExperimentRequestCodec):
            raise TypeError("codec must be an ExperimentRequestCodec")
        if codec.key in self._by_key:
            raise ValueError(
                "experiment request codec already registered for "
                f"material={codec.material_id!r}, workflow={codec.workflow_id!r}"
            )
        self._by_key[codec.key] = codec

    @property
    def codecs(self) -> tuple[ExperimentRequestCodec, ...]:
        return tuple(self._by_key.values())

    @property
    def material_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(key[0] for key in self._by_key))

    def codec(self, material_id: str, workflow_id: str) -> ExperimentRequestCodec:
        key = (material_id, workflow_id)
        codec = self._by_key.get(key)
        if codec is not None:
            return codec
        if material_id not in self.material_ids:
            raise ExperimentMaterialError(
                f"unknown experiment material_id {material_id!r}"
            )
        raise ExperimentWorkflowError(
            "no experiment request codec is registered for "
            f"material_id={material_id!r}, workflow_id={workflow_id!r}"
        )


@dataclass(frozen=True)
class LoadedExperiment:
    """Decoded request plus its material identity and optional editor metadata."""

    material_id: str
    workflow_id: str
    request: Any
    presentation_payload: dict[str, Any]
    schema_version: int = EXPERIMENT_SCHEMA_VERSION


def _validate_launchplane_presentation(
    request: Any,
    presentation_payload: Mapping[str, Any],
) -> None:
    editor_value = presentation_payload.get("beam_editor")
    if editor_value is None:
        return
    editor = require_mapping(editor_value, name="presentation_payload.beam_editor")
    require_exact_keys(
        editor,
        required={"provider", "schema_version", "beam_stack"},
        name="presentation_payload.beam_editor",
    )
    if editor["provider"] != "launchplane":
        raise ExperimentPresentationError(
            "presentation_payload.beam_editor.provider must be 'launchplane'"
        )
    if type(editor["schema_version"]) is not int or editor["schema_version"] != 1:
        raise ExperimentPresentationError(
            "unsupported LaunchPlane presentation schema version: "
            f"{editor['schema_version']!r}"
        )
    try:
        from launchplane.serialization import beam_stack_from_dict

        from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop

        editor_stack = beam_stack_from_dict(editor["beam_stack"])
        canonical_stack = beam_stack_definition_to_lcprop(editor_stack)
    except (ImportError, TypeError, ValueError) as exc:
        raise ExperimentPresentationError(
            f"invalid LaunchPlane presentation metadata: {exc}"
        ) from exc
    request_beams = getattr(request, "beams", None)
    if canonical_stack != request_beams:
        raise ExperimentPresentationError(
            "LaunchPlane enabled beams disagree with canonical request beams"
        )


def _validated_presentation(
    request: Any,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        presentation = require_mapping(
            {} if value is None else value,
            name="presentation_payload",
        )
        _validate_launchplane_presentation(request, presentation)
    except ExperimentPresentationError:
        raise
    except ExperimentPayloadError as exc:
        raise ExperimentPresentationError(str(exc)) from exc
    return presentation


def write_experiment_file(
    path: str | Path,
    request: Any,
    *,
    material_id: str,
    workflow_id: str,
    registry: ExperimentCodecRegistry,
    presentation_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Encode one request and atomically write its experiment JSON file."""

    _validate_identifier("material_id", material_id)
    _validate_identifier("workflow_id", workflow_id)
    codec = registry.codec(material_id, workflow_id)
    if not isinstance(request, codec.request_type):
        raise ExperimentPayloadError(
            f"codec for {codec.key!r} requires {codec.request_type.__name__}, "
            f"got {type(request).__name__}"
        )
    try:
        request_payload = require_mapping(
            codec.encode_request(request),
            name="request_payload",
        )
    except ExperimentError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"could not encode request: {exc}") from exc
    presentation = _validated_presentation(request, presentation_payload)
    document = {
        "format": EXPERIMENT_FORMAT,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "material_id": material_id,
        "workflow_id": workflow_id,
        "request_payload": request_payload,
        "presentation_payload": presentation,
    }
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        encoded = json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"experiment payload is not portable JSON: {exc}"
        ) from exc
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    return output


def _read_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentFormatError(f"malformed experiment JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ExperimentFormatError("experiment file is not valid UTF-8") from exc
    if not isinstance(document, Mapping):
        raise ExperimentFormatError("experiment document must be a JSON object")
    document = dict(document)
    required = {
        "format",
        "schema_version",
        "material_id",
        "workflow_id",
        "request_payload",
    }
    optional = {"presentation_payload"}
    actual = set(document)
    if required - actual:
        raise ExperimentFormatError(
            "experiment document is missing required field(s): "
            + ", ".join(sorted(required - actual))
        )
    if actual - required - optional:
        raise ExperimentFormatError(
            "experiment document has unexpected field(s): "
            + ", ".join(sorted(actual - required - optional))
        )
    if document["format"] != EXPERIMENT_FORMAT:
        raise ExperimentFormatError(
            f"invalid experiment format marker: {document['format']!r}"
        )
    version = document["schema_version"]
    if type(version) is not int or version != EXPERIMENT_SCHEMA_VERSION:
        raise ExperimentSchemaError(
            f"unsupported experiment schema version: {version!r}"
        )
    for name in ("material_id", "workflow_id"):
        try:
            _validate_identifier(name, document[name])
        except ExperimentFormatError:
            raise
    return document


def read_experiment_file(
    path: str | Path,
    *,
    registry: ExperimentCodecRegistry,
    expected_material_id: str | None = None,
) -> LoadedExperiment:
    """Read, identity-check, decode, and validate one experiment request."""

    document = _read_document(Path(path))
    material_id = document["material_id"]
    workflow_id = document["workflow_id"]
    if expected_material_id is not None:
        _validate_identifier("expected_material_id", expected_material_id)
        if material_id != expected_material_id:
            raise ExperimentMaterialError(
                f"experiment material_id {material_id!r} does not match "
                f"expected material_id {expected_material_id!r}",
                actual_material_id=material_id,
                expected_material_id=expected_material_id,
            )
    codec = registry.codec(material_id, workflow_id)
    request_payload = require_mapping(
        document["request_payload"],
        name="request_payload",
    )
    try:
        request = codec.decode_request(request_payload)
    except ExperimentError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"could not decode request: {exc}") from exc
    if not isinstance(request, codec.request_type):
        raise ExperimentPayloadError(
            "experiment codec returned an incompatible request type: "
            f"expected {codec.request_type.__name__}, "
            f"got {type(request).__name__}"
        )
    presentation = _validated_presentation(
        request,
        document.get("presentation_payload"),
    )
    return LoadedExperiment(
        material_id=material_id,
        workflow_id=workflow_id,
        request=request,
        presentation_payload=presentation,
    )


__all__ = [
    "EXPERIMENT_FILE_EXTENSION",
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentCodecRegistry",
    "ExperimentError",
    "ExperimentFormatError",
    "ExperimentMaterialError",
    "ExperimentPayloadError",
    "ExperimentPresentationError",
    "ExperimentRequestCodec",
    "ExperimentRuntimeStateError",
    "ExperimentSchemaError",
    "ExperimentWorkflowError",
    "LoadedExperiment",
    "dataclass_values",
    "decode_beam_stack",
    "encode_beam_stack",
    "read_experiment_file",
    "require_exact_keys",
    "require_mapping",
    "write_experiment_file",
]
