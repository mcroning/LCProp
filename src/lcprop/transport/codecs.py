"""Material-neutral composition records for portable workflow codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from lcprop.transport.result_policy import (
    FULL_RESULT_POLICY,
    normalize_result_policy,
)


@dataclass(frozen=True)
class PortablePayload:
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedRequest:
    payload: PortablePayload
    scientific_backend_requested: str


@dataclass(frozen=True)
class EncodedResult:
    payload: PortablePayload
    scientific_status: str
    converged: bool | None
    cancelled: bool
    termination_reason: str | None
    scientific_backend_resolved: str
    device_summary: Mapping[str, Any] | None = None
    result_policy: str = FULL_RESULT_POLICY


@dataclass(frozen=True)
class TransportCodec:
    material_id: str
    workflow_id: str
    request_codec_id: str
    request_codec_version: int
    request_type: type
    encode_request: Callable[[Any], EncodedRequest]
    decode_request: Callable[[Mapping[str, Any], Mapping[str, np.ndarray]], Any]
    result_codec_id: str
    result_codec_version: int
    result_type: type
    encode_result: Callable[[Any], EncodedResult]
    decode_result: Callable[[Mapping[str, Any], Mapping[str, np.ndarray]], Any]
    encode_result_projection: Callable[[Any, str], EncodedResult] | None = None

    def __post_init__(self) -> None:
        for name in (
            "material_id",
            "workflow_id",
            "request_codec_id",
            "result_codec_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        for name in ("request_codec_version", "result_codec_version"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "encode_request",
            "decode_request",
            "encode_result",
            "decode_result",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if (
            self.encode_result_projection is not None
            and not callable(self.encode_result_projection)
        ):
            raise TypeError("encode_result_projection must be callable or None")

    @property
    def key(self) -> tuple[str, str]:
        return self.material_id, self.workflow_id

    def encode_result_for_policy(self, result: Any, policy: str) -> EncodedResult:
        """Encode one codec-owned projection without material logic in callers."""

        resolved = normalize_result_policy(policy)
        if self.encode_result_projection is None:
            encoded = self.encode_result(result)
            if resolved != FULL_RESULT_POLICY:
                raise ValueError(
                    f"codec {self.result_codec_id!r} does not support Fast results"
                )
            return encoded
        encoded = self.encode_result_projection(result, resolved)
        if normalize_result_policy(encoded.result_policy) != resolved:
            raise ValueError("result codec returned the wrong result policy")
        return encoded


class TransportCodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[tuple[str, str], TransportCodec] = {}

    def register(self, codec: TransportCodec) -> None:
        if not isinstance(codec, TransportCodec):
            raise TypeError("codec must be a TransportCodec")
        if codec.key in self._codecs:
            raise ValueError(f"transport codec already registered for {codec.key!r}")
        self._codecs[codec.key] = codec

    def codec(self, material_id: str, workflow_id: str) -> TransportCodec:
        try:
            return self._codecs[(material_id, workflow_id)]
        except KeyError as exc:
            raise KeyError(
                "no transport codec registered for "
                f"material_id={material_id!r}, workflow_id={workflow_id!r}"
            ) from exc

    @property
    def codecs(self) -> tuple[TransportCodec, ...]:
        return tuple(self._codecs.values())


__all__ = [
    "EncodedRequest",
    "EncodedResult",
    "PortablePayload",
    "TransportCodec",
    "TransportCodecRegistry",
]
