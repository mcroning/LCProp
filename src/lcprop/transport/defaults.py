"""Explicit application composition for the initially supported transports."""

from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.transport_codec import LC_STATIC_TRANSPORT_CODEC
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.transport_codec import PR_TRANSVERSE_STATIC_TRANSPORT_CODEC
from lcprop.transport.codecs import TransportCodecRegistry


def default_transport_registry() -> TransportCodecRegistry:
    registry = TransportCodecRegistry()
    registry.register(LC_STATIC_TRANSPORT_CODEC)
    registry.register(PR_TRANSVERSE_STATIC_TRANSPORT_CODEC)
    return registry


def default_transport_operations():
    return (LC_STATIC_OPERATION, PR_TRANSVERSE_STATIC_OPERATION)


__all__ = ["default_transport_operations", "default_transport_registry"]
