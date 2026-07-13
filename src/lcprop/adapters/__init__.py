"""Adapters for optional external packages."""

from lcprop.adapters.launchplane import (
    beam_definition_to_channel,
    beam_stack_definition_to_lcprop,
)

__all__ = [
    "beam_definition_to_channel",
    "beam_stack_definition_to_lcprop",
]
