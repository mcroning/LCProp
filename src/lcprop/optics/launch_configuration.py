"""Immutable composition of canonical beams and launch-plane elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from lcprop.core.beams import BeamStack
from lcprop.optics.screens import (
    ChannelLaunchElements,
    validate_channel_launch_elements,
)


@dataclass(frozen=True)
class LaunchConfiguration:
    """One validated beam stack and its ordered per-channel launch elements."""

    beams: BeamStack
    channel_elements: tuple[ChannelLaunchElements, ...] = ()

    def __post_init__(self) -> None:
        self.beams.validate()
        validate_channel_launch_elements(
            self.channel_elements,
            n_channels=len(self.beams.channels),
        )

    @property
    def launch_elements(self) -> tuple[ChannelLaunchElements, ...]:
        """Return the ordered launch-element assignments."""

        return self.channel_elements

    def __iter__(self) -> Iterator[Any]:
        """Preserve the Stage-B2 ``beams, elements = ...`` convenience."""

        yield self.beams
        yield self.channel_elements


def reject_prepared_launch_conflict(
    initial_A: Any | None,
    launch_elements: tuple[ChannelLaunchElements, ...],
) -> None:
    """Reject ambiguous simultaneous raw and declarative launch inputs."""

    if initial_A is not None and launch_elements:
        raise ValueError(
            "initial_A is an already-prepared runtime launch and cannot be "
            "combined with declarative launch_elements"
        )


__all__ = ["LaunchConfiguration", "reject_prepared_launch_conflict"]
