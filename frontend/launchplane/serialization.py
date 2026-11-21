"""JSON serialization helpers for LaunchPlane beam definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from launchplane.model import BeamDefinition, BeamStackDefinition

SCHEMA_VERSION = 4


def beam_stack_to_dict(stack: BeamStackDefinition) -> dict[str, Any]:
    """Convert a validated beam stack to a JSON-compatible dictionary."""
    stack.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "beams": [asdict(beam) for beam in stack.beams],
    }


def beam_stack_from_dict(data: Mapping[str, Any]) -> BeamStackDefinition:
    """Construct and validate a beam stack from serialized data."""
    if not isinstance(data, Mapping):
        raise TypeError("Serialized beam stack must be a mapping")

    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r}; physical intent requires schema 4. "
            "Recreate/re-export obsolete development definitions; laboratory waists "
            "and wavevectors cannot be reinterpreted as external physical intent."
        )
    if set(data) != {"schema_version", "beams"}:
        raise ValueError("Unexpected physical-intent stack fields")

    beam_items = data.get("beams")
    if not isinstance(beam_items, list):
        raise ValueError("Serialized beam stack must contain a 'beams' list")

    beams: list[BeamDefinition] = []
    for index, item in enumerate(beam_items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Beam entry {index} must be a mapping")
        beam_data = dict(item)
        expected = {f.name for f in fields(BeamDefinition)}
        if set(beam_data) != expected:
            raise ValueError(
                f"Invalid beam entry {index}: schema-4 fields must be exact; "
                f"missing={sorted(expected - set(beam_data))}, "
                f"unexpected={sorted(set(beam_data) - expected)}"
            )
        try:
            beam = BeamDefinition(**beam_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid beam entry {index}: {exc}") from exc
        beams.append(beam)

    stack = BeamStackDefinition(beams=tuple(beams))
    stack.validate()
    return stack


def save_beam_stack_json(
    stack: BeamStackDefinition,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Serialize a beam stack to a UTF-8 JSON file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = beam_stack_to_dict(stack)
    output_path.write_text(
        json.dumps(payload, indent=indent, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_beam_stack_json(path: str | Path) -> BeamStackDefinition:
    """Load and validate a beam stack from a UTF-8 JSON file."""
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc
    return beam_stack_from_dict(payload)


__all__ = [
    "SCHEMA_VERSION",
    "beam_stack_to_dict",
    "beam_stack_from_dict",
    "save_beam_stack_json",
    "load_beam_stack_json",
]
