"""Notebook beam objects backed by canonical channels and presentation metadata."""
from dataclasses import asdict, replace

from lcprop.adapters.launchplane import beam_definition_to_channel, beam_stack_to_launchplane
from lcprop.core.beams import BeamChannel, LEGACY_COHERENCE_GROUP
from lcprop.notebook.batch_requests import beam_stack

DISABLED_BEAMS = "notebook_disabled_beams"


def objects(beams, presentation):
    """Return ordered (channel, enabled) pairs for the editable Gaussian stack."""
    if "beam_editor" in presentation:
        from launchplane.serialization import beam_stack_from_dict

        stack = beam_stack_from_dict(presentation["beam_editor"]["beam_stack"])
        return tuple((beam_definition_to_channel(beam), beam.enabled) for beam in stack.beams)
    disabled = presentation.get(DISABLED_BEAMS, [])
    if not isinstance(disabled, list):
        raise ValueError("Invalid notebook disabled beam list")
    if any(not isinstance(item, dict) or set(item) != set(BeamChannel.__dataclass_fields__)
           for item in disabled):
        raise ValueError("Invalid notebook disabled beam fields")
    return (*((channel, True) for channel in beams.channels),
            *((BeamChannel(**item), False) for item in disabled))


def validate_objects(items):
    names = set()
    for channel, _ in items:
        channel.validate()
        if not isinstance(channel.name, str) or not channel.name.strip() or channel.name in names:
            raise ValueError("Beam names must be nonempty and unique")
        if channel.wavelength_um != 0.633 or channel.n_ext != 1.0:
            raise ValueError("Beam editor requires 0.633 µm Gaussian beams with external index 1")
        names.add(channel.name)
    if not 1 <= sum(enabled for _, enabled in items) <= 2:
        raise ValueError("Keep one or two beams enabled")


def edited_stack(original, items, presentation, coherence):
    validate_objects(items)
    enabled = tuple(channel for channel, active in items if active)
    # A re-enabled object can come from a legacy all-implicit stack. Resolve
    # implicit grouping only when mixing it with explicit objects or regrouping.
    legacy = [ch.coherence_group == LEGACY_COHERENCE_GROUP for ch in enabled]
    if any(legacy) and (not all(legacy) or coherence is not None):
        enabled = tuple(
            replace(ch, coherence_group=f"implicit:{ch.name}") if implicit else ch
            for ch, implicit in zip(enabled, legacy)
        )
    beams = beam_stack(original, enabled, coherence)
    if "beam_editor" in presentation:
        from launchplane.serialization import beam_stack_from_dict, beam_stack_to_dict

        editor = presentation["beam_editor"]
        saved = beam_stack_from_dict(editor["beam_stack"])
        beams = replace(beams, coherence=original.coherence)
        converted = iter(beam_stack_to_launchplane(beams).beams)
        # Disabled objects retain their original intent and stack position.
        definitions = []
        for channel, active in items:
            if active:
                definition = next(converted)
            else:
                definition = beam_stack_to_launchplane(
                    replace(original, channels=(channel,))
                ).beams[0]
                definition = replace(definition, enabled=False)
            definitions.append(definition)
        return beams, {"beam_editor": {
            **editor, "beam_stack": beam_stack_to_dict(replace(saved, beams=tuple(definitions)))
        }}
    disabled = [asdict(channel) for channel, active in items if not active]
    return beams, {DISABLED_BEAMS: disabled} if disabled else {}
