"""Round-trip the desktop's optional headless experiment metadata."""

from copy import deepcopy
from dataclasses import replace

from lcprop.adapters.launchplane import beam_stack_to_launchplane
from lcprop.notebook.batch_beams import DISABLED_BEAMS
from lcprop.transport.result_policy import normalize_result_policy


def desktop_metadata(loaded):
    """Admit only metadata the notebook can faithfully write back."""
    presentation = loaded.presentation_payload
    unknown = set(presentation) - {"beam_editor", DISABLED_BEAMS}
    if unknown:
        raise ValueError(f"Unsupported presentation fields: {', '.join(sorted(unknown))}.")
    if "beam_editor" in presentation and presentation["beam_editor"] is None:
        raise ValueError("Unsupported null beam_editor presentation.")
    intent = loaded.execution_intent
    if intent is not None:
        if intent.target != "local":
            raise ValueError("Saved Slurm execution intent is outside the local CPU editor.")
        if (intent.requested_backend, intent.precision) != ("numpy", "float64"):
            raise ValueError("Saved execution intent must use NumPy float64.")
        if normalize_result_policy(intent.retrieval_policy) not in ("fast", "full"):
            raise ValueError("Saved retrieval policy is outside the bounded local TD editor.")
    return deepcopy(presentation), intent


def update_presentation(presentation, beams):
    """Replace enabled slots, retaining disabled editor objects and their order."""
    from launchplane.serialization import beam_stack_from_dict, beam_stack_to_dict

    editor = presentation["beam_editor"]
    saved = beam_stack_from_dict(editor["beam_stack"])
    enabled = beam_stack_to_launchplane(beams).beams
    remaining = iter(enabled)
    updated = []
    for beam in saved.beams:
        if beam.enabled:
            replacement = next(remaining, None)
            if replacement is not None:
                updated.append(replacement)
        else:
            updated.append(beam)
    updated.extend(remaining)
    stack = replace(saved, beams=tuple(updated))
    return {"beam_editor": {**editor, "beam_stack": beam_stack_to_dict(stack)}}
