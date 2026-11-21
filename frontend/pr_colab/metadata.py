"""Frontend-only experiment intent and inactive launch metadata.

Native request/intent codecs remain authoritative. Inactive screens are editor
state keyed by unique beam name; they never enter a solver's enabled channels.
"""
from copy import deepcopy
from lcprop.optics.screens import ChannelLaunchElements, validate_channel_launch_elements
from lcprop.pr.portable_launch import encode_launch_elements, decode_launch_elements
from lcprop.persistence.execution_intent import validate_execution_intent
from lcprop.transport.result_policy import normalize_result_policy
from .batch_beams import DISABLED_BEAMS, objects, validate_objects

INACTIVE_LAUNCH = 'notebook_inactive_launch_elements_v1'


def inactive_elements(beams, presentation):
    value = presentation.get(INACTIVE_LAUNCH, {})
    if not isinstance(value, dict):
        raise ValueError('Inactive beam screens must be a name-keyed object')
    disabled = {ch.name for ch, enabled in objects(beams, presentation) if not enabled}
    if not set(value) <= disabled:
        raise ValueError('Inactive screens must belong to existing disabled beams')
    result = {}
    for name, payload in value.items():
        assignments = decode_launch_elements(payload, n_channels=1)
        if len(assignments) != 1 or assignments[0].channel_index != 0:
            raise ValueError('Inactive screen entry must contain one canonical beam assignment')
        result[name] = assignments[0].elements
    return result


def remap_elements(request, items, old_presentation, new_presentation):
    """Keep attachment ownership through enabled-channel reindexing/deletion."""
    validate_channel_launch_elements(request.launch_elements, n_channels=len(request.beams.channels))
    owners = inactive_elements(request.beams, old_presentation)
    for assignment in request.launch_elements:
        owners[request.beams.channels[assignment.channel_index].name] = assignment.elements
    active = []; inactive = {}; index = 0
    for channel, enabled in items:
        elements = owners.get(channel.name)
        if enabled:
            if elements is not None:
                active.append(ChannelLaunchElements(index, elements))
            index += 1
        elif elements is not None:
            inactive[channel.name] = encode_launch_elements((ChannelLaunchElements(0, elements),))
    updated = dict(new_presentation)
    if inactive:
        updated[INACTIVE_LAUNCH] = inactive
    else:
        updated.pop(INACTIVE_LAUNCH, None)
    return tuple(active), updated


def experiment_metadata(loaded, operation):
    """Admit supported local intent without the historical bounded-UI limits."""
    presentation = deepcopy(loaded.presentation_payload)
    unknown = set(presentation) - {'beam_editor', DISABLED_BEAMS, INACTIVE_LAUNCH}
    if unknown:
        raise ValueError('Unsupported presentation fields: '+', '.join(sorted(unknown)))
    if 'beam_editor' in presentation and presentation['beam_editor'] is None:
        raise ValueError('Unsupported null beam_editor presentation')
    if DISABLED_BEAMS in presentation and 'beam_editor' in presentation:
        raise ValueError('Conflicting beam editor metadata')
    validate_objects(objects(loaded.request.beams, presentation))
    inactive_elements(loaded.request.beams, presentation)
    intent = validate_execution_intent(loaded.execution_intent, loaded.request)
    policy = 'full'
    if intent is not None:
        if intent.target != 'local':
            raise ValueError('Saved Slurm execution intent is outside this local notebook editor')
        policy = normalize_result_policy(intent.retrieval_policy)
        if not operation.supports_result_policy and policy != 'full':
            raise ValueError('This engine workflow has fixed native products; saved retention must be full')
    return presentation, intent, policy
