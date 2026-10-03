"""Versioned transverse face-to-optical-node reconstruction; no registration.

Borrowed state buffers are never mutated. Each returned component is detached,
in state precision/on the input device. Only explicitly requested axes allocate.
Structural validation here is not a material-equilibrium acceptance certificate.
"""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

from .operators import Geometry


PROJECTION_ID = "adjacent_face_arithmetic_to_optical_node_v1"


def electric_field_face(state, *, component="x"):
    """Native oriented E at index i means the positive face i+1/2.

    Use the exact M3 field arithmetic on one requested active axis. A reduced
    y batch is never a derivative axis; its b_x broadcasts independently.
    """
    state.validate_structure()
    if component not in state.spatial.active_axes:
        raise ValueError("requested field component is not an active transport axis")
    xp = sys.modules[state.backend]
    context = state.psi.device if state.backend == "cupy" else nullcontext()
    with context:
        g = Geometry(state.spatial.active_shape, state.spatial.normalized_lengths,
                     SimpleNamespace(xp=xp))
        axis = state.spatial.active_axes.index(component)
        b = state.b[..., axis] if state.spatial.batch_shape else state.b[axis]
        return b - (g.neighbor(state.psi, axis) - state.psi) / g.spacing[axis]


def electric_field_optical_node(state, *, component="x", identity=PROJECTION_ID):
    """Adjacent native faces -> optical node, without an extra filter.

    Do not replace by the algebraically equivalent centered potential stencil:
    this identity specifies the face subtraction, addition, then multiply order.
    A single face buffer, periodic rolled buffer and output suffice per axis.
    """
    if identity != PROJECTION_ID:
        raise ValueError("unknown face-to-optical projection identity")
    face = electric_field_face(state, component=component)
    xp = sys.modules[state.backend]
    context = face.device if state.backend == "cupy" else nullcontext()
    with context:
        axis = state.spatial.active_axes.index(component)
        return (face + xp.roll(face, 1, axis=axis)) * face.dtype.type(.5)
