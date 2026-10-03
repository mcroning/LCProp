"""Bounded 1D unified material solve; deliberately unregistered with workflows.

Returns (owned canonical state, scalar diagnostics). No optical or z semantics.
Every independent column has its own Newton/Armijo path. On failure no partially
accepted plane is returned, and caller buffers are never changed.
"""
from contextlib import nullcontext

from ._backend import MaterialBackend
from ._newton import solve_column
from .operators import Geometry, flux
from .specs import A7_CURRENT, FIXED_FIELD, PRESCRIBED_CURRENT, UNBIASED, PRElectricalClosureSpec
from .state import PRMaterialDiagnostics, PRTransportIntensity, PRUnifiedMaterialState


class MaterialConvergenceError(RuntimeError):
    """One independent column failed; there is no accepted plane result."""

    def __init__(self, column, reason):
        self.column = column
        super().__init__(f'Unified material column {column}: {reason}')


def solve_static_material(intensity: PRTransportIntensity, *, closure):
    """Solve one dark-inclusive transport plane under an explicit 1D closure.

    Borrowed intensity must stay unchanged for the duration of this synchronous
    call. Returned q/psi/b are detached, owned arrays, borrowed by the M1 record.
    Physical gates are the frozen research gates, not M1 structural validation.
    Diagnostics use column-qualified scalar names; provenance includes the exact
    precision identity in the state and explicit backend/policy indicator values.
    """
    if not isinstance(intensity, PRTransportIntensity):
        raise TypeError('PRTransportIntensity required')
    intensity.validate_structure()
    spatial, precision = intensity.spatial, intensity.precision
    if not isinstance(closure, PRElectricalClosureSpec):
        raise TypeError('PRElectricalClosureSpec required')
    closure.validate_spatial(spatial)
    if spatial.dimension != 1 or closure.identity not in (
            UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, A7_CURRENT):
        raise ValueError('M2 requires a supported one-dimensional Static closure')
    if closure.identity == A7_CURRENT and closure.background_intensity != intensity.total_background:
        raise ValueError('A7 reservoir background must match transport background provenance')
    # Allocate on the already-validated input device, not an unrelated current GPU.
    context = intensity.values.device if intensity.backend == 'cupy' else nullcontext()
    with context:
        backend = MaterialBackend(intensity.backend, precision.state_dtype)
        xp = backend.xp
        fixed = closure.identity in (UNBIASED, FIXED_FIELD)
        electrical = (backend.array([[1. if fixed else 0.]]),
                      backend.array([[0. if fixed else 1.]]), backend.array(closure.target))
        q = xp.empty(spatial.field_shape, dtype=backend.dtype)
        psi = xp.empty_like(q)
        b = xp.empty(spatial.harmonic_shape, dtype=backend.dtype)
        count = spatial.batch_shape[0] if spatial.batch_shape else 1
        observations = [('backend_numpy', float(intensity.backend == 'numpy')),
                        ('backend_cupy', float(intensity.backend == 'cupy')),
                        ('state_bits', 32. if precision.state_dtype == 'float32' else 64.),
                        ('linear_bits', 64.), ('bernoulli_bits', 64.), ('columns', float(count))]
        limits = []
        geometry = Geometry(spatial.active_shape, spatial.normalized_lengths, backend)
        names = ('gauss_rms', 'gauss_max', 'flux_divergence_rms', 'flux_divergence_max',
                 'closure_rms', 'closure_max', 'neutrality', 'gauge', 'carrier_min',
                 'carrier_max', 'finite', 'gauss_scale', 'flux_scale', 'face_flux_max')
        for column in range(count):
            src = intensity.values[:, column] if spatial.batch_shape else intensity.values
            try:
                solution = solve_column(geometry, src, electrical, zero=closure.identity == UNBIASED)
            except (RuntimeError, ValueError) as exc:
                raise MaterialConvergenceError(column, str(exc)) from exc
            trace = solution['trace']
            prefix = f'column_{column}.'
            observations.extend((prefix + name, float(value)) for name, value in zip(names, trace[-1]['values']))
            observations.extend(((prefix + 'iterations', float(len(trace)-1)),
                                 (prefix + 'converged', 1.)))
            current = flux(geometry, src, solution['q'], solution['psi'], solution['b'])[0]
            harmonic, mean_current = backend.status([solution['b'][0], xp.mean(current)])
            observations.extend(((prefix + 'harmonic_field', float(harmonic)),
                                 (prefix + 'mean_current', float(mean_current))))
            for group, rms, maximum in zip(('gauss', 'flux_divergence', 'closure'),
                                           trace[-1]['rms_limits'], trace[-1]['max_limits']):
                limits.extend(((prefix + group + '_rms', rms), (prefix + group + '_max', maximum)))
            eps = 2.**-23 if precision.state_dtype == 'float32' else 0.
            limits.extend(((prefix + 'neutrality', max(1e-11, 8*eps)),
                           (prefix + 'gauge', max(1e-11, 8*eps))))
            if closure.identity == UNBIASED:
                limits.append((prefix + 'face_flux_max', max(1e-9, 32*eps*trace[-1]['values'][12])))
            for step in trace:
                tag = prefix + f"iteration_{step['iteration']}."
                for name, value in zip(names, step['values']):
                    observations.append((tag + name, float(value)))
                for key in ('alpha', 'halvings', 'relative_potential_correction', 'q_b_correction'):
                    if key in step:
                        observations.append((tag + key, float(step[key])))
            if spatial.batch_shape:
                q[:, column], psi[:, column], b[column, :] = solution['q'], solution['psi'], solution['b']
            else:
                q[:], psi[:], b[:] = solution['q'], solution['psi'], solution['b']
        state = PRUnifiedMaterialState(q, psi, b, spatial, closure, precision, intensity.backend)
        state.validate_structure()
        diagnostics = PRMaterialDiagnostics(tuple(observations), tuple(limits), 'reported')
        return state, diagnostics
