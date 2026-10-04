"""Explicit material solver identities. No workflow/planner registration."""
from dataclasses import dataclass
from .specs import (
    MIXED_PRECISION, DOUBLE_PRECISION, POSITIVE_PRECISION,
    UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE,
)

REDUCED = 'pr_unified_reduced_columns_direct_v1'
DIRECT = 'pr_unified_connected_direct_v1'
SCALABLE = 'pr_unified_connected_scalable_v1'
ITERATIVE_POLICY = 's2_pcg_gmres_constant_schur_fft_forcing_v1'
DIRECT_POLICY = 'unified_static_direct_newton_v1'
CONVERGENCE_POLICY = 'unified_static_physical_acceptance_v1'
INITIALIZATION_POLICY = 'unified_static_cold_start_v1'


@dataclass(frozen=True)
class PRUnifiedSolverSpec:
    identity: str
    linear_policy: str
    convergence_policy: str = CONVERGENCE_POLICY
    initialization_policy: str = INITIALIZATION_POLICY

    def __post_init__(self):
        self.validate()

    def validate(self):
        policy = {REDUCED: DIRECT_POLICY, DIRECT: DIRECT_POLICY,
                  SCALABLE: ITERATIVE_POLICY}.get(self.identity)
        if (policy is None or self.linear_policy != policy
                or self.convergence_policy != CONVERGENCE_POLICY
                or self.initialization_policy != INITIALIZATION_POLICY):
            raise ValueError('unknown or contradictory material solver policy')

    def validate_material(self, spatial, closure, precision):
        self.validate()
        spatial.validate()
        closure.validate_spatial(spatial)
        precision.validate()
        if self.identity == REDUCED:
            if spatial.dimension != 1:
                raise ValueError('reduced solver requires independent x columns')
        elif spatial.dimension != 2 or spatial.batch_shape:
            raise ValueError('connected solver requires genuine x-y transport')
        if self.identity == SCALABLE:
            if closure.identity not in (UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE):
                raise ValueError('unsupported scalable closure')
            allowed = (POSITIVE_PRECISION, DOUBLE_PRECISION)
        else:
            allowed = (MIXED_PRECISION, DOUBLE_PRECISION)
        if precision.identity not in allowed:
            raise ValueError('solver/precision combination not supported')
        if self.identity == DIRECT and spatial.active_shape[0]*spatial.active_shape[1] > 12288:
            raise ValueError('connected direct solver exceeds 12,288-node bound')


def legacy_solver(spatial):
    """Fixed historical meaning, independent of planners and runtime hardware."""
    return PRUnifiedSolverSpec(REDUCED if spatial.dimension == 1 else DIRECT, DIRECT_POLICY)


def validate_execution(solver, spatial, closure, precision, backend):
    """Metadata-only qualification; never probe or allocate a device."""
    if not isinstance(solver, PRUnifiedSolverSpec):
        raise TypeError('explicit PRUnifiedSolverSpec required')
    solver.validate_material(spatial, closure, precision)
    if solver.identity != SCALABLE:
        return
    if backend not in ('numpy', 'cupy'):
        raise ValueError('unknown scalable backend')
    bridge = spatial.active_shape == (384, 32) and closure.identity == UNBIASED
    limit = 96 if backend == 'numpy' else (256 if precision.identity == POSITIVE_PRECISION else 512)
    if not bridge and (max(spatial.active_shape) > limit or
                       spatial.active_shape[0]*spatial.active_shape[1] > limit*limit):
        raise ValueError('outside scalable commissioned envelope')
