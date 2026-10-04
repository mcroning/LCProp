"""Headless adapter; core policy/equations remain owned by committed S4-M1."""
from ._scalable import solve_material
from .state import PRMaterialDiagnostics
from .specs import UNBIASED


def solve_with_diagnostics(intensity, *, closure, solver):
    state, trace = solve_material(intensity, closure=closure, solver=solver)
    final = trace[-1]
    names = ('gauss_rms', 'gauss_max', 'flux_divergence_rms', 'flux_divergence_max',
             'closure_rms', 'closure_max', 'neutrality', 'gauge', 'carrier_min',
             'carrier_max', 'finite', 'gauss_scale', 'flux_scale', 'face_flux_max')
    observations = list(zip(names, final['values']))
    observations.extend((('iterations', float(len(trace)-1)), ('converged', 1.)))
    limits = []
    for group, rms, maximum in zip(('gauss', 'flux_divergence', 'closure'),
                                  final['rms_limits'], final['max_limits']):
        limits.extend(((group+'_rms', rms), (group+'_max', maximum)))
    eps = 2.**-23 if intensity.precision.state_dtype == 'float32' else 0.
    limits.extend((('neutrality', max(1e-11, 8*eps)), ('gauge', max(1e-11, 8*eps))))
    if closure.identity == UNBIASED:
        limits.append(('face_flux_max', max(1e-9, 32*eps*final['values'][12])))
    for key in ('relative_potential_correction', 'q_b_correction'):
        if key in final:
            observations.append((key, final[key]))
    observations.extend((('linear_pcg',float(closure.identity==UNBIASED)),
                         ('linear_gmres',float(closure.identity!=UNBIASED))))
    for step in trace:
        tag=f"iteration_{step['iteration']}."
        observations.append((tag+'inner_iterations',float(len(step.get('linear',[])))))
        for key in ('eta','alpha','halvings','final_true_relative','postcast_true_relative'):
            if key in step:observations.append((tag+key,float(step[key])))
        for index,inner in enumerate(step.get('linear',[])):
            for key,value in inner.items():
                if isinstance(value,(int,float)):
                    observations.append((tag+f'linear_{index}.'+key,float(value)))
    observations.append(('inner_iterations_total',float(sum(len(step.get('linear',[])) for step in trace))))
    # The core returns only after its physical AND state32 correction gates.
    return state, PRMaterialDiagnostics(tuple(observations), tuple(limits), 'reported')
