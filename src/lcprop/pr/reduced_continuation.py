"""Physical illumination interventions on accepted reduced E; no material solve."""
from dataclasses import replace
import hashlib
import json
import numpy as np
from lcprop.pr.illumination import INTEGRAL_NORMALIZATION, resolve_material_illumination

IDENTITY = 'pr_reduced_accepted_E_physical_illumination_segments_v1'


def physical_launch(request, grid, complex_dtype):
    from lcprop.optics.launch import build_launch, LaunchResult
    from lcprop.optics.launch import OpticalLaunchContext
    xp = grid.xp
    if sum(ch.power_mW for ch in request.beams.channels) == 0:
        n = len(request.beams.channels)
        return LaunchResult(A0=xp.zeros((n,grid.Nx,grid.Ny),dtype=complex_dtype),
            physical_powers_mW=xp.zeros(n), power_fractions=xp.zeros(n),
            physical_total_power_mW=0.,
            wavelengths_um=xp.asarray([c.wavelength_um for c in request.beams.channels]),
            coherence=request.beams.coherence, coherence_groups=request.beams.coherence_groups,
            power_metadata={'convention':'exact_zero_physical_optical_launch_v1'})
    return build_launch(request.beams, grid, complex_dtype=complex_dtype,
        launch_elements=request.launch_elements, context=OpticalLaunchContext(
            grid=grid,n_ref=request.material.refractive_index,
            interaction_length_um=request.grid.z_length_um))


def describe_continuation(request, checkpoint):
    from lcprop.pr.checkpoint import validate_pr_checkpoint, validate_pr_continuation
    from lcprop.core.backend import get_backend
    from lcprop.core.grid import make_grid
    from lcprop.pr.persistence import _request_to_dict
    validate_pr_checkpoint(checkpoint)
    old = checkpoint.request
    if any(r.material.normalization_identity != INTEGRAL_NORMALIZATION for r in (old,request)):
        raise ValueError('illumination continuation requires explicit physical normalization')
    if request.backend.backend != 'numpy':
        raise ValueError('illumination continuation is qualified locally on NumPy only')
    if request.initial_A is not None:
        raise ValueError('physical TD prepared initial_A requires an explicit physical scale')
    request.material.validate();request.beams.validate();request.solver.validate()
    # Only illumination metadata may differ. Keep wavelength/channel identity,
    # material calibration/closure, dt and all spatial/algorithmic fields fixed.
    fields=('power_mW','x0_um','y0_um','theta_ext_rad','phi_rad','phase_rad')
    if len(request.beams.channels)!=len(old.beams.channels):
        raise ValueError('channel layout is locked; disable channels with zero power')
    canonical_beams=replace(request.beams,channels=tuple(replace(c,**{
        name:getattr(o,name) for name in fields}) for c,o in zip(request.beams.channels,old.beams.channels)))
    canonical_material=replace(request.material,
        dark_irradiance_W_cm2=old.material.dark_irradiance_W_cm2,
        uniform_irradiance_W_cm2=old.material.uniform_irradiance_W_cm2)
    validate_pr_continuation(replace(request,beams=canonical_beams,material=canonical_material),checkpoint)
    backend=get_backend(request.backend)
    grid=make_grid(request.grid,xp=backend.xp,real_dtype=backend.real_dtype)
    def reference(r):
        launch=physical_launch(r,grid,backend.complex_dtype)
        ref,_=resolve_material_illumination(r.material,launch.A0,grid=grid,
            optical_scale_W_cm2=launch.physical_total_power_mW*1e5,
            coherence_groups=r.beams.coherence_groups,xp=grid.xp)
        return ref.metadata()
    a=_request_to_dict(old);b=_request_to_dict(request)
    return {'identity':IDENTITY,'source_segment':len(checkpoint.segment_lineage),
        'source_cumulative_time':checkpoint.time_normalized,
        'source_state_sha256':hashlib.sha256(np.asarray(checkpoint.E_current).tobytes()).hexdigest(),
        'normalization_identity':INTEGRAL_NORMALIZATION,
        'changed_parameters':{k:{'old':a[k],'new':b[k]} for k in a if a[k]!=b[k]},
        'old_reference':reference(old),'new_reference':reference(request)}


def continue_physical_pr(request, checkpoint, additional_steps=None, **kwargs):
    from lcprop.pr.workflow import continue_pr_timedependent
    descriptor=describe_continuation(request,checkpoint)
    # Rebind only after strict compatibility check; raw E_current is unchanged.
    inherited=replace(checkpoint,request=replace(request,initial_A=None,initial_E=None))
    result=continue_pr_timedependent(request,inherited,
        request.solver.Nt if additional_steps is None else additional_steps,**kwargs)
    descriptor.update(completed_steps=result.completed_steps,
        cumulative_time=result.time_normalized,status=result.status)
    lineage=checkpoint.segment_lineage+(descriptor,)
    json.dumps(lineage,allow_nan=False)
    cp=replace(result.checkpoint,segment_lineage=lineage)
    from lcprop.pr.checkpoint import validate_pr_checkpoint
    validate_pr_checkpoint(cp)
    return replace(result,checkpoint=cp,diagnostics=dict(result.diagnostics,
        continuation_identity=IDENTITY,segment_lineage=lineage))
