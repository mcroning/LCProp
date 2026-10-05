"""Owned accepted-psi checkpoints and physical-illumination TD segments.

Only local NumPy nonlinear spectral IMEX continuation is qualified here.
No material-state conversion, regauging, or viewer-state reconstruction.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from lcprop.core.grid import round_nz
from lcprop.pr.illumination import INTEGRAL_NORMALIZATION, reference_from_metadata
from lcprop.pr.specs import PR_TD_PUBLISHED_COUPLING
from .specs import PRTransverseRunRequest, PR_TRANSVERSE_IMEX_EULER

COMPATIBILITY = 'pr_transverse_accepted_psi_physical_segments_v1'
LAYOUT = 'periodic_spectral_psi_zxy_v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def state_identity(a):
    if not isinstance(a, np.ndarray):
        raise TypeError('continuation currently requires a host NumPy accepted state')
    return _digest({'dtype':a.dtype.str, 'shape':list(a.shape),
                    'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()})


def request_metadata(request):
    from .timedependent_transport_codec import encode_pr_transverse_timedependent_transport_request
    clean = replace(request, initial_A=None, initial_psi=None)
    payload = encode_pr_transverse_timedependent_transport_request(clean).payload
    if payload.arrays:
        raise ValueError('continuation request requires portable beam-defined illumination')
    return deepcopy(payload.metadata)


def _eligible(request):
    from .workflow import _validate_request
    if not isinstance(request, PRTransverseRunRequest):
        raise TypeError('full-transverse TD request required')
    _validate_request(request)
    if (request.material.normalization_identity != INTEGRAL_NORMALIZATION
            or request.material_response.model != 'nonlinear'
            or request.solver.integrator != PR_TRANSVERSE_IMEX_EULER
            or request.resolved_optical_coupling != PR_TD_PUBLISHED_COUPLING
            or request.backend.backend != 'numpy'):
        raise ValueError('continuation requires physical-normalized NumPy nonlinear published IMEX TD')


def compatibility_key(request):
    _eligible(request)
    meta = request_metadata(request)
    # Physical illumination may change; state units, optical calibration and
    # canonical scattering realization do not. Duration/cadence are numerical.
    meta.pop('beams'); meta.pop('launch_elements'); meta.pop('initial_A'); meta.pop('initial_psi')
    meta['material'].pop('dark_irradiance_W_cm2')
    meta['material'].pop('uniform_irradiance_W_cm2')
    meta['solver'].pop('Nt'); meta['solver'].pop('dt_normalized')
    meta['backend'].pop('verbose', None)
    meta['wavelengths_um'] = sorted(set(ch.wavelength_um for ch in request.beams.channels))
    return meta


@dataclass(frozen=True)
class TransverseTDCheckpoint:
    """Owns an immutable-byte-backed psi buffer; metadata is hash-checked on use."""
    request: PRTransverseRunRequest
    psi: np.ndarray
    record: dict
    identity: str

    @property
    def completed_steps(self):
        return self.record['cumulative_completed_steps']

    @property
    def time_normalized(self):
        return self.record['cumulative_time']

    @property
    def requested_steps(self):
        return self.request.solver.Nt

    def validate(self):
        _eligible(self.request)
        r = self.record
        if r.get('compatibility') != COMPATIBILITY or r.get('state_layout') != LAYOUT:
            raise ValueError('unknown continuation identity/layout')
        if r['request'] != request_metadata(self.request) or r['compatibility_key'] != compatibility_key(self.request):
            raise ValueError('checkpoint request identity mismatch')
        if r['state_identity'] != state_identity(self.psi) or _digest(r) != self.identity:
            raise ValueError('checkpoint state/provenance identity mismatch')
        shape=(round_nz(self.request.grid.z_length_um,self.request.grid.dz_um),self.request.grid.Nx,self.request.grid.Ny)
        if self.psi.shape != shape or self.psi.dtype != np.dtype(self.request.backend.precision):
            raise ValueError('checkpoint layout/precision mismatch')
        if not np.isfinite(self.psi).all() or r['status'] not in ('completed','cancelled','failed'):
            raise ValueError('invalid accepted checkpoint')
        if not math.isfinite(self.time_normalized) or self.time_normalized < 0 or self.completed_steps < 0:
            raise ValueError('invalid accepted time')
        blocks=r['time_blocks']
        if (not isinstance(blocks,list) or not blocks or any(set(b)!={'dt','steps'}
                or not isinstance(b['steps'],int) or b['steps']<0
                or not math.isfinite(b['dt']) or b['dt']<=0 for b in blocks)
                or sum(b['steps'] for b in blocks)!=self.completed_steps
                or math.fsum(b['steps']*b['dt'] for b in blocks)!=self.time_normalized):
            raise ValueError('checkpoint cumulative time ledger mismatch')
        ref=reference_from_metadata(r['source_normalization'])
        if (r.get('normalization_identity') != INTEGRAL_NORMALIZATION
                or ref.dark_irradiance_W_cm2 != self.request.material.dark_irradiance_W_cm2
                or ref.uniform_irradiance_W_cm2 != self.request.material.uniform_irradiance_W_cm2):
            raise ValueError('checkpoint physical illumination mismatch')
        from .transport import state_from_potential
        k=self.request.material.characteristic_wavenumber_per_um
        state=state_from_potential(self.psi,dx_normalized=k*self.request.grid.x_aperture_um/self.request.grid.Nx,
            dy_normalized=k*self.request.grid.y_aperture_um/self.request.grid.Ny,
            h_y=self.request.dielectric.h_y,applied_field_x=self.request.boundary.applied_field_x,xp=np)
        if not np.isfinite(state.carrier_density).all() or not np.all(state.carrier_density>0):
            raise ValueError('checkpoint is not a physical accepted carrier state')
        return self


def _checkpoint(request, psi, *, steps, time, status, reference, result_identity, lineage, time_blocks=None):
    clean=replace(request, initial_A=None, initial_psi=None)
    a=np.frombuffer(np.ascontiguousarray(psi).tobytes(),dtype=psi.dtype).reshape(psi.shape)
    record={'compatibility':COMPATIBILITY,'state_layout':LAYOUT,
        'normalization_identity':INTEGRAL_NORMALIZATION,'request':request_metadata(clean),
        'compatibility_key':compatibility_key(clean),'state_identity':state_identity(a),
        'cumulative_completed_steps':int(steps),'cumulative_time':float(time),'status':status,
        'source_normalization':deepcopy(reference),'source_result_identity':result_identity,
        'lineage':deepcopy(lineage),
        'time_blocks':deepcopy(time_blocks if time_blocks is not None else [{'dt':request.solver.dt_normalized,'steps':steps}])}
    return TransverseTDCheckpoint(clean,a,record,_digest(record)).validate()


def checkpoint_from_result(request, result):
    """Export only a retained authoritative accepted state, never a Fast preview."""
    _eligible(request)
    if result.psi_final is None or not result.diagnostics.get('physical_state_valid'):
        raise ValueError('result lacks a valid retained accepted material state')
    if result.status not in ('completed','cancelled'):
        raise ValueError('failed sources require an explicitly preserved accepted checkpoint')
    validate_result_lineage(result)
    meta=request_metadata(request)
    if _json(meta) != _json(result.resolved_profile.get('continuation_request')):
        raise ValueError('source result lacks matching complete continuation request provenance')
    for key, profile_key in (('grid','grid_request'),('beams','beam_request'),('material','material'),
                             ('solver','solver'),('transport','transport'),('dielectric','dielectric'),
                             ('projection','projection'),('boundary','boundary')):
        # JSON normalizes tuple/list representation on reopen.
        if _json(meta[key]) != _json(result.resolved_profile[profile_key]):
            raise ValueError('source request/result mismatch: '+key)
    from .timedependent_transport_codec import encode_pr_transverse_timedependent_transport_result
    payload=encode_pr_transverse_timedependent_transport_result(result,result_policy='full').payload
    result_id=_digest({'metadata':payload.metadata,'arrays':{k:state_identity(v) for k,v in payload.arrays.items()}})
    segment=result.resolved_profile.get('continuation_segment')
    return _checkpoint(request,result.psi_final,
        steps=result.completed_steps if segment is None else segment['cumulative_completed_steps'],
        time=result.time_normalized if segment is None else segment['cumulative_time'],
        status=result.status,reference=result.resolved_profile['source_normalization'],
        result_identity=result_id,lineage=[] if segment is None else segment['lineage'],
        time_blocks=None if segment is None else segment['time_blocks'])


def validate_continuation(request, checkpoint):
    checkpoint.validate()
    if request.initial_A is not None or request.initial_psi is not None:
        raise ValueError('continuation state comes exclusively from accepted checkpoint')
    if compatibility_key(request) != checkpoint.record['compatibility_key']:
        raise ValueError('incompatible continuation grid/model/precision/normalization/calibration')
    if sum(ch.power_mW for ch in request.beams.channels)==0 and (
            request.material.dark_irradiance_W_cm2+request.material.uniform_irradiance_W_cm2)<=0:
        raise ValueError('total physical illumination must be positive')


def describe_continuation(request, checkpoint):
    validate_continuation(request,checkpoint)
    old=checkpoint.record['request'];new=request_metadata(request)
    changed={k:{'from':old[k],'to':new[k]} for k in new if old[k]!=new[k]}
    return {'compatibility':COMPATIBILITY,'source_checkpoint':checkpoint.identity,
        'source_result':checkpoint.record['source_result_identity'],
        'source_state':checkpoint.record['state_identity'],'source_status':checkpoint.record['status'],
        'source_cumulative_time':checkpoint.time_normalized,
        'source_cumulative_steps':checkpoint.completed_steps,
        'segment_index':len(checkpoint.record['lineage'])+1,'changed_parameters':changed,
        'normalization_identity':INTEGRAL_NORMALIZATION,
        'source_normalization':deepcopy(checkpoint.record['source_normalization']),
        'new_physical_request':new,'time_units':'characteristic_time'}


def _time_ledger(checkpoint, steps, dt):
    # Coalesce equal-cadence blocks so N1+N2 has the same floating time
    # as (N1+N2)*dt, independent of segmentation. Never rescale old time.
    blocks=deepcopy(checkpoint.record['time_blocks'])
    if blocks[-1]['dt']==dt:blocks[-1]['steps']+=int(steps)
    else:blocks.append({'dt':dt,'steps':int(steps)})
    return blocks,math.fsum(b['steps']*b['dt'] for b in blocks)


class ContinuationFailure(RuntimeError):
    """Original failure plus last accepted owned state; never a candidate."""
    def __init__(self, checkpoint, cause):
        super().__init__(str(cause));self.checkpoint=checkpoint


def continue_transverse_td(request, checkpoint, *, cancellation_token=None, progress_callback=None):
    from .workflow import run_pr_transverse_timedependent
    segment=describe_continuation(request,checkpoint)
    latest=[None,0,None]
    def accepted(psi,steps,reference):
        latest[:]=[psi.copy(),steps,reference]
    def observe(progress):
        if progress_callback is not None:
            progress_callback(replace(progress,
                cumulative_completed_steps=checkpoint.completed_steps+progress.completed_units,
                cumulative_time=_time_ledger(checkpoint,progress.completed_units,request.solver.dt_normalized)[1],
                segment_start_time=checkpoint.time_normalized, prior_completed_steps=checkpoint.completed_steps,
                completed_step=checkpoint.completed_steps+progress.completed_units,
                current_time=_time_ledger(checkpoint,progress.completed_units,request.solver.dt_normalized)[1]))
    try:
        result=run_pr_transverse_timedependent(request,cancellation_token=cancellation_token,
            progress_callback=observe if progress_callback is not None else None,
            _continuation_psi=checkpoint.psi,_accepted_callback=accepted)
    except Exception as exc:
        state,steps,reference=latest
        if state is None:
            # Launch preparation failed: there is no accepted new segment.
            record=deepcopy(checkpoint.record);record['status']='failed'
            record['failed_attempt']=segment
            failed=TransverseTDCheckpoint(checkpoint.request,checkpoint.psi,record,_digest(record)).validate()
        else:
            failed_segment=dict(segment,segment_completed_steps=steps,
                segment_time=steps*request.solver.dt_normalized,status='failed',
                new_normalization=reference)
            blocks,total_time=_time_ledger(checkpoint,steps,request.solver.dt_normalized)
            failed=_checkpoint(request,state,steps=checkpoint.completed_steps+steps,
                time=total_time,status='failed',time_blocks=blocks,
                reference=reference,result_identity=checkpoint.record['source_result_identity'],
                lineage=checkpoint.record['lineage']+[failed_segment])
        raise ContinuationFailure(failed,exc) from exc
    blocks,total_time=_time_ledger(checkpoint,result.completed_steps,request.solver.dt_normalized)
    segment.update(segment_completed_steps=result.completed_steps,segment_time=result.time_normalized,time_blocks=blocks,
        cumulative_completed_steps=checkpoint.completed_steps+result.completed_steps,
        cumulative_time=total_time,
        new_normalization=deepcopy(result.resolved_profile['source_normalization']))
    lineage=deepcopy(checkpoint.record['lineage'])+[deepcopy(segment)]
    segment['lineage']=lineage
    return replace(result,resolved_profile=dict(result.resolved_profile,continuation_segment=segment,
        continuation_compatibility=COMPATIBILITY),
        td_scalar_history=tuple(dict(row,segment_time=row['material_time_normalized'],
            cumulative_time=_time_ledger(checkpoint,index+1,request.solver.dt_normalized)[1])
            for index,row in enumerate(result.td_scalar_history)))


def validate_result_lineage(result):
    """Validate new segment metadata without reinterpreting historical results."""
    profile=result.resolved_profile
    if 'continuation_segment' not in profile and 'continuation_compatibility' not in profile:return
    if profile.get('continuation_compatibility')!=COMPATIBILITY:
        raise ValueError('unknown/missing continuation compatibility identity')
    segment=profile.get('continuation_segment')
    required={'compatibility','source_checkpoint','source_result','source_state','source_status',
        'source_cumulative_time','source_cumulative_steps','segment_index','changed_parameters',
        'normalization_identity','source_normalization','new_physical_request','time_units',
        'segment_completed_steps','segment_time','time_blocks','cumulative_completed_steps',
        'cumulative_time','new_normalization','lineage'}
    if not isinstance(segment,dict) or set(segment)!=required:
        raise ValueError('incomplete continuation segment metadata')
    if (segment['compatibility']!=COMPATIBILITY or segment['normalization_identity']!=INTEGRAL_NORMALIZATION
            or segment['time_units']!='characteristic_time'
            or segment['segment_completed_steps']!=result.completed_steps
            or segment['segment_time']!=result.time_normalized
            or segment['new_normalization']!=profile.get('source_normalization')
            or segment['new_physical_request']!=profile.get('continuation_request')
            or segment['cumulative_completed_steps']!=segment['source_cumulative_steps']+result.completed_steps
            or segment['segment_index']!=len(segment['lineage'])):
        raise ValueError('inconsistent continuation segment provenance')
    reference_from_metadata(segment['source_normalization'])
    reference_from_metadata(segment['new_normalization'])
    blocks=segment['time_blocks']
    if (not blocks or any(set(b)!={'dt','steps'} or not isinstance(b['steps'],int)
            or b['steps']<0 or not math.isfinite(b['dt']) or b['dt']<=0 for b in blocks)
            or sum(b['steps'] for b in blocks)!=segment['cumulative_completed_steps']
            or math.fsum(b['dt']*b['steps'] for b in blocks)!=segment['cumulative_time']):
        raise ValueError('invalid cumulative characteristic-time ledger')
    if not segment['lineage'] or segment['lineage'][-1]!={k:v for k,v in segment.items() if k!='lineage'}:
        raise ValueError('inconsistent segment lineage')


def save_checkpoint(checkpoint, directory):
    checkpoint.validate();path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    np.save(path/'accepted-psi.npy',checkpoint.psi,allow_pickle=False)
    (path/'transverse-continuation.json').write_text(_json({'identity':checkpoint.identity,'record':checkpoint.record})+'\n')
    return path


def load_checkpoint(directory):
    from .timedependent_transport_codec import decode_pr_transverse_timedependent_transport_request
    path=Path(directory);envelope=json.loads((path/'transverse-continuation.json').read_text())
    if set(envelope)!={'identity','record'}:raise ValueError('invalid checkpoint envelope')
    r=envelope['record'];request=decode_pr_transverse_timedependent_transport_request(r['request'],{})
    psi=np.load(path/'accepted-psi.npy',allow_pickle=False)
    a=np.frombuffer(psi.tobytes(),dtype=psi.dtype).reshape(psi.shape)
    return TransverseTDCheckpoint(request,a,r,envelope['identity']).validate()


def notify_failure_checkpoint(failure, progress_callback):
    """Explicit accepted-state failure evidence for the normal worker signal."""
    if progress_callback is None:return
    from lcprop.core.execution import RunProgress
    from .specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    cp=failure.checkpoint
    progress_callback(RunProgress(workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,status='failed',
        completed_units=cp.completed_steps,total_units=cp.completed_steps,
        current_coordinate=cp.time_normalized,coordinate_name='cumulative_material_time',
        coordinate_unit='characteristic_time',elapsed_wall_time=0.,checkpoint_available=True,
        latest_field_state={'accepted_continuation_checkpoint':cp},
        message='Failed segment: last accepted material checkpoint preserved'))


def run_continuable_transverse_td(request, *, cancellation_token=None, progress_callback=None):
    """Normal registered execution with failure-state retention for eligible runs."""
    from .workflow import run_pr_transverse_timedependent
    try:
        _eligible(request)
        request_metadata(request)
    except (ValueError,TypeError):
        return run_pr_transverse_timedependent(request,cancellation_token=cancellation_token,
                                               progress_callback=progress_callback)
    latest=[None,0,None]
    def accepted(psi,steps,reference):latest[:]=[psi.copy(),steps,reference]
    try:
        return run_pr_transverse_timedependent(request,cancellation_token=cancellation_token,
            progress_callback=progress_callback,_accepted_callback=accepted)
    except Exception as exc:
        state,steps,ref=latest
        if state is None:raise
        cp=_checkpoint(request,state,steps=steps,time=steps*request.solver.dt_normalized,
            status='failed',reference=ref,
            result_identity=_digest({'request':request_metadata(request),'accepted_state':state_identity(state),
                                     'steps':steps,'status':'failed'}),lineage=[])
        failure=ContinuationFailure(cp,exc)
        notify_failure_checkpoint(failure,progress_callback)
        raise failure from exc
