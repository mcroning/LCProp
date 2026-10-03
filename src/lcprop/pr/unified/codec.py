"""Standalone versioned JSON/NPY persistence. No pickle, registry or resume.

Encoding explicitly exports selected backend arrays. Decoding is host-only;
executing a stored request requires explicit materialize(). Execution backend
provenance is distinct from the host representation of a decoded result.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

from lcprop.core.context import GridSpec
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec, canonical_slab_range
from .specs import (PRUnifiedSpatialSpec,PRElectricalClosureSpec,PRMaterialPrecisionSpec,
                    MATERIAL_ID,NORMALIZATION_ID,STATE_ID)
from .state import PRUnifiedMaterialState,PRMaterialDiagnostics
from .projection import PROJECTION_ID
from .workflow import (UnifiedStaticRequest,UnifiedStaticResult,UnifiedProductSelection,
                       WORKFLOW_ID,ARITHMETIC_ID,_validate,_material_gate)
from .products import UnifiedSelection,UnifiedSelectedResult,PRODUCTS_SCHEMA,product_locations,next_address

REQUEST_SCHEMA = 'pr_unified_static_request_v1'
RESULT_SCHEMA = 'pr_unified_static_result_v1'
STATE_SCHEMA = 'pr_unified_static_canonical_state_v1'
GAUGE = 'zero_mean_potential_per_active_domain_v1'


def _host(a):
    if isinstance(a,np.ndarray): return a
    import cupy as cp
    if not isinstance(a,cp.ndarray): raise ValueError('explicit numerical array required')
    return cp.asnumpy(a)  # Deliberate persistence boundary, never a solver fallback.


def _pack(metadata, arrays):
    stream=io.BytesIO();descriptors={}
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        for index,(name,value) in enumerate(sorted(arrays.items())):
            a=_host(value)
            if a.dtype.kind not in 'fc' or not a.dtype.isnative or not np.all(np.isfinite(a)):
                raise ValueError('finite native floating arrays required')
            buffer=io.BytesIO();np.save(buffer,a,allow_pickle=False);data=buffer.getvalue()
            path=f'arrays/{index}.npy'
            descriptors[name]=dict(path=path,shape=list(a.shape),dtype=a.dtype.name,
                bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            archive.writestr(path,data)
        archive.writestr('metadata.json',json.dumps(dict(metadata,arrays=descriptors),
            sort_keys=True,allow_nan=False,separators=(',',':')))
    return stream.getvalue()


def _unpack(payload, schema):
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)): raise ValueError('duplicate archive entries')
            meta=json.loads(archive.read('metadata.json'),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            if meta['schema']!=schema: raise ValueError('unknown persistence schema')
            arrays={};paths=[]
            for name,r in meta.pop('arrays').items():
                paths.append(r['path']);data=archive.read(r['path'])
                if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:
                    raise ValueError('array checksum mismatch')
                a=np.load(io.BytesIO(data),allow_pickle=False)
                if (list(a.shape)!=r['shape'] or a.dtype.name!=r['dtype'] or a.dtype.kind not in 'fc'
                        or not a.dtype.isnative or not np.all(np.isfinite(a))):
                    raise ValueError('array shape/dtype/content mismatch')
                arrays[name]=a
            if len(paths)!=len(set(paths)) or set(names)!=set(paths)|{'metadata.json'}:
                raise ValueError('unexpected archive members')
            return meta,arrays
    except (KeyError,TypeError,AttributeError,zipfile.BadZipFile,OSError) as exc:
        raise ValueError('malformed unified archive') from exc


def _keys(value, expected):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise ValueError('unknown or missing schema fields')


def _construct(cls, value):
    _keys(value,[f.name for f in fields(cls)])
    try: return cls(**value)
    except TypeError as exc: raise ValueError('malformed specification') from exc


def request_metadata(r):
    result={f.name:getattr(r,f.name) for f in fields(r) if f.name!='initial_A'}
    for key in ('grid','spatial','closure','precision','material'):
        result[key]=asdict(result[key])
    result['scattering']=asdict(r.scattering) if r.scattering is not None else None
    return result


def _request(config, launch):
    _keys(config,[f.name for f in fields(UnifiedStaticRequest) if f.name!='initial_A'])
    values=dict(config)
    for key,cls in (('grid',GridSpec),('spatial',PRUnifiedSpatialSpec),('closure',PRElectricalClosureSpec),
                    ('precision',PRMaterialPrecisionSpec),('material',PRMaterialSpec)):
        values[key]=_construct(cls,values[key])
    if values['scattering'] is not None:
        values['scattering']=_construct(PRCanonicalScatteringSpec,values['scattering'])
    if values['coherence_groups'] is not None: values['coherence_groups']=tuple(values['coherence_groups'])
    r=UnifiedStaticRequest(initial_A=launch,**values)
    _validate(r,UnifiedProductSelection())
    return r


@dataclass(frozen=True)
class StoredUnifiedRequest:
    config: dict
    launch: np.ndarray
    selection: UnifiedSelection

    def materialize(self, *, backend=None):
        """Explicit new-run allocation; no state conversion or continuation."""
        config=dict(self.config)
        if backend is not None: config['backend']=backend
        if config['backend']=='numpy': launch=self.launch.copy()
        elif config['backend']=='cupy':
            import cupy as cp
            launch=cp.asarray(self.launch)
            cp.cuda.get_current_stream().synchronize()  # upload owns completed snapshot
        else: raise ValueError('explicit backend required')
        return _request(config,launch)


def encode_request(request, *, selection=UnifiedSelection()):
    config=request_metadata(request)
    _validate(request,UnifiedProductSelection());selection.validate(request.spatial)
    payload=_pack(dict(schema=REQUEST_SCHEMA,material_identity=MATERIAL_ID,
        normalization=NORMALIZATION_ID,config=config,selection=asdict(selection)),{'launch':request.initial_A})
    decode_request(payload)  # Same strict acceptance for encode/decode.
    return payload


def _decode_request(payload):
    meta,arrays=_unpack(payload,REQUEST_SCHEMA)
    _keys(meta,('schema','material_identity','normalization','config','selection'))
    if meta['material_identity']!=MATERIAL_ID or meta['normalization']!=NORMALIZATION_ID or set(arrays)!={'launch'}:
        raise ValueError('incompatible material/normalization/launch record')
    r=_request(meta['config'],arrays['launch'])
    selection=_construct(UnifiedSelection,meta['selection']);selection.validate(r.spatial)
    a=arrays['launch'];dtype='complex64' if r.precision.state_dtype=='float32' else 'complex128'
    if a.ndim!=3 or a.shape[0]<1 or a.shape[1:]!=r.spatial.field_shape or a.dtype.name!=dtype:
        raise ValueError('launch geometry/precision mismatch')
    from lcprop.core.beams import normalize_coherence_groups
    normalize_coherence_groups(a.shape[0],coherent=False,coherence_groups=r.coherence_groups)
    return StoredUnifiedRequest(request_metadata(r),a,selection)


def _identity_request(identity):
    # M4 result provenance, not an old request with extension flags.
    required={'workflow','arithmetic','projection','material','spatial','closure','precision','backend',
              'electro_optic','optical_boundary','wavelength_um','material_parameters','grid','scattering'}
    if not required<=set(identity) or set(identity)-required-{'coherence_groups','peak_intensity_reference'}:
        raise ValueError('incomplete or unknown execution provenance')
    if (identity['material']!=MATERIAL_ID or identity['optical_boundary']!='periodic'
            or identity['electro_optic']!='existing_scalar_x_delta_n'):
        raise ValueError('incompatible physical identity')
    config=dict(grid=identity['grid'],spatial=identity['spatial'],closure=identity['closure'],
        material=identity['material_parameters'],precision=identity['precision'],backend=identity['backend'],
        wavelength_um=identity['wavelength_um'],coherence_groups=identity.get('coherence_groups'),
        scattering=identity['scattering'],workflow_identity=identity['workflow'],
        arithmetic_identity=identity['arithmetic'],projection_identity=identity['projection'])
    return _request(config,None)


def _state_meta(state):
    return dict(schema=STATE_SCHEMA,state_identity=STATE_ID,material_identity=MATERIAL_ID,gauge=GAUGE,
        spatial=asdict(state.spatial),closure=asdict(state.closure),precision=asdict(state.precision))


def _launch_provenance(identity, endpoint, launch=None):
    """Validate accepted-launch provenance without inferring missing metadata.

    Exact reference reconstruction is only authoritative on the execution
    backend. Host decoding of GPU results must not invent a cross-backend
    reduction tolerance. An evolved endpoint is never a launch reference.
    """
    if endpoint is None:
        return
    from lcprop.core.beams import normalize_coherence_groups
    from lcprop.pr.source import channel_peak_intensity_reference
    groups=identity.get('coherence_groups')
    if not isinstance(groups,(tuple,list)):
        raise ValueError('accepted launch requires explicit coherence groups')
    normalize_coherence_groups(endpoint.shape[0],coherent=False,coherence_groups=groups)
    peak=identity.get('peak_intensity_reference')
    if type(peak) not in (int,float) or not math.isfinite(peak) or peak<=0:
        raise ValueError('accepted launch requires a finite positive numeric peak reference')
    if launch is not None:
        import sys
        backend=identity['backend']
        xp=sys.modules.get(backend)
        if xp is not None and isinstance(launch,xp.ndarray):
            if launch.shape!=endpoint.shape or launch.dtype!=endpoint.dtype:
                raise ValueError('retained launch layout mismatch')
            if channel_peak_intensity_reference(launch,xp=xp)!=peak:
                raise ValueError('accepted-launch peak reference mismatch')


def encode_result(result):
    s=result.scientific
    if result.schema!=PRODUCTS_SCHEMA: raise ValueError('unknown product schema')
    launch=result.arrays.get('launch')
    if launch is None and s.completed_cells==0: launch=s.boundary_field
    _launch_provenance(s.identities,s.boundary_field,launch)
    arrays={'product/'+k:v for k,v in result.arrays.items()}
    arrays.update({'coordinate/'+k:v for k,v in result.coordinates.items()})
    if s.boundary_field is not None: arrays['endpoint']=s.boundary_field
    state=None
    if s.material_state is not None:
        state=_state_meta(s.material_state)
        arrays.update({'state/'+k:getattr(s.material_state,k) for k in ('q','psi','b')})
    science={name:getattr(s,name) for name in ('status','reason','completed_cells','reached_z_um',
        'boundary_z_um','material_z_um','ledger','identities','failure')}
    meta=dict(schema=RESULT_SCHEMA,products_schema=PRODUCTS_SCHEMA,scientific=science,
        selection=asdict(result.selection),locations=result.locations,state=state,next_address=result.next_address,
        normalization=NORMALIZATION_ID)
    payload=_pack(meta,arrays)
    decode_result(payload)
    return payload


def _decode_result(payload):
    meta,arrays=_unpack(payload,RESULT_SCHEMA)
    _keys(meta,('schema','products_schema','scientific','selection','locations','state','next_address','normalization'))
    if meta['products_schema']!=PRODUCTS_SCHEMA: raise ValueError('unknown products identity')
    if meta['normalization']!=NORMALIZATION_ID: raise ValueError('unknown source normalization')
    s=meta['scientific']
    _keys(s,('status','reason','completed_cells','reached_z_um','boundary_z_um','material_z_um','ledger','identities','failure'))
    r=_identity_request(s['identities'])
    selection=_construct(UnifiedSelection,meta['selection']);selection.validate(r.spatial)
    count=s['completed_cells'];boundary=tuple(s['boundary_z_um']);material=tuple(s['material_z_um'])
    if type(count) is not int or count<0 or len(s['ledger'])!=count or s['status'] not in ('completed','cancelled','failed'):
        raise ValueError('invalid completion ledger')
    total=_validate(r,UnifiedProductSelection())
    if count>total or len(material)!=count or len(boundary) not in ((0,1) if count==0 else (count+1,)):
        raise ValueError('accepted coordinate count mismatch')
    if ((boundary and (boundary[0]!=0 or boundary[-1]!=s['reached_z_um']))
            or (not boundary and s['reached_z_um']!=0) or material!=boundary[1:]):
        raise ValueError('accepted coordinate/location mismatch')
    if (s['status']=='completed' and (count!=total or s['reached_z_um']!=r.grid.z_length_um)):
        raise ValueError('false completion claim')
    if (s['status']=='failed')!=(s['failure'] is not None): raise ValueError('failure/status mismatch')
    if not isinstance(s['reason'],str): raise ValueError('result reason must be text')
    if s['failure'] is not None and (not isinstance(s['failure'],dict) or not {'stage','cell_index','type','reason'}<=set(s['failure'])):
        raise ValueError('failure evidence missing')
    for k,record in enumerate(s['ledger']):
        _keys(record,('cell_index','z_start_um','z_end_um','material_z_um','optical_distance_um',
                     'material_weight_um','canonical_slabs','observations','limits'))
        if record['cell_index']!=k or record['z_end_um']!=material[k] or record['material_z_um']!=material[k]:
            raise ValueError('ledger coordinate mismatch')
        import math
        start=record['z_start_um'];end=record['z_end_um'];width=record['optical_distance_um']
        if (not math.isfinite(width) or width<=0 or width!=record['material_weight_um']
                or abs(start-boundary[k])>2*max(math.ulp(start),math.ulp(boundary[k]))
                or abs(start+width-end)>2*max(math.ulp(start+width),math.ulp(end))):
            raise ValueError('invalid physical cell bounds')
        expected=None
        if r.scattering is not None:
            slabs=canonical_slab_range(r.scattering,z_start_um=start,dz_um=width,z_length_um=r.grid.z_length_um)
            expected=(slabs.start,slabs.stop)
        actual=tuple(record['canonical_slabs']) if record['canonical_slabs'] is not None else None
        if actual!=expected: raise ValueError('scattering address mismatch')
        record['canonical_slabs']=actual
    if meta['next_address']!=next_address(r,count,s['ledger']): raise ValueError('next accepted address mismatch')
    state=None;state_keys={'state/q','state/psi','state/b'}
    if count:
        sm=meta['state']
        if sm is None: raise ValueError('accepted material state missing')
        _keys(sm,('schema','state_identity','material_identity','gauge','spatial','closure','precision'))
        if (sm['schema']!=STATE_SCHEMA or sm['state_identity']!=STATE_ID or sm['material_identity']!=MATERIAL_ID
                or sm['gauge']!=GAUGE): raise ValueError('unknown canonical-state convention')
        if any(sm[k]!=s['identities'][k] for k in ('spatial','closure','precision')):
            raise ValueError('state/execution metadata mismatch')
        if not state_keys<=set(arrays): raise ValueError('canonical q/psi/b required')
        state=PRUnifiedMaterialState(*(arrays.pop('state/'+k) for k in ('q','psi','b')),
            r.spatial,r.closure,r.precision,'numpy')
        state.validate_structure()
        for record in s['ledger']:
            _material_gate(state,PRMaterialDiagnostics(tuple(record['observations'].items()),
                           tuple(record['limits'].items()),'reported'))
    elif meta['state'] is not None or state_keys&set(arrays): raise ValueError('state without accepted cell')
    endpoint=arrays.pop('endpoint',None)
    dtype=np.dtype(r.precision.state_dtype);complex_dtype=np.dtype('complex64' if dtype==np.float32 else 'complex128')
    if bool(boundary)!=(endpoint is not None): raise ValueError('accepted endpoint missing/extra')
    if endpoint is not None and (endpoint.ndim!=3 or endpoint.shape[0]<1 or endpoint.shape[1:]!=r.spatial.field_shape or endpoint.dtype!=complex_dtype):
        raise ValueError('endpoint layout mismatch')
    coordinates={k[11:]:v for k,v in arrays.items() if k.startswith('coordinate/')}
    products={k[8:]:v for k,v in arrays.items() if k.startswith('product/')}
    if len(coordinates)+len(products)!=len(arrays): raise ValueError('unknown array role')
    for name,values in (('boundary_z_um',boundary),('material_z_um',material)):
        a=coordinates.get(name)
        if a is None or a.dtype!=np.float64 or a.ndim!=1 or not np.array_equal(a,np.asarray(values)) or np.any(np.diff(a)<=0):
            raise ValueError('invalid explicit longitudinal coordinates')
    locations=product_locations(selection,r.spatial,r.grid.x_aperture_um/r.grid.Nx,r.grid.y_aperture_um/r.grid.Ny)
    if meta['locations']!=locations: raise ValueError('product location/projection metadata mismatch')
    partial_products=(s['failure'] is not None and (s['failure']['stage']=='selected_products'
                      or 'selected_product_failure' in s['failure']))
    _validate_products(products,coordinates,selection,r,endpoint,count,s['status'],partial_products)
    launch=products.get('launch')
    if launch is None and count==0: launch=endpoint
    _launch_provenance(s['identities'],endpoint,launch)
    if state is not None:
        for name,attr in (('q_log_carrier','q'),('potential_node','psi'),('harmonic_field','b')):
            if name in products and not np.array_equal(products[name],getattr(state,attr)):
                raise ValueError('canonical product/state disagreement')
    # Restore structural tuple metadata while retaining recorded execution backend.
    identity=dict(s['identities'],spatial=asdict(r.spatial),closure=asdict(r.closure),precision=asdict(r.precision))
    if 'coherence_groups' in identity: identity['coherence_groups']=tuple(identity['coherence_groups'])
    scientific=UnifiedStaticResult(s['status'],s['reason'],count,s['reached_z_um'],endpoint,state,
        boundary,material,tuple(s['ledger']),{},identity,s['failure'])
    return UnifiedSelectedResult(scientific,selection,products,coordinates,locations,next_address=meta['next_address'])


def decode_request(payload):
    try: return _decode_request(payload)
    except (KeyError,TypeError,AttributeError,IndexError,OverflowError) as exc:
        raise ValueError('malformed unified request') from exc


def decode_result(payload):
    try: return _decode_result(payload)
    except (KeyError,TypeError,AttributeError,IndexError,OverflowError) as exc:
        raise ValueError('malformed unified result') from exc


def _validate_products(products,coords,selection,r,endpoint,count,status,partial_products=False):
    expected={};coordinate_keys={'boundary_z_um','material_z_um'}
    real=np.dtype(r.precision.state_dtype);complex_type=np.dtype('complex64' if real==np.float32 else 'complex128')
    nx,ny=r.spatial.field_shape
    if endpoint is not None:
        coordinate_keys|={'x_um','y_um'}
        for name,n,aperture in (('x_um',nx,r.grid.x_aperture_um),('y_um',ny,r.grid.y_aperture_um)):
            a=coords.get(name)
            if a is None or a.shape!=(n,) or a.dtype!=real or np.any(np.diff(a)<=0):
                raise ValueError('invalid transverse coordinates')
            ideal=(np.arange(n,dtype=real)-n/2)*(aperture/n)+.5*(aperture/n)
            if not np.array_equal(a,ideal.astype(real)): raise ValueError('transverse grid mismatch')
        nch=endpoint.shape[0];nb=count+1
        if selection.launch: expected['launch']=(endpoint.shape,complex_type)
        if selection.boundary_intensity: expected['boundary_intensity']=((nx,ny),real)
        for name,enabled in (('optical',selection.optical_cuts),('intensity',selection.intensity_cuts)):
            if enabled:
                for axis,n in (('x',nx),('y',ny)):
                    expected[name+'_cut_'+axis]=((nb,)+( (nch,) if name=='optical' else ())+ (n,),complex_type if name=='optical' else real)
        if count:
            for name in selection.material_fields:
                expected[name]=(r.spatial.harmonic_shape if name=='harmonic_field' else (nx,ny),
                                complex_type if name=='material_phase_optical_node' else real)
            for name in selection.material_cuts:
                for axis,n in (('x',nx),('y',ny)):
                    expected[name+'_cut_'+axis]=((count,n),complex_type if name=='material_phase_optical_node' else real)
        if selection.far_field and ('far_field_intensity' in products or status!='failed'):
            expected['far_field_intensity']=((nx,ny),real);coordinate_keys|={'s_x','s_y'}
            for name,n in (('s_x',nx),('s_y',ny)):
                if name not in coords or coords[name].shape!=(n,) or coords[name].dtype!=np.float64:
                    raise ValueError('far-field axes mismatch')
    if (not set(products)<=set(expected) or (not partial_products and set(products)!=set(expected))
            or set(coords)!=coordinate_keys): raise ValueError('product selection mismatch')
    for name,(shape,dtype) in expected.items():
        if name not in products: continue
        if products[name].shape!=shape or products[name].dtype!=dtype: raise ValueError('product shape/precision mismatch')


def save_request(path,request,*,selection=UnifiedSelection()):
    Path(path).write_bytes(encode_request(request,selection=selection))


def load_request(path): return decode_request(Path(path).read_bytes())


def save_result(path,result): Path(path).write_bytes(encode_result(result))


def load_result(path): return decode_result(Path(path).read_bytes())
