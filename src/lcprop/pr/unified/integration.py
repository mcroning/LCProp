"""Fresh launch adapter and normal runner registration for the certified M4/M5 core.

No scientific implementation lives here. Prepared requests/results use M5 codecs;
fresh request definitions preserve launch provenance without allocating a plane.
"""
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
import numpy as np

from lcprop.core.backend import BackendSpec, get_backend, asnumpy
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.products.data_model import RunData, Geometry, FieldCollection, make_field
from lcprop.runners.base import WorkflowOperation
from lcprop.transport.codecs import TransportCodec, EncodedRequest, EncodedResult, PortablePayload
from lcprop.transport.result_policy import normalize_result_policy, static_product_selection
from . import codec, products
from .workflow import UnifiedStaticRequest, UnifiedProductSelection, WORKFLOW_ID, ARITHMETIC_ID, _validate
from .specs import PRUnifiedSpatialSpec, PRElectricalClosureSpec, PRMaterialPrecisionSpec, MIXED_PRECISION, DOUBLE_PRECISION, POSITIVE_PRECISION
from .projection import PROJECTION_ID
from .resources import estimate_resources

VOLUME_PRODUCTS = {
    'unified_potential_volume':'potential_node', 'unified_carrier_volume':'carrier_node',
    'unified_optical_field_volume':'electric_field_x_optical_node',
    'unified_face_x_volume':'electric_field_x_face', 'unified_face_y_volume':'electric_field_y_face',
}
ANALYSIS_PRODUCTS = ('far_field_intensity', 'complex_output', 'unified_intensity_volume', *VOLUME_PRODUCTS)
DEFAULT_MATERIAL_VOLUMES = ('potential_node','carrier_node','electric_field_x_optical_node')
LEGACY_FRESH_SCHEMA = 'pr_unified_static_fresh_launch_v1'
FRESH_SCHEMA = 'pr_unified_static_fresh_launch_v2'
from .solver_specs import PRUnifiedSolverSpec, REDUCED, DIRECT, SCALABLE, DIRECT_POLICY, ITERATIVE_POLICY


@dataclass(frozen=True)
class UnifiedFreshRequest:
    grid: GridSpec
    beams: object
    material: PRMaterialSpec
    closure: PRElectricalClosureSpec
    backend: BackendSpec = BackendSpec("numpy", "float64", False)
    launch_elements: tuple = ()
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()
    scattering: PRCanonicalScatteringSpec | None = None
    workflow_identity: str = WORKFLOW_ID
    arithmetic_identity: str = ARITHMETIC_ID
    projection_identity: str = PROJECTION_ID
    solver: PRUnifiedSolverSpec | None = None
    schema: str = FRESH_SCHEMA

    def __post_init__(self):
        if self.solver is None:
            object.__setattr__(self,'solver',PRUnifiedSolverSpec(
                REDUCED if self.closure.dimension==1 else DIRECT,DIRECT_POLICY))


def core_request(request, initial_A=None, *, backend=None):
    """Translate metadata without an allocation; explicit launch preparation below."""
    d=request.closure.dimension;g=request.grid;k=request.material.characteristic_wavenumber_per_um
    spatial=PRUnifiedSpatialSpec((g.Nx,) if d==1 else (g.Nx,g.Ny),
        (k*g.x_aperture_um,) if d==1 else (k*g.x_aperture_um,k*g.y_aperture_um),
        active_axes=('x',) if d==1 else ('x','y'),batch_shape=(g.Ny,) if d==1 else (),
        batch_axes=('y',) if d==1 else ())
    dtype=request.backend.precision
    precision=PRMaterialPrecisionSpec(identity=(POSITIVE_PRECISION if request.solver.identity==SCALABLE else MIXED_PRECISION) if dtype=='float32' else DOUBLE_PRECISION,
        state_dtype=dtype,output_dtype=dtype)
    return UnifiedStaticRequest(grid=g,spatial=spatial,closure=request.closure,initial_A=initial_A,
        material=request.material,precision=precision,backend=backend or request.backend.backend,
        wavelength_um=request.beams.channels[0].wavelength_um,coherence_groups=request.beams.coherence_groups,
        scattering=request.scattering,workflow_identity=request.workflow_identity,
        arithmetic_identity=request.arithmetic_identity,projection_identity=request.projection_identity,
        solver=request.solver,persistence_schema=codec.LEGACY_REQUEST_SCHEMA if request.schema==LEGACY_FRESH_SCHEMA else codec.REQUEST_SCHEMA)


def validate_fresh(request):
    for value in (request.grid,request.beams,request.material,request.backend,request.optical_boundary):value.validate()
    if request.backend.backend not in ('numpy','cupy'):
        raise ValueError('Unified Static requires an explicit NumPy or CuPy backend')
    if request.optical_boundary.mode!='periodic':raise ValueError('Unified Static supports periodic boundaries only')
    if len({c.wavelength_um for c in request.beams.channels})!=1:raise ValueError('Unified Static requires one wavelength')
    LaunchConfiguration(request.beams,request.launch_elements)
    if request.schema not in (FRESH_SCHEMA,LEGACY_FRESH_SCHEMA): raise ValueError('unknown fresh schema')
    if request.schema==LEGACY_FRESH_SCHEMA and request.solver.identity==SCALABLE: raise ValueError('legacy request cannot represent scalable solver')
    _validate(core_request(request),UnifiedProductSelection())


def prepare_request(request):
    """Use the same Product launch primitive on the selected execution backend."""
    validate_fresh(request)
    b=get_backend(request.backend);g=make_grid(request.grid,xp=b.xp,real_dtype=b.real_dtype)
    launch=build_launch(request.beams,g,complex_dtype=b.complex_dtype,launch_elements=request.launch_elements,
        context=OpticalLaunchContext(g,request.material.refractive_index,request.grid.z_length_um))
    return core_request(request,launch.A0,backend=b.name if hasattr(b,'name') else request.backend.backend)


def selection_for_policy(policy):
    kind,names=static_product_selection(policy)
    if set(names)-set(ANALYSIS_PRODUCTS):raise ValueError('Unknown unified Analysis product')
    volume_default = normalize_result_policy(policy) in ('interactive','full')
    return products.UnifiedSelection(boundary_intensity=True,intensity_cuts=True,far_field=True,
        intensity_volume=volume_default or 'unified_intensity_volume' in names,
        material_volumes=DEFAULT_MATERIAL_VOLUMES if volume_default else tuple(
            value for key,value in VOLUME_PRODUCTS.items() if key in names))


def resource_plan(request,policy='fast'):
    validate_fresh(request)
    # Estimator consumes shape/dtype metadata only; no launch construction in preflight.
    launch=SimpleNamespace(ndim=3,shape=(len(request.beams.channels),request.grid.Nx,request.grid.Ny),
        dtype=np.dtype('complex64' if request.backend.precision=='float32' else 'complex128'))
    plan = estimate_resources(core_request(request,launch),selection=selection_for_policy(policy))
    real_bytes = 4 if request.backend.precision == 'float32' else 8
    nx, ny = request.grid.Nx, request.grid.Ny
    plan['presentation'] = dict(
        longitudinal_intensity_cut_bytes=(plan['cells']+1)*(nx+ny)*real_bytes,
        far_field_bytes=nx*ny*real_bytes+(nx+ny)*8,
        ephemeral_preview_max_bytes=128*128*4,
        ephemeral_preview_axes_max_bytes=2*128*8,
        backend_preview_workspace_scenario_bytes=nx*ny*(real_bytes+4*8),
        retained_progress_history_bytes=0)
    if normalize_result_policy(policy)=='fast':
        nfields=6 if request.closure.dimension==2 else 5
        plan['presentation']['fast_material_and_optical_preview_upper_bytes']=nfields*4*1024*1024
    if request.backend.backend=='numpy' and request.solver.identity==SCALABLE:
        from .local_planning import assess_local_resources, physical_memory_bytes
        plan['local_assessment']=assess_local_resources(plan,(nx,ny),physical_ram_bytes=physical_memory_bytes())
    return plan


def encode_fresh(request):
    from lcprop.persistence.experiments import encode_beam_stack
    from lcprop.pr.portable_launch import encode_launch_elements
    validate_fresh(request)
    result = dict(schema=request.schema,workflow_identity=request.workflow_identity,
        arithmetic_identity=request.arithmetic_identity,projection_identity=request.projection_identity,
        grid=asdict(request.grid),beams=encode_beam_stack(request.beams),material=asdict(request.material),
        closure=asdict(request.closure),backend=asdict(request.backend),optical_boundary=asdict(request.optical_boundary),
        scattering=None if request.scattering is None else asdict(request.scattering),
        launch_elements=encode_launch_elements(request.launch_elements))
    if request.schema==FRESH_SCHEMA:
        result.update(solver=asdict(request.solver),precision=asdict(core_request(request).precision))
    return result



# Fresh V2 writes complete records, including explicit nulls. This validator
# mirrors that serialized vocabulary, not Python constructor default values.
_FRESH_KEYS = {'schema','workflow_identity','arithmetic_identity','projection_identity',
    'grid','beams','material','closure','backend','optical_boundary','scattering',
    'launch_elements','solver','precision'}
_RASTER_KEYS = {'source_kind','display_name','basename','sha256','width','height',
    'encoded_format','decoded_mode','preprocessing_policy','asset_id','alpha_policy',
    'grayscale_dtype','grayscale_shape','grayscale_sha256','grayscale_base64','encoded_bytes_base64'}


def _raw_record(value, keys, path, *, nullable=()):
    if not isinstance(value,dict):raise ValueError(f'{path}: schema object required')
    missing=set(keys)-set(value);extra=set(value)-set(keys)
    if missing or extra:
        raise ValueError(f'{path}: incomplete schema fields; missing={sorted(missing)}, unexpected={sorted(extra)}')
    for key in keys:
        if value[key] is None and key not in nullable:
            raise ValueError(f'{path}.{key}: null is not permitted')


def _raw_list(value,path):
    if not isinstance(value,list):raise ValueError(f'{path}: schema array required')
    return value


def _validate_fresh_v2_tree(payload):
    """Reject incomplete scientific trees before constructing any request records."""
    from dataclasses import fields
    from lcprop.core.beams import BeamChannel
    from lcprop.optics.screens import ScreenPlacement
    def record(value,cls,path,nullable=()):
        _raw_record(value,{f.name for f in fields(cls)},path,nullable=nullable)
    _raw_record(payload,_FRESH_KEYS,'request',nullable=('scattering',))
    for key,cls,nullable in (
        ('grid',GridSpec,()),
        ('material',PRMaterialSpec,('characteristic_wavenumber_per_um_override',)),
        ('backend',BackendSpec,()),('optical_boundary',TransverseBoundarySpec,()),
        ('closure',PRElectricalClosureSpec,('reservoir_field','background_intensity')),
        ('solver',PRUnifiedSolverSpec,()),('precision',PRMaterialPrecisionSpec,())):
        record(payload[key],cls,key,nullable)
    if payload['scattering'] is not None:
        record(payload['scattering'],PRCanonicalScatteringSpec,'scattering')
    beams=payload['beams'];_raw_record(beams,{'coherence','channels'},'beams')
    for i,channel in enumerate(_raw_list(beams['channels'],'beams.channels')):
        record(channel,BeamChannel,f'beams.channels[{i}]')
    for i,assignment in enumerate(_raw_list(payload['launch_elements'],'launch_elements')):
        path=f'launch_elements[{i}]'
        _raw_record(assignment,{'channel_index','elements'},path)
        for j,element in enumerate(_raw_list(assignment['elements'],path+'.elements')):
            ep=path+f'.elements[{j}]'
            _raw_record(element,{'element_type','interpretation','source','placement','invert','preprocessing_policy'},ep)
            record(element['placement'],ScreenPlacement,ep+'.placement')
            _raw_record(element['source'],_RASTER_KEYS,ep+'.source',
                nullable=('asset_id','encoded_bytes_base64'))


def decode_fresh(payload):
    from lcprop.persistence.experiments import decode_beam_stack
    from lcprop.pr.portable_launch import decode_launch_elements
    expected={'schema','workflow_identity','arithmetic_identity','projection_identity','grid','beams','material',
              'closure','backend','optical_boundary','scattering','launch_elements'}
    if not isinstance(payload,dict):raise ValueError('request: schema object required')
    modern=payload.get('schema')==FRESH_SCHEMA
    if modern:
        _validate_fresh_v2_tree(payload)
        expected|={'solver','precision'}
    if set(payload)!=expected or payload['schema'] not in (FRESH_SCHEMA,LEGACY_FRESH_SCHEMA):raise ValueError('request.schema: invalid identity or missing schema fields')
    v=dict(payload);precision=v.pop('precision',None)
    if modern:
        v['solver']=codec._construct(PRUnifiedSolverSpec,v['solver'])
    v['beams']=decode_beam_stack(v['beams'])
    for k,cls in [('grid',GridSpec),('material',PRMaterialSpec),('closure',PRElectricalClosureSpec),
                  ('backend',BackendSpec),('optical_boundary',TransverseBoundarySpec)]:v[k]=cls(**v[k])
    v['scattering']=None if v['scattering'] is None else PRCanonicalScatteringSpec(**v['scattering'])
    v['launch_elements']=decode_launch_elements(v['launch_elements'],n_channels=len(v['beams'].channels))
    r=UnifiedFreshRequest(**v);validate_fresh(r)
    if modern and asdict(core_request(r).precision)!=precision:raise ValueError("contradictory precision identity")
    return r


@dataclass(frozen=True)
class UnifiedExecutionResult:
    run: object
    result_policy: str
    backend: str

    @property
    def status(self):return self.run.scientific.status


def execute_unified(request, *, result_policy='fast', cancellation_token=None, progress_callback=None):
    policy=normalize_result_policy(result_policy)
    prepared=prepare_request(request)
    from time import perf_counter
    from lcprop.core.execution import RunProgress
    started = perf_counter()
    def accepted(event):
        progress_callback(RunProgress(WORKFLOW_ID, 'running', event.completed_cells,
            event.total_cells, event.z_um, 'z', 'um', perf_counter()-started,
            latest_field_state=event, diagnostics=event.diagnostics))
    result=products.run_unified_products(prepared,selection=selection_for_policy(policy),
        cancellation_token=cancellation_token,observer=accepted if progress_callback is not None else None,
        preview=policy=='fast')
    return UnifiedExecutionResult(result,policy,prepared.backend)


def unified_to_run_data(result):
    """Display only M5 output arrays/axes; never reconstruct or FFT an endpoint."""
    r=result.run;s=r.scientific;items=[]
    suffix=f'({s.status}; accepted z={s.reached_z_um:g} µm)'
    for key,axes in [('boundary_intensity',('x','y')),('far_field_intensity',('s_x','s_y')),
                     ('intensity_cut_x',('z','x')),('intensity_cut_y',('z','y'))]:
        if key not in r.arrays:continue
        coords={a:asnumpy(r.coordinates[{'x':'x_um','y':'y_um','z':'boundary_z_um'}.get(a,a)]) for a in axes}
        unit={a:'1' if a.startswith('s_') else 'um' for a in axes}
        items.append((key,make_field(key,('Output Far-Field Intensity' if key=='far_field_intensity' else 'Output Plane Intensity' if key=='boundary_intensity' else key.replace('_',' ').title())+' '+suffix,asnumpy(r.arrays[key]),axes,
            'intensity',unit,coordinates=coords)))
    if 'intensity_cut_x' in r.arrays and 'intensity_cut_y' in r.arrays:
        x = asnumpy(r.coordinates['x_um']); y = asnumpy(r.coordinates['y_um'])
        ix, iy = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
        for key, source, axis, partner in (
                ('intensity_xz','intensity_cut_x','x','intensity_yz'),
                ('intensity_yz','intensity_cut_y','y','intensity_xz')):
            items.append((key, make_field(key, 'Intensity '+key[-2:]+' '+suffix,
                dict(items)[source].data, ('z',axis), 'intensity', {'z':'um',axis:'um'},
                source_volume_key=None,
                coordinates={'z':asnumpy(r.coordinates['boundary_z_um']), axis:x if axis=='x' else y,
                    'paired_cut_key':partner,'show_in_field_selector':True,'cut_x_index':ix,'cut_y_index':iy,
                    'x_cut_um':float(x[ix]),'y_cut_um':float(y[iy])})))
    from lcprop.products.data_model import CurveData, CurveCollection, DiagnosticData, DiagnosticCollection
    names={'intensity_volume':'Optical Intensity', 'potential_node_volume':'Potential',
        'carrier_node_volume':'Carrier', 'electric_field_x_optical_node_volume':'Optical-response E_x (optical nodes)',
        'electric_field_x_face_volume':'Native x-face E_x', 'electric_field_y_face_volume':'Native y-face E_y'}
    for key,label in names.items():
        if key not in r.arrays or r.arrays[key].shape[0]==0:continue
        values=asnumpy(r.arrays[key]);location=r.locations[key]
        longitudinal=location['longitudinal']
        offset=location.get('offset_um',(0.,0.))
        coords={'z':asnumpy(r.coordinates[longitudinal]),
            'x':asnumpy(r.coordinates['x_um'])+offset[0],
            'y':asnumpy(r.coordinates['y_um'])+offset[1], 'product_location':dict(location)}
        if key=='intensity_volume':
            items.append(('input_intensity',make_field('input_intensity','Input Plane Intensity',values[0],('x','y'),'intensity',{'x':'um','y':'um'},coordinates=coords)))
        kind='intensity' if key=='intensity_volume' else 'field'
        items.append((key,make_field(key,label+' '+suffix,values,('z','x','y'),kind,
            {'x':'um','y':'um','z':'um'},coordinates=coords)))
        items.append((key+'_xy',make_field(key+'_xy',label+' xy '+suffix,values[-1],('x','y'),kind,
            {'x':'um','y':'um'},source_volume_key=key,coordinates=coords)))
    curves=[]
    observations=[{k:v for k,v in record['observations'].items() if 'iteration_' not in k} for record in s.ledger]
    if observations:
        z=asnumpy(r.coordinates['material_z_um'])
        keys=sorted({key for record in observations for key in record if key.endswith(
            ('gauss_rms','gauss_max','flux_divergence_rms','flux_divergence_max','closure_rms','closure_max'))})
        for key in keys:
            curves.append((key,CurveData(key,key.replace('_',' ').title()+' vs z',z,
                np.asarray([record[key] for record in observations]),'Material z','Residual',{'x':'um'})))
        rms=np.asarray([max(abs(v) for k,v in record.items() if k.endswith('_rms')) for record in observations])
        curves.insert(0,('material_rms',CurveData('material_rms','Material residual RMS vs z',z,rms,'Material z','Maximum component RMS',{'x':'um'})))
        for key,label,reduce in [('material_max','Material residual max vs z',max),('carrier_min','Carrier min vs z',min),('carrier_max','Carrier max vs z',max)]:
            def relevant(k):
                return k.endswith(('gauss_max','flux_divergence_max','closure_max')) if key=='material_max' else k.endswith(key)
            values=np.asarray([reduce(v for k,v in record.items() if relevant(k)) for record in observations])
            curves.append((key,CurveData(key,label,z,values,'Material z',label.split(' vs ')[0],{'x':'um'})))
    coords=r.coordinates
    run_data = RunData(WORKFLOW_ID,geometry=Geometry(asnumpy(coords.get('x_um',np.array([]))),
        asnumpy(coords.get('y_um',np.array([]))),asnumpy(coords['boundary_z_um'])),fields=FieldCollection(items),curves=CurveCollection(curves),
        diagnostics=DiagnosticCollection([('summary',DiagnosticData('summary','Accepted unified result',
            dict(status=s.status,completed_cells=s.completed_cells,reached_z_um=s.reached_z_um,reason=s.reason,
                 solver=s.identities.get('solver'),material_failure=(codec._failure_metadata(s.failure) or {}).get('material_failure'),
                 material_iterations=[{k:v for k,v in row['observations'].items() if k in ('iterations','inner_iterations_total','linear_pcg','linear_gmres')} for row in s.ledger],
                 retrieval_policy=result.result_policy,
                 presentation_state=f'{s.status.title()} result — accepted z={s.reached_z_um:g} µm')))]))
    from lcprop.pr.material_previews import add_fields
    return add_fields(run_data,r.viewer_previews)


def _encode_request(r):return EncodedRequest(PortablePayload(encode_fresh(r),{}),r.backend.backend)
def _decode_request(m,a):
    if a:raise ValueError('Fresh unified request has no runtime arrays')
    return decode_fresh(m)
def _encode_result(r,policy=None):
    if policy is not None and normalize_result_policy(policy)!=r.result_policy:raise ValueError('Result policy mismatch')
    s=r.run.scientific
    from lcprop.pr.transport_common import pack_portable
    from lcprop.pr.material_previews import validate
    arrays={'unified_package':np.frombuffer(codec.encode_result(r.run),dtype=np.uint8)}
    metadata=dict(result_policy=r.result_policy,backend=r.backend)
    if r.run.viewer_previews:
        if r.result_policy!='fast':raise ValueError('bounded previews require Fast policy')
        validate(r.run.viewer_previews)
        metadata['viewer_previews']=pack_portable(r.run.viewer_previews,arrays,'viewer_previews')
    p=PortablePayload(metadata,arrays)
    return EncodedResult(p,s.status,None,s.status=='cancelled',s.reason,r.backend,result_policy=r.result_policy)
def _decode_result(m,a):
    if not {'result_policy','backend'}<=set(m) or set(m)-{'result_policy','backend','viewer_previews'} or 'unified_package' not in a:raise ValueError('Invalid unified result envelope')
    if 'viewer_previews' not in m and set(a)!={'unified_package'}:raise ValueError('Unexpected viewer arrays')
    payload=a['unified_package'].tobytes()
    run=codec.decode_result(payload);policy=normalize_result_policy(m['result_policy'])
    import io, json, zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        legacy=json.loads(archive.read('metadata.json'))['products_schema']=='pr_unified_static_products_v1'
    expected = selection_for_policy(policy)
    # Preserve already-saved M6 minimal archives; do not fabricate their spectrum.
    legacy_selection = replace(expected,intensity_volume=False,material_volumes=())
    legacy_minimal = replace(legacy_selection,far_field=False)
    kind, names = static_product_selection(policy)
    if run.selection != expected and not (legacy and (run.selection==legacy_selection or (
            kind != 'full' and 'far_field_intensity' not in names and run.selection==legacy_minimal))):
        raise ValueError('Result selection/policy mismatch')
    if m['backend'] not in ('numpy','cupy') or m['backend']!=run.scientific.identities['backend']:
        raise ValueError('Unified execution backend provenance mismatch')
    if 'viewer_previews' in m:
        from lcprop.pr.transport_common import unpack_portable
        from lcprop.pr.material_previews import validate
        if policy!='fast':raise ValueError('bounded previews require Fast policy')
        from lcprop.pr.transport_common import ARRAY_MARKER
        def references(value):
            if isinstance(value,dict):
                if set(value)=={ARRAY_MARKER}:return {value[ARRAY_MARKER]}
                return set().union(*(references(v) for v in value.values()))
            if isinstance(value,list):return set().union(*(references(v) for v in value))
            return set()
        if references(m['viewer_previews']) != set(a)-{'unified_package'}:raise ValueError('Unexpected viewer arrays')
        previews=unpack_portable(m['viewer_previews'],a);validate(previews)
        run=replace(run,viewer_previews=previews)
    return UnifiedExecutionResult(run,policy,m['backend'])

UNIFIED_OPERATION=WorkflowOperation('pr',WORKFLOW_ID,execute_unified,unified_to_run_data,supports_result_policy=True)
UNIFIED_TRANSPORT_CODEC=TransportCodec(material_id='pr',workflow_id=WORKFLOW_ID,
    request_codec_id='pr_unified_static_fresh',request_codec_version=2,compatible_request_codec_versions=(1,),request_type=UnifiedFreshRequest,
    encode_request=_encode_request,decode_request=_decode_request,
    result_codec_id='pr_unified_static_result',result_codec_version=2,compatible_result_codec_versions=(1,),result_type=UnifiedExecutionResult,
    encode_result=_encode_result,decode_result=_decode_result,encode_result_projection=_encode_result)


def progress_to_run_data(event):
    """Display the detached accepted preview without new scientific work."""
    label=f'Accepted intensity preview (cells {event.completed_cells}/{event.total_cells}; z={event.z_um:g} µm)'
    return RunData(WORKFLOW_ID, geometry=Geometry(event.x_um,event.y_um,np.array([event.z_um])),
        fields=FieldCollection([('accepted_preview',make_field('accepted_preview',label,
            event.preview,('x','y'),'intensity',{'x':'um','y':'um'},
            coordinates={'x':event.x_um,'y':event.y_um}))]))
