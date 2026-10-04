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
from .specs import PRUnifiedSpatialSpec, PRElectricalClosureSpec, PRMaterialPrecisionSpec, MIXED_PRECISION, DOUBLE_PRECISION
from .projection import PROJECTION_ID
from .resources import estimate_resources

ANALYSIS_PRODUCTS = ('far_field_intensity', 'complex_output')
FRESH_SCHEMA = 'pr_unified_static_fresh_launch_v1'


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


def core_request(request, initial_A=None, *, backend=None):
    """Translate metadata without an allocation; explicit launch preparation below."""
    d=request.closure.dimension;g=request.grid;k=request.material.characteristic_wavenumber_per_um
    spatial=PRUnifiedSpatialSpec((g.Nx,) if d==1 else (g.Nx,g.Ny),
        (k*g.x_aperture_um,) if d==1 else (k*g.x_aperture_um,k*g.y_aperture_um),
        active_axes=('x',) if d==1 else ('x','y'),batch_shape=(g.Ny,) if d==1 else (),
        batch_axes=('y',) if d==1 else ())
    dtype=request.backend.precision
    precision=PRMaterialPrecisionSpec(identity=MIXED_PRECISION if dtype=='float32' else DOUBLE_PRECISION,
        state_dtype=dtype,output_dtype=dtype)
    return UnifiedStaticRequest(grid=g,spatial=spatial,closure=request.closure,initial_A=initial_A,
        material=request.material,precision=precision,backend=backend or request.backend.backend,
        wavelength_um=request.beams.channels[0].wavelength_um,coherence_groups=request.beams.coherence_groups,
        scattering=request.scattering,workflow_identity=request.workflow_identity,
        arithmetic_identity=request.arithmetic_identity,projection_identity=request.projection_identity)


def validate_fresh(request):
    for value in (request.grid,request.beams,request.material,request.backend,request.optical_boundary):value.validate()
    if request.backend.backend not in ('numpy','cupy'):
        raise ValueError('Unified Static requires an explicit NumPy or CuPy backend')
    if request.optical_boundary.mode!='periodic':raise ValueError('Unified Static supports periodic boundaries only')
    if len({c.wavelength_um for c in request.beams.channels})!=1:raise ValueError('Unified Static requires one wavelength')
    LaunchConfiguration(request.beams,request.launch_elements)
    if request.closure.dimension==2 and request.grid.Nx*request.grid.Ny>12288:
        raise ValueError('Full x-y exceeds the 12,288 active-node reference-solver limit; H200 does not remove this algorithmic limit')
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
    if set(names)-set(ANALYSIS_PRODUCTS):raise ValueError('Unified Analysis supports far_field_intensity and complex_output')
    return products.UnifiedSelection(boundary_intensity=True,intensity_cuts=True,
        far_field=kind=='full' or 'far_field_intensity' in names)


def resource_plan(request,policy='fast'):
    validate_fresh(request)
    # Estimator consumes shape/dtype metadata only; no launch construction in preflight.
    launch=SimpleNamespace(ndim=3,shape=(len(request.beams.channels),request.grid.Nx,request.grid.Ny),
        dtype=np.dtype('complex64' if request.backend.precision=='float32' else 'complex128'))
    return estimate_resources(core_request(request,launch),selection=selection_for_policy(policy))


def encode_fresh(request):
    from lcprop.persistence.experiments import encode_beam_stack
    from lcprop.pr.portable_launch import encode_launch_elements
    validate_fresh(request)
    return dict(schema=FRESH_SCHEMA,workflow_identity=request.workflow_identity,
        arithmetic_identity=request.arithmetic_identity,projection_identity=request.projection_identity,
        grid=asdict(request.grid),beams=encode_beam_stack(request.beams),material=asdict(request.material),
        closure=asdict(request.closure),backend=asdict(request.backend),optical_boundary=asdict(request.optical_boundary),
        scattering=None if request.scattering is None else asdict(request.scattering),
        launch_elements=encode_launch_elements(request.launch_elements))


def decode_fresh(payload):
    from lcprop.persistence.experiments import decode_beam_stack
    from lcprop.pr.portable_launch import decode_launch_elements
    expected={'schema','workflow_identity','arithmetic_identity','projection_identity','grid','beams','material',
              'closure','backend','optical_boundary','scattering','launch_elements'}
    if set(payload)!=expected or payload['schema']!=FRESH_SCHEMA:raise ValueError('Invalid unified fresh request identity')
    v=dict(payload);v.pop('schema');v['beams']=decode_beam_stack(v['beams'])
    for k,cls in [('grid',GridSpec),('material',PRMaterialSpec),('closure',PRElectricalClosureSpec),
                  ('backend',BackendSpec),('optical_boundary',TransverseBoundarySpec)]:v[k]=cls(**v[k])
    v['scattering']=None if v['scattering'] is None else PRCanonicalScatteringSpec(**v['scattering'])
    v['launch_elements']=decode_launch_elements(v['launch_elements'],n_channels=len(v['beams'].channels))
    r=UnifiedFreshRequest(**v);validate_fresh(r);return r


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
    result=products.run_unified_products(prepared,selection=selection_for_policy(policy),cancellation_token=cancellation_token)
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
        items.append((key,make_field(key,key.replace('_',' ').title()+' '+suffix,asnumpy(r.arrays[key]),axes,
            'intensity',unit,coordinates=coords)))
    coords=r.coordinates
    return RunData(WORKFLOW_ID,geometry=Geometry(asnumpy(coords.get('x_um',np.array([]))),
        asnumpy(coords.get('y_um',np.array([]))),asnumpy(coords['boundary_z_um'])),fields=FieldCollection(items))


def _encode_request(r):return EncodedRequest(PortablePayload(encode_fresh(r),{}),r.backend.backend)
def _decode_request(m,a):
    if a:raise ValueError('Fresh unified request has no runtime arrays')
    return decode_fresh(m)
def _encode_result(r,policy=None):
    if policy is not None and normalize_result_policy(policy)!=r.result_policy:raise ValueError('Result policy mismatch')
    s=r.run.scientific
    p=PortablePayload(dict(result_policy=r.result_policy,backend=r.backend),
        {'unified_package':np.frombuffer(codec.encode_result(r.run),dtype=np.uint8)})
    return EncodedResult(p,s.status,None,s.status=='cancelled',s.reason,r.backend,result_policy=r.result_policy)
def _decode_result(m,a):
    if set(m)!={'result_policy','backend'} or set(a)!={'unified_package'}:raise ValueError('Invalid unified result envelope')
    run=codec.decode_result(a['unified_package'].tobytes());policy=normalize_result_policy(m['result_policy'])
    if run.selection!=selection_for_policy(policy):raise ValueError('Result selection/policy mismatch')
    if m['backend'] not in ('numpy','cupy') or m['backend']!=run.scientific.identities['backend']:
        raise ValueError('Unified execution backend provenance mismatch')
    return UnifiedExecutionResult(run,policy,m['backend'])

UNIFIED_OPERATION=WorkflowOperation('pr',WORKFLOW_ID,execute_unified,unified_to_run_data,supports_result_policy=True)
UNIFIED_TRANSPORT_CODEC=TransportCodec(material_id='pr',workflow_id=WORKFLOW_ID,
    request_codec_id='pr_unified_static_fresh',request_codec_version=1,request_type=UnifiedFreshRequest,
    encode_request=_encode_request,decode_request=_decode_request,
    result_codec_id='pr_unified_static_result',result_codec_version=1,result_type=UnifiedExecutionResult,
    encode_result=_encode_result,decode_result=_decode_result,encode_result_projection=_encode_result)
