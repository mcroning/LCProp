"""Fresh published optical-first requests and material-neutral execution integration.

No continuation or migration of midpoint requests. Only committed streaming
products are exposed; unsupported exact Analysis selections fail explicitly.
"""
from dataclasses import asdict, dataclass, replace
from time import monotonic

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy
from lcprop.core.context import GridSpec
from lcprop.core.execution import RunProgress
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.launch_configuration import LaunchConfiguration, reject_prepared_launch_conflict
from lcprop.pr.published_static import PublishedStaticRequest, PR_PUBLISHED_STATIC_WORKFLOW, PR_PUBLISHED_STATIC_ARITHMETIC
from lcprop.pr.local_plane_workflow import validate_local_plane_domain
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.local_plane_products import LocalPlaneProductSelection
from lcprop.pr.published_static_products import run_published_products
from lcprop.pr.published_static_codec import encode_published_products, decode_published_products
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2, canonical_slab_range
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec, PR_MATERIAL_RESPONSE_FIELD_LINEAR
from lcprop.products.data_model import RunData, Geometry, FieldCollection, make_field
from lcprop.runners.base import WorkflowOperation
from lcprop.transport.codecs import TransportCodec, EncodedRequest, EncodedResult, PortablePayload
from lcprop.transport.result_policy import normalize_result_policy, static_product_selection


PUBLISHED_STATIC_ANALYSIS_PRODUCTS = ('far_field_intensity', 'complex_output')


def validate_published_static_request(request):
    if not isinstance(request, PublishedStaticRequest):
        raise TypeError('requires an explicit published optical-first request; midpoint requests are not migrated')
    for value in (request.grid, request.material, request.beams, request.backend,
                  request.optical_boundary, request.material_response):
        value.validate()
    if request.material_response.model not in (PR_MATERIAL_RESPONSE_FIELD_LINEAR, "nonlinear"):
        raise ValueError('published optical-first Static requires Local-I or nonlinear hopping response')
    if request.optical_boundary.mode != 'periodic' or request.grid.Nx < 3:
        raise ValueError('published optical-first Static requires periodic boundaries and Nx >= 3')
    if len({c.wavelength_um for c in request.beams.channels}) != 1:
        raise ValueError('published optical-first Static requires one wavelength')
    LaunchConfiguration(request.beams, request.launch_elements)
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    request.material_solver.validate()
    n = validate_local_plane_domain(request.grid.z_length_um, request.grid.dz_um)
    if not all(np.isfinite(v) and v > 0 for v in (request.residual_rms_tolerance, request.residual_max_tolerance)):
        raise ValueError('material residual validity limits must be finite and positive')
    if request.scattering is not None:
        if request.scattering.algorithm_version != PR_CANONICAL_SCATTERING_V2:
            raise ValueError('published optical-first Static requires canonical V2 scattering')
        for k in range(n):
            canonical_slab_range(request.scattering, z_start_um=k*request.grid.dz_um,
                dz_um=request.grid.dz_um, z_length_um=request.grid.z_length_um)


def encode_published_static_request(request):
    from lcprop.persistence.experiments import encode_beam_stack
    from lcprop.pr.portable_launch import encode_launch_elements
    validate_published_static_request(request)
    if request.initial_A is not None:
        raise ValueError('saved fresh published optical-first requests cannot contain runtime arrays; continuation is unsupported')
    return dict(schema_version=1, workflow_id=PR_PUBLISHED_STATIC_WORKFLOW,
        arithmetic_id=PR_PUBLISHED_STATIC_ARITHMETIC, grid=asdict(request.grid),
        beams=encode_beam_stack(request.beams), material=asdict(request.material),
        backend=asdict(request.backend), material_response=request.material_response.to_payload(),
        optical_boundary=asdict(request.optical_boundary),
        scattering=None if request.scattering is None else asdict(request.scattering),
        launch_elements=encode_launch_elements(request.launch_elements),
        residual_rms_tolerance=request.residual_rms_tolerance,
        residual_max_tolerance=request.residual_max_tolerance, material_solver=asdict(request.material_solver))


def decode_published_static_request(payload):
    from lcprop.persistence.experiments import decode_beam_stack, require_exact_keys
    from lcprop.pr.portable_launch import decode_launch_elements
    require_exact_keys(payload, required={'schema_version','workflow_id','arithmetic_id','grid','beams',
        'material','backend','material_response','optical_boundary','scattering','launch_elements',
        'residual_rms_tolerance','residual_max_tolerance','material_solver'}, name='published optical-first request')
    if (payload['schema_version'] != 1 or payload['workflow_id'] != PR_PUBLISHED_STATIC_WORKFLOW
            or payload['arithmetic_id'] != PR_PUBLISHED_STATIC_ARITHMETIC):
        raise ValueError('incompatible published optical-first workflow/arithmetic; no midpoint migration')
    beams = decode_beam_stack(payload['beams'])
    request = PublishedStaticRequest(grid=GridSpec(**payload['grid']), beams=beams,
        material=PRMaterialSpec(**payload['material']), backend=BackendSpec(**payload['backend']),
        material_response=PRTransverseMaterialResponseSpec(**payload['material_response']),
        optical_boundary=TransverseBoundarySpec(**payload['optical_boundary']),
        scattering=None if payload['scattering'] is None else PRCanonicalScatteringSpec(**payload['scattering']),
        launch_elements=decode_launch_elements(payload['launch_elements'], n_channels=len(beams.channels)),
        residual_rms_tolerance=payload['residual_rms_tolerance'], residual_max_tolerance=payload['residual_max_tolerance'], material_solver=PRStaticSolverOptions(**payload['material_solver']))
    validate_published_static_request(request)
    return request


def selection_for_policy(policy):
    kind, names = static_product_selection(policy)
    if set(names) - set(PUBLISHED_STATIC_ANALYSIS_PRODUCTS):
        raise ValueError('published optical-first Analysis supports exact far_field_intensity and complex_output only')
    return LocalPlaneProductSelection(intensity_cuts=True, preview=True,
        material_cuts=kind == 'full', source_cuts=kind == 'full', residual_cuts=kind == 'full',
        optical_cuts=kind == 'full', endpoint=kind == 'full' or 'complex_output' in names,
        far_field=kind == 'full' or 'far_field_intensity' in names)


@dataclass(frozen=True)
class PublishedStaticExecutionResult:
    run: object
    result_policy: str
    backend: str

    @property
    def status(self):
        return self.run.scientific.status


def execute_published_static(request, *, result_policy='fast', cancellation_token=None, progress_callback=None):
    validate_published_static_request(request)
    policy = normalize_result_policy(result_policy)
    selected = selection_for_policy(policy)
    start = monotonic()
    def progress(event):
        if progress_callback is not None:
            last = event.last_cell or {}
            progress_callback(RunProgress(PR_PUBLISHED_STATIC_WORKFLOW, 'running', event.completed_cells,
                event.requested_cells, event.reached_z_um, 'z', 'um', monotonic()-start,
                message='Accepted published optical-first cells', diagnostics=dict(
                    phase='cells', material_residual_rms=last.get('residual_rms'),
                    material_residual_max=last.get('residual_max'))))
    run = run_published_products(request, selection=selected, cancellation_token=cancellation_token,
                                  progress_callback=progress)
    return PublishedStaticExecutionResult(run, policy, run.scientific.launch_identity['backend']
        if run.scientific.launch_identity else request.backend.backend)


def published_static_to_run_data(result):
    p, s = result.run.products, result.run.scientific
    items = []
    for name, pair in p.cuts.items():
        if name == 'optical':  # Complex channels are exact Analysis data, not scalar display images.
            continue
        z = p.boundary_z_um if name == 'intensity' else p.material_z_um
        if not len(z):
            continue
        for suffix, transverse, a in zip(('xz','yz'), ('x','y'), pair):
            key = f'published_static_{name}_{suffix}'
            items.append((key, make_field(key, f'{name.title()} {suffix}', a, ('z',transverse),
                'intensity' if name in ('intensity','source') else 'pr_space_charge',
                {'z':'um',transverse:'um'}, coordinates={'z':z,transverse:p.x_um if transverse=='x' else p.y_um})))
    if p.preview is not None:
        items.append(('intensity_preview', make_field('intensity_preview', 'Orthogonal-slice intensity preview',
            p.preview, ('z','x','y'), 'intensity', {'z':'um','x':'um','y':'um'},
            coordinates={'z':p.boundary_z_um,'x':p.preview_x_um,'y':p.preview_y_um})))
    if s.far_field is not None:
        f = s.far_field
        items.append(('far_field_intensity', make_field('far_field_intensity', 'Exact final far-field intensity',
            asnumpy(f.intensity), ('s_x','s_y'), 'intensity', {'s_x':'1','s_y':'1'},
            coordinates={'s_x':asnumpy(f.s_x),'s_y':asnumpy(f.s_y)})))
    return RunData(PR_PUBLISHED_STATIC_WORKFLOW, geometry=Geometry(p.x_um,p.y_um,p.boundary_z_um),
                   fields=FieldCollection(items))


def _encode_request(request):
    payload = encode_published_static_request(replace(request, initial_A=None))
    arrays = {} if request.initial_A is None else {'initial_A':asnumpy(request.initial_A)}
    return EncodedRequest(PortablePayload(payload, arrays), request.backend.backend)


def _decode_request(metadata, arrays):
    if set(arrays) - {'initial_A'}:
        raise ValueError('unexpected published optical-first request arrays; no continuation')
    return replace(decode_published_static_request(metadata), initial_A=arrays.get('initial_A'))


def _encode_result(result, policy=None):
    if policy is not None and normalize_result_policy(policy) != result.result_policy:
        raise ValueError('published optical-first results must be constructed with the requested product policy')
    s = result.run.scientific
    payload = PortablePayload(dict(result_policy=result.result_policy, backend=result.backend),
        {'published_static_package':np.frombuffer(encode_published_products(result.run), dtype=np.uint8)})
    return EncodedResult(payload, s.status, None, s.status == 'cancelled', s.reason,
                         result.backend, result_policy=result.result_policy)


def _decode_result(metadata, arrays):
    if set(metadata) != {'result_policy','backend'} or set(arrays) != {'published_static_package'}:
        raise ValueError('invalid published optical-first result envelope')
    run = decode_published_products(arrays['published_static_package'].tobytes())
    policy = normalize_result_policy(metadata['result_policy'])
    if run.products.metadata['selection'] != asdict(selection_for_policy(policy)):
        raise ValueError('published optical-first result policy/selection mismatch')
    return PublishedStaticExecutionResult(run, policy, metadata['backend'])


PUBLISHED_STATIC_OPERATION = WorkflowOperation('pr', PR_PUBLISHED_STATIC_WORKFLOW, execute_published_static,
                                         published_static_to_run_data, supports_result_policy=True)
PUBLISHED_STATIC_TRANSPORT_CODEC = TransportCodec(
    material_id='pr', workflow_id=PR_PUBLISHED_STATIC_WORKFLOW,
    request_codec_id='pr_published_static_request', request_codec_version=1, request_type=PublishedStaticRequest,
    encode_request=_encode_request, decode_request=_decode_request,
    result_codec_id='pr_published_static_result', result_codec_version=1, result_type=PublishedStaticExecutionResult,
    encode_result=_encode_result, decode_result=_decode_result, encode_result_projection=_encode_result)
