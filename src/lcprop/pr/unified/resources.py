"""Headless planning estimates, not measured native peaks or solver guarantees."""
import math
from .workflow import UnifiedProductSelection, _validate
from .products import UnifiedSelection

RESOURCE_SCHEMA = 'pr_unified_static_resource_plan_v1'


def estimate_resources(request, *, selection=UnifiedSelection()):
    cells=_validate(request,UnifiedProductSelection())
    selection.validate(request.spatial)
    nx,ny=request.spatial.field_shape;n=nx*ny
    if (request.initial_A.ndim!=3 or request.initial_A.shape[1:]!=(nx,ny)
            or request.initial_A.dtype.name!=('complex64' if request.precision.state_dtype=='float32' else 'complex128')):
        raise ValueError('launch geometry/precision mismatch')
    nch=request.initial_A.shape[0]
    if nch<1: raise ValueError('channel count must be positive')
    r=4 if request.precision.state_dtype=='float32' else 8;c=2*r
    dim=request.spatial.dimension
    b=math.prod(request.spatial.harmonic_shape)
    domain=nx if dim==1 else n
    unknowns=2*domain+dim  # conservative coupled q/psi/b, also for zero-flux fast solve
    state=(4*n+2*b)*r  # accepted + candidate q/psi/b
    optical=2*nch*n*c
    material_workspace=12*n*r  # stencil, residual, candidate/carrier/phase transverse allowance
    # Sparse assembly and double solve buffers; one reduced column at a time.
    nnz=(12 if dim==1 else 20)*domain+2*dim*domain+dim*dim
    mixed=nnz*(8+4)+(unknowns+1)*4+4*unknowns*8
    face_flux=2*dim*n*r
    scattering=3*n*r+n*c if request.scattering is not None else 0
    material_observation=bool(selection.material_fields or selection.material_cuts or selection.material_volumes)
    optical_observation=selection.optical_cuts or selection.intensity_cuts or selection.intensity_volume
    observation=((nch*n*c if optical_observation or selection.launch else 0)
        + ((3*n+b)*r if material_observation else 0)
        + ((nx+ny)*r if material_observation or optical_observation or selection.launch else 0))
    selected=0
    if selection.launch:selected+=nch*n*c
    if selection.boundary_intensity:selected+=n*r
    if selection.optical_cuts:selected+=(cells+1)*nch*(nx+ny)*c
    if selection.intensity_cuts:selected+=(cells+1)*(nx+ny)*r
    last_fields=0
    for name in selection.material_fields:
        last_fields+=(b if name=='harmonic_field' else n)*(c if name=='material_phase_optical_node' else r)
        selected+=(b if name=='harmonic_field' else n)*(c if name=='material_phase_optical_node' else r)
    for name in selection.material_cuts:selected+=cells*(nx+ny)*(c if name=='material_phase_optical_node' else r)
    volumes={}
    if selection.intensity_volume:volumes['intensity_volume']=(cells+1)*n*r
    for name in selection.material_volumes:
        volumes[name+'_volume']=cells*n*(c if name=='material_phase_optical_node' else r)
    volume_bytes=sum(volumes.values())
    selected+=volume_bytes
    if selection.far_field:selected+=n*r+(nx+ny)*8
    coordinates=(2*cells+1)*8+(nx+ny)*r
    canonical_endpoint=nch*n*c+(2*n+b)*r
    output=canonical_endpoint+selected+coordinates
    return dict(schema=RESOURCE_SCHEMA,active_dimensions=dim,independent_columns=ny if dim==1 else 1,
        cells=cells,precision_identity=request.precision.identity,
        bytes=dict(scientific_state=state,optical_working_field=optical,
            material_trial_workspace=material_workspace,float64_sparse_workspace=mixed,
            face_fluxes=face_flux,scattering=scattering,isolated_observation=observation,
            selected_products=selected,coordinates=coordinates,canonical_final_outputs=canonical_endpoint,
            selected_product_transaction_workspace=last_fields+selected,
            optical_fft_workspace_scenario=4*nch*n*c+4*n*8,
            host_serialization_array_and_archive_scenario=3*output),
        direct_solver=dict(domain_nodes=domain,estimated_unknowns=unknowns,
            dense_factorization_scenario_bytes=3*unknowns*unknowns*8,
            note='Fill/workspace is solver dependent; dense scenario is not a guaranteed peak bound.'),
        longitudinal_full_volume_bytes=volume_bytes,full_volumes_supported=True,
        longitudinal_product_bytes=volumes,
        host_retrieval_array_bytes=output,serialized_package_array_bytes=output,
        measured_native_peak=None,
        limitations='Array/workspace planning only. Excludes allocator caches, FFT plans, library overhead and metadata JSON bytes. No large-2D feasibility claim.')
