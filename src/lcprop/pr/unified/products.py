"""Selected runtime products, on isolated M4 observations; no public dispatch."""
from dataclasses import dataclass, replace
from contextlib import nullcontext
import sys
from types import SimpleNamespace

from lcprop.optics.splitstep import total_intensity
from lcprop.pr.optical_response import delta_n_from_E
from lcprop.pr.scattering import canonical_slab_range
from .operators import Geometry, flux
from .projection import PROJECTION_ID, electric_field_face, electric_field_optical_node
from .workflow import UnifiedProductSelection, run_unified_static

PRODUCTS_SCHEMA = 'pr_unified_static_products_v1'
FIELDS = ('q_log_carrier', 'carrier_node', 'potential_node', 'harmonic_field', 'transport_intensity_node',
          'electric_field_x_face', 'electric_field_y_face', 'hopping_current_x_face',
          'hopping_current_y_face', 'electric_field_x_optical_node',
          'delta_n_optical_node', 'material_phase_optical_node')


@dataclass(frozen=True)
class UnifiedSelection:
    """No full z volumes. Canonical accepted state/endpoint are mandatory.

    Fields mean last accepted material plane; cuts mean right-endpoint series.
    Intensities are raw physical optical products, not preview normalization.
    """
    launch: bool = False
    boundary_intensity: bool = False
    optical_cuts: bool = False
    intensity_cuts: bool = False
    material_fields: tuple[str, ...] = ()
    material_cuts: tuple[str, ...] = ()
    far_field: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'material_fields', tuple(self.material_fields))
        object.__setattr__(self, 'material_cuts', tuple(self.material_cuts))
        for key in ('launch','boundary_intensity','optical_cuts','intensity_cuts','far_field'):
            if type(getattr(self,key)) is not bool:
                raise ValueError('product selection requires booleans')
        for names in (self.material_fields,self.material_cuts):
            if len(set(names)) != len(names) or any(n not in FIELDS for n in names):
                raise ValueError('unknown or duplicate material product')
        if 'harmonic_field' in self.material_cuts:
            raise ValueError('harmonic field is a domain/batch quantity, not a transverse cut')

    def validate(self, spatial):
        if spatial.dimension == 1 and any('_y_' in n for n in self.material_fields+self.material_cuts):
            raise ValueError('reduced transport has no y field/current products')


@dataclass(frozen=True)
class UnifiedSelectedResult:
    scientific: object
    selection: UnifiedSelection
    arrays: dict
    coordinates: dict
    locations: dict
    schema: str = PRODUCTS_SCHEMA
    next_address: dict | None = None


def material_products(state, I, names, identities, width):
    """Only requested products; native current uses the unchanged fitted flux."""
    xp = sys.modules[state.backend]
    out = {}
    s = state.spatial
    for name, value in (('q_log_carrier',state.q),('potential_node',state.psi),('harmonic_field',state.b)):
        if name in names: out[name] = value
    if 'transport_intensity_node' in names: out['transport_intensity_node'] = I
    if 'carrier_node' in names: out['carrier_node'] = xp.exp(state.q)
    for c in ('x','y'):
        name = f'electric_field_{c}_face'
        if name in names: out[name] = electric_field_face(state,component=c)
    currents = [n for n in names if n.startswith('hopping_current_')]
    if currents:
        g = Geometry(s.active_shape,s.normalized_lengths,SimpleNamespace(xp=xp))
        if s.batch_shape:
            current = xp.empty_like(I)
            for j in range(s.batch_shape[0]):
                current[:,j] = flux(g,I[:,j],state.q[:,j],state.psi[:,j],state.b[j])[0]
            out['hopping_current_x_face'] = current
        else:
            values = flux(g,I,state.q,state.psi,state.b)
            for name in currents: out[name] = values[s.active_axes.index(name.split('_')[2])]
    optical = {'electric_field_x_optical_node','delta_n_optical_node','material_phase_optical_node'}
    if set(names)&optical:
        E = electric_field_optical_node(state)
        if 'electric_field_x_optical_node' in names: out['electric_field_x_optical_node'] = E
        if set(names)&{'delta_n_optical_node','material_phase_optical_node'}:
            dn = delta_n_from_E(E,gain_length_product=identities['material_parameters']['gain_length_product'],
                interaction_length_um=identities['grid']['z_length_um'],wavelength_um=identities['wavelength_um'])
            if 'delta_n_optical_node' in names: out['delta_n_optical_node'] = dn
            if 'material_phase_optical_node' in names:
                import math
                out['material_phase_optical_node'] = xp.exp(1j*(2*math.pi/identities['wavelength_um'])*width*dn)
    return out


class _Collector:
    """Internal only. Prepare a prospective collection without mutating previous.

    M4 owns atomic promotion. Snapshots are detached and mutation-checked.
    Retain only requested last planes and copied reduced cuts, never snapshots
    as a longitudinal series. No external user callback is invoked.
    """
    def __init__(self, selection): self.selection = selection

    @property
    def needs_material(self): return bool(self.selection.material_fields or self.selection.material_cuts)

    @property
    def needs_optical(self): return self.selection.optical_cuts or self.selection.intensity_cuts

    def prepare(self, previous, A, state, I, grid, record, identities):
        xp = sys.modules[identities['backend']]
        selection = self.selection
        ix,iy = int(xp.argmin(abs(grid.x_um)).item()),int(xp.argmin(abs(grid.y_um)).item())
        if previous is None:
            previous = dict(series={},last={},launch=A if selection.launch else None,
                            x=grid.x_um,y=grid.y_um,ix=ix,iy=iy)
        series = dict(previous['series'])
        def append(name, value):
            pair = (value[..., :,iy].copy(),value[..., ix,:].copy())
            series[name] = series.get(name,())+(pair,)
        if selection.optical_cuts: append('optical',A)
        if selection.intensity_cuts:
            append('intensity',total_intensity(A,coherence_groups=identities['coherence_groups'],xp=xp))
        last = previous['last']
        if state is not None:
            names = tuple(dict.fromkeys(selection.material_fields+selection.material_cuts))
            values = material_products(state,I,names,identities,record['material_weight_um'])
            for name in selection.material_cuts: append(name,values[name])
            last = {name:values[name] for name in selection.material_fields}
        return dict(previous,series=series,last=last)


def product_locations(selection, spatial, dx, dy):
    """Semantic locations; explicit transverse coordinates are stored separately."""
    result = {}
    for name in set(selection.material_fields+selection.material_cuts):
        axis = 'y' if '_y_' in name else 'x'
        face = name.endswith('_face')
        result[name] = dict(transverse='oriented_face' if face else ('harmonic_domain' if name=='harmonic_field' else 'node'),
            component=axis if face or name.startswith('electric_field_') else None,
            orientation='positive' if face else None,
            offset_um=[dx/2 if face and axis=='x' else 0.,dy/2 if face and axis=='y' else 0.],
            longitudinal='material_z_um',
            projection=PROJECTION_ID if name.endswith('_optical_node') else None)
    return result


def run_unified_products(request, *, selection=UnifiedSelection(), cancellation_token=None):
    context = request.initial_A.device if request.backend == 'cupy' else nullcontext()
    with context:
        return _run_products(request,selection,cancellation_token)


def next_address(request, count, ledger):
    slab=None
    if request.scattering is not None:
        slab=(ledger[-1]['canonical_slabs'][1] if ledger else
              canonical_slab_range(request.scattering,z_start_um=0.,dz_um=request.grid.dz_um,
                                   z_length_um=request.grid.z_length_um).start)
    return dict(cell_index=count,canonical_slab=slab)


def _run_products(request, selection, cancellation_token):
    selection.validate(request.spatial)
    collector = _Collector(selection)
    observing = collector.needs_material or collector.needs_optical or selection.launch
    result = run_unified_static(request,
        selection=UnifiedProductSelection(material_state=True,far_field=selection.far_field),
        cancellation_token=cancellation_token,_collector=collector if observing else None)
    xp = sys.modules[request.backend]
    arrays = {}
    collection = result.collection
    if not observing and result.boundary_field is not None:
        collection = dict(series={},last={},launch=None,x=result.products['x_um'],y=result.products['y_um'])
    coordinates = dict(boundary_z_um=xp.asarray(result.boundary_z_um,dtype=xp.float64),
                       material_z_um=xp.asarray(result.material_z_um,dtype=xp.float64))
    try:
        return _finish_products(request,selection,result,xp,arrays,coordinates,collection)
    except Exception as exc:
        failure=dict(result.failure) if result.failure is not None else dict(
            stage='selected_products',cell_index=None,type=type(exc).__name__,reason=str(exc))
        if result.failure is not None: failure['selected_product_failure']=str(exc)
        result=replace(result,status='failed',reason=failure['reason'],failure=failure)
        # Keep accepted canonical science and every already constructed product.
        locations=product_locations(selection,request.spatial,
            request.grid.x_aperture_um/request.grid.Nx,request.grid.y_aperture_um/request.grid.Ny)
        return UnifiedSelectedResult(replace(result,collection=None,products={}),selection,
            arrays,coordinates,locations,next_address=next_address(request,result.completed_cells,result.ledger))


def _finish_products(request,selection,result,xp,arrays,coordinates,collection):
    if collection is not None:
        coordinates.update(x_um=collection['x'],y_um=collection['y'])
        if selection.launch: arrays['launch'] = collection['launch']
        arrays.update(collection['last'])
        for name,pairs in collection['series'].items():
            for axis,index in (('x',0),('y',1)):
                arrays[name+'_cut_'+axis] = xp.stack([pair[index] for pair in pairs])
        # Canonical products share accepted state buffers, not duplicate snapshots.
        if result.material_state is not None:
            for name,attr in (('q_log_carrier','q'),('potential_node','psi'),('harmonic_field','b')):
                if name in arrays: arrays[name] = getattr(result.material_state,attr)
        if selection.boundary_intensity:
            arrays['boundary_intensity'] = total_intensity(result.boundary_field,
                coherence_groups=result.identities['coherence_groups'],xp=xp)
        spectrum = result.products.get('far_field')
        if spectrum is not None:
            arrays['far_field_intensity'] = spectrum.intensity
            coordinates.update(s_x=spectrum.s_x,s_y=spectrum.s_y)
    locations = product_locations(selection,request.spatial,
        request.grid.x_aperture_um/request.grid.Nx,request.grid.y_aperture_um/request.grid.Ny)
    # No hidden collector snapshots or repeated legacy products in final result.
    result = replace(result,collection=None,products={})
    return UnifiedSelectedResult(result,selection,arrays,coordinates,locations,
        next_address=next_address(request,result.completed_cells,result.ledger))
