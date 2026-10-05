"""Bounded, non-authoritative final-state material products for the common viewer.

Block means preserve the established preview policy, not pointwise extrema.
Derivatives are evaluated on the original grid before reduction, never on previews.
"""
import numpy as np
from lcprop.pr.visualization import _bin_edges, _block_centers, FAST_MPR_TARGET_BYTES

LABELS = {
    'potential_node': 'Material potential ψ', 'carrier_node': 'Normalized carrier',
    'electric_field_x_optical_node': 'Projected material response E_x (optical nodes)',
    'electric_field_x_face': 'Material E_x (x faces)',
    'electric_field_y_face': 'Material E_y (y faces)',
    'optical_intensity': 'Optical intensity', 'psi': 'Material potential ψ',
    'E_x': 'Material E_x', 'E_y': 'Material E_y', 'carrier': 'Normalized carrier',
}
ANALYSIS_LABELS = {
    'unified_intensity_volume': 'Optical intensity volume',
    'unified_potential_volume': 'Material potential ψ volume',
    'unified_carrier_volume': 'Normalized carrier volume',
    'unified_optical_field_volume': 'Projected material response E_x volume (optical nodes)',
    'unified_face_x_volume': 'Material E_x volume (x faces)',
    'unified_face_y_volume': 'Material E_y volume (y faces)',
}

def reduce_plane(value, xp):
    """Backend-first block means, float64 accumulation, original output dtype."""
    if value.ndim != 2 or value.dtype.kind != 'f':
        raise ValueError('preview requires a real plane')
    xe, ye = (_bin_edges(n,96) for n in value.shape)
    a = value.astype(xp.float64, copy=False)
    if xp is np:
        a = np.add.reduceat(a,xe[:-1],axis=0)
        a = np.add.reduceat(a,ye[:-1],axis=1)
    else:
        from lcprop.pr.visualization import _movie_reduceat
        a = _movie_reduceat(a,xe,axis=0,xp=xp)
        a = _movie_reduceat(a,ye,axis=1,xp=xp)
    a /= xp.asarray(np.diff(xe)[:,None]*np.diff(ye)[None,:])
    return a.astype(value.dtype,copy=False),xe,ye

def plane_indices(n, shape, itemsize):
    maximum = max(1,FAST_MPR_TARGET_BYTES//(min(shape[0],96)*min(shape[1],96)*itemsize))
    return np.unique(np.rint(np.linspace(0,n-1,min(n,maximum))).astype(int))

def record(data, *, name, x, y, z, source_shape, indices, xe, ye):
    result=dict(data=data,metadata=dict(schema='pr_bounded_material_preview_v1',
        name=name,source_shape=list(source_shape),retained_shape=list(data.shape),
        dtype=str(data.dtype),bytes=int(data.nbytes),visualization_only=True,
        reduction='contiguous_block_arithmetic_mean; actual longitudinal planes',
        x_block_bounds=xe.tolist(),y_block_bounds=ye.tolist(),z_indices=list(map(int,indices)),
        coordinates={'x':_block_centers(np.asarray(x),xe).tolist(),
                     'y':_block_centers(np.asarray(y),ye).tolist(),'z':list(map(float,z))}))
    validate({name:result})
    return result

def validate(previews):
    if not isinstance(previews,dict) or set(previews)-set(LABELS):
        raise ValueError('unknown material preview')
    for name,r in previews.items():
        a=r['data'];m=r['metadata'];c=m['coordinates']
        if (m['schema']!='pr_bounded_material_preview_v1' or m['name']!=name
                or a.ndim!=3 or a.dtype.kind!='f' or a.nbytes>FAST_MPR_TARGET_BYTES
                or max(a.shape[1:])>96 or list(a.shape)!=m['retained_shape']
                or str(a.dtype)!=m['dtype'] or a.nbytes!=m['bytes']
                or tuple(len(c[k]) for k in ('z','x','y'))!=a.shape
                or len(m['z_indices'])!=a.shape[0]
                or not np.all(np.isfinite(a))
                or any(not np.all(np.isfinite(c[k])) or np.any(np.diff(c[k])<=0) for k in ('x','y','z'))):
            raise ValueError('invalid/beyond-budget material preview')

def add_fields(run, previews):
    from lcprop.products.data_model import make_field
    validate(previews)
    for name,r in previews.items():
        key='preview_'+name;m=r['metadata'];a=r['data'];coords=dict(m['coordinates'],preview_metadata=m)
        label=LABELS[name]+' (bounded final accepted-state preview)'
        kind='intensity' if name=='optical_intensity' else 'field'
        run.fields.add(key,make_field(key,label,a,('z','x','y'),kind,
            {'z':'um','x':'um','y':'um'},coordinates=coords,
            quantity=name,value_unit='1/µm²' if kind=='intensity' else '1',
            colormap='viridis' if name in ('carrier','carrier_node','optical_intensity') else 'coolwarm'))
        run.fields.add(key+'_xy',make_field(key+'_xy',label+' xy',a[-1],('x','y'),kind,
            {'x':'um','y':'um'},source_volume_key=key,coordinates=coords,
            quantity=name,value_unit='1/µm²' if kind=='intensity' else '1',
            colormap='viridis' if name in ('carrier','carrier_node','optical_intensity') else 'coolwarm'))
    return run

def td_previews(result, *, transverse=False):
    """Observe retained accepted TD state; no time history or equilibrium solve."""
    old=result.diagnostics.get('material_previews',{})
    state=result.psi_final if transverse else result.E_final
    if state is None:
        validate(old);return old
    g=result.grid_summary;n,nx,ny=state.shape
    x=(np.arange(nx)-.5*(nx-1))*g['dx_um'];y=(np.arange(ny)-.5*(ny-1))*g['dy_um']
    offset=1 if result.diagnostics.get('optical_coupling')=='frozen_material_published_optical_first_v1' else 0
    indices=plane_indices(n,(nx,ny),state.dtype.itemsize);values={}
    for iz in indices:
        if transverse:
            from lcprop.pr.transverse.transport import state_from_potential
            p=result.resolved_profile
            s=state_from_potential(state[iz],dx_normalized=p['dx_normalized'],dy_normalized=p['dy_normalized'],
                h_y=p['dielectric']['h_y'],applied_field_x=p['boundary']['applied_field_x'],xp=np)
            planes={'psi':s.psi,'E_x':s.E_x,'E_y':s.E_y,'carrier':s.carrier_density}
        else: planes={'E_x':state[iz]}
        for name,a in planes.items():
            reduced,xe,ye=reduce_plane(a,np);values.setdefault(name,[]).append(reduced)
    return {name:record(np.stack(planes),name=name,x=x,y=y,z=(indices+offset)*g['dz_um'],
        source_shape=state.shape,indices=indices,xe=xe,ye=ye) for name,planes in values.items()}

class StaticPreviews:
    """Transactional, bounded per-plane reduction; never retain full planes."""
    def __init__(self, request):
        self.request=request
        self.names=('potential_node','carrier_node','electric_field_x_optical_node','electric_field_x_face')
        if request.spatial.dimension==2:self.names+=('electric_field_y_face',)

    def prepare(self, previous, A, state, I, grid, record_, identities, xp):
        from lcprop.optics.splitstep import total_intensity
        from lcprop.pr.unified.products import material_products
        r=self.request;count=round(r.grid.z_length_um/r.grid.dz_um)
        values={'optical_intensity':total_intensity(A,coherence_groups=identities['coherence_groups'],xp=xp)}
        if state is not None:values.update(material_products(state,I,self.names,identities,record_['material_weight_um']))
        previous=previous or {};out=dict(previous)
        for name,a in values.items():
            optical=name=='optical_intensity';idx=0 if record_ is None else record_['cell_index']+(1 if optical else 0)
            z=0. if record_ is None else record_['z_end_um']
            # Reserve a slot for the latest accepted plane on partial runs.
            maximum=FAST_MPR_TARGET_BYTES//(min(a.shape[0],96)*min(a.shape[1],96)*a.dtype.itemsize)
            planned=set(np.rint(np.linspace(0,count-1+int(optical),min(count+int(optical),maximum-1))).astype(int))
            old=previous.get(name,{})
            kept={i:v for i,v in old.items() if i in planned}
            reduced,xe,ye=reduce_plane(a,xp)
            kept[idx]=(z,reduced,xe,ye)
            out[name]=kept
        return out

    def finish(self, collection, scientific, xp):
        from lcprop.core.backend import asnumpy
        out={};r=self.request
        for name,samples in (collection or {}).get('bounded_previews',{}).items():
            indices=sorted(samples);z=[samples[i][0] for i in indices]
            # Host export occurs only after backend reduction, with a strict bound.
            a=xp.stack([samples[i][1] for i in indices])
            if a.nbytes>FAST_MPR_TARGET_BYTES or max(a.shape[1:])>96:
                raise ValueError('preview transfer exceeds bound')
            a=asnumpy(a);xe,ye=samples[indices[0]][2:]
            x=asnumpy(collection['x']);y=asnumpy(collection['y'])
            if name=='electric_field_x_face':x=x+.5*r.grid.x_aperture_um/r.grid.Nx
            if name=='electric_field_y_face':y=y+.5*r.grid.y_aperture_um/r.grid.Ny
            n=scientific.completed_cells+int(name=='optical_intensity')
            out[name]=record(a,name=name,x=x,y=y,z=z,source_shape=(n,r.grid.Nx,r.grid.Ny),indices=indices,xe=xe,ye=ye)
        return out


def displayed_policy(result):
    """Result-owned provenance; never read the next-request selector."""
    policy=getattr(result,'result_policy',None) or getattr(result,'retention_summary',{}).get('policy','full')
    label={'minimal':'Minimal','fast':'Fast / Exploratory','interactive':'Interactive','full':'Full'}.get(policy,policy)
    if policy=='fast' and hasattr(result,'run') and not result.run.viewer_previews:
        label+=' (historical fixed-cut products)'
    return label
