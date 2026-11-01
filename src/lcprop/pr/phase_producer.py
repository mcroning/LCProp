"""Construction-only historical full-grid phases, streamed through disk.

Never used by the scientific march. RandomState ordering, SciPy reflect/truncate
4 separable filtering, and full longitudinal normalization are preserved.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .frozen_scattering import scattering_context
from .disk_scattering import file_sha256, _IO_BUFFER_BYTES


def produce_disk_phases(directory, *, request, original_shape, original_aperture_um,
                        crop, cells, source_files):
    """Freeze a prefix at original spacing; return manifest path and SHA256.

    ``crop=((x_start,x_stop),(y_start,y_stop))`` uses original endpoint-excluded
    coordinates -aperture/2 + index*spacing. ``source_files`` maps provenance
    names to (path, independently expected SHA256). Complete seed list required.
    The destination must not exist. A failed build has no finalized manifest.
    """
    if request.backend.precision != 'float64' or request.solver.partition_independent_scattering is not None:
        raise ValueError('Only historical float64 noise is supported')
    nx, ny = original_shape
    if any(type(n) is not int or n < 1 for n in (nx,ny)):
        raise ValueError('Invalid original shape')
    ax, ay = original_aperture_um
    if any(not math.isfinite(a) or a <= 0 for a in (ax,ay)):
        raise ValueError('Invalid original aperture')
    (x0,x1),(y0,y1) = crop
    if (any(type(n) is not int for n in (x0,x1,y0,y1)) or
        not 0 <= x0 < x1 <= nx or not 0 <= y0 < y1 <= ny):
        raise ValueError('Invalid original-index crop')
    g = request.grid
    if (g.Nx,g.Ny) != (x1-x0,y1-y0) or not np.allclose(
        [g.x_aperture_um/g.Nx,g.y_aperture_um/g.Ny], [ax/nx,ay/ny], rtol=0,atol=1e-15):
        raise ValueError('Crop must preserve original sampling')
    nz = round(g.z_length_um/g.dz_um)
    if nz < 1 or g.z_length_um != nz*g.dz_um:
        raise ValueError('Nonintegral longitudinal grid')
    seeds = request.solver.volume_noise_seeds
    if (seeds is None or len(seeds) != nz or
        any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Complete original seed sequence required')
    if type(cells) is not int or not 1 <= cells <= nz:
        raise ValueError('Invalid prefix')
    eps = request.solver.volume_noise_epsilon
    corr = request.solver.volume_noise_correlation_um
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(corr) or corr <= 0:
        raise ValueError('Positive finite historical scattering required')
    identities = {}
    if not source_files:
        raise ValueError('Source identities required')
    for name,(path,expected) in source_files.items():
        if file_sha256(path) != expected:
            raise ValueError('Source identity mismatch')
        identities[name] = expected
    sigma = (corr*nx/ax,corr*ny/ay)
    radius = int(4*sigma[0]+.5)
    if min(nx,2*radius+1)*ny*8 > 16*1024*1024:
        raise ValueError("Filter halo exceeds bounded 16 MiB workspace; no allocation attempted")
    scale = math.sqrt(eps/nz*4*math.pi)
    factor = math.sqrt(sigma[0]*sigma[1])
    dest = Path(directory)
    dest.mkdir(parents=True,exist_ok=False)
    entries = []
    # One raw full-grid file, not a memory map; Python holds only halo rows.
    with tempfile.TemporaryFile(dir=dest, buffering=_IO_BUFFER_BYTES) as raw:
        for index in range(cells):
            rng = np.random.RandomState(seeds[index])
            raw.seek(0)
            raw.truncate()
            for _ in range(nx):
                row = rng.normal(0,scale,size=ny).astype('<f8')
                raw.write(row.tobytes())
            raw.flush()
            filename = f'phase-{index:06d}.f64'
            h = hashlib.sha256()
            with (dest/filename).open('xb', buffering=_IO_BUFFER_BYTES) as output:
                for x in range(x0,x1):
                    lo,hi = max(0,x-radius),min(nx,x+radius+1)
                    raw.seek(lo*ny*8)
                    block = np.frombuffer(raw.read((hi-lo)*ny*8),dtype='<f8').reshape(hi-lo,ny)
                    # Interior stencil has full halo. At global edges reflect
                    # operates at the original boundary, never the crop edge.
                    first = gaussian_filter1d(block,sigma[0],axis=0,mode='reflect',truncate=4.0)
                    final = gaussian_filter1d(first[x-lo],sigma[1],axis=0,mode='reflect',truncate=4.0)
                    data = (final[y0:y1]*factor).astype('<f8').tobytes()
                    output.write(data)
                    h.update(data)
                    del block, first, final, data
            (dest/filename).chmod(0o444)
            entries.append(dict(index=index,z_um=(index+1)*g.dz_um,file=filename,sha256=h.hexdigest()))
    for path,expected in source_files.values():
        if file_sha256(path) != expected:
            raise ValueError("Source changed during production; manifest not finalized")
    import scipy
    manifest = dict(schema=2,units='rad',layout='zxy',dtype='<f8',shape=[cells,g.Nx,g.Ny],
        context=scattering_context(request),source_identities=identities,planes=entries,
        generation=dict(original_shape=list(original_shape),original_aperture_um=list(original_aperture_um),
            crop=[list(c) for c in crop],coordinate_convention='-aperture/2 + index*spacing',
            complete_seeds=list(seeds),full_Nz=nz,sigma_pixels=list(sigma),raw_scale=scale,
            post_filter_scale=factor,filter='gaussian axis0 then axis1; reflect; order0; truncate4',
            rng='numpy.RandomState normal, original C order',numpy=np.__version__,scipy=scipy.__version__))
    mp = dest/'manifest.json'
    mp.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    mp.chmod(0o444)
    return mp,file_sha256(mp)
