"""Hash-bound disk phase inputs; no RNG or filtering in the propagation path."""
import hashlib
import json
import math
import os
from numbers import Integral
from pathlib import Path
import numpy as np
from .frozen_scattering import scattering_context


# Explicit capacity, independent of filesystem st_blksize. No double chunk
# allocation in hashing; phase bytes remain immutable and hash-verified.
_IO_BUFFER_BYTES = 64 * 1024


def file_sha256(path):
    h = hashlib.sha256()
    buffer = bytearray(_IO_BUFFER_BYTES)
    view = memoryview(buffer)
    with open(path, 'rb', buffering=0) as f:
        while True:
            count = f.readinto(buffer)
            if not count:
                break
            h.update(view[:count])
    return h.hexdigest()


def _read(path, size):
    # No symlink substitution; a verified immutable bytes snapshot is consumed.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb', buffering=_IO_BUFFER_BYTES) as f:
        before = os.fstat(f.fileno())
        if before.st_size != size:
            raise ValueError('Phase file size changed')
        data = f.read(size)
        after = os.fstat(f.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(data) != size or signature(before) != signature(after):
        raise ValueError('Phase file changed during read')
    return data, signature(after)


class DiskScatteringPhases:
    """One-plane reader. Construct from an externally trusted manifest digest.

    Source identities must also be independently supplied as {name: SHA256}.
    The manifest binds original geometry, seed order, filtering and physical z.
    File metadata and bytes are checked before every use, including replay.
    No phase cache, backend import, or mutable array is retained.
    """
    __slots__ = ('_root', '_raw', '_digest', '_signatures')

    def __setattr__(self, name, value):
        raise AttributeError('DiskScatteringPhases is immutable')

    def __init__(self, manifest_path, *, manifest_sha256, source_identities):
        p = Path(manifest_path)
        if p.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Oversized manifest')
        raw, _ = _read(p, p.stat().st_size)
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('Manifest SHA256 mismatch')
        m = json.loads(raw)
        if any(m.get(k) != v for k,v in dict(schema=2, dtype='<f8', units='rad', layout='zxy').items()):
            raise ValueError('Unsupported disk phase manifest')
        shape = m['shape']
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('Invalid shape')
        if not source_identities or m['source_identities'] != source_identities:
            raise ValueError('Source identities mismatch')
        if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in source_identities.values()):
            raise ValueError('Invalid source digest')
        if len(m['planes']) != shape[0] or not m.get('generation'):
            raise ValueError('Incomplete manifest')
        _validate_generation(m)
        dz = m['context']['grid']['dz_um']
        if not np.isfinite(dz) or dz <= 0:
            raise ValueError('Invalid longitudinal spacing')
        for i, entry in enumerate(m['planes']):
            if (type(entry['index']) is not int or entry['index'] != i or
                entry['file'] != f'phase-{i:06d}.f64' or entry['z_um'] != (i+1)*dz):
                raise ValueError('Phase order/physical indexing mismatch')
        object.__setattr__(self, '_root', p.absolute())
        object.__setattr__(self, '_raw', raw)
        object.__setattr__(self, '_digest', manifest_sha256)
        signatures = []
        for i in range(shape[0]):
            plane, sig = self._load(i)
            signatures.append(sig)
            del plane
        object.__setattr__(self, '_signatures', tuple(signatures))

    def _load(self, index):
        raw, _ = _read(self._root, len(self._raw))
        if raw != self._raw:
            raise ValueError('Manifest modified')
        m = json.loads(raw)
        e = m['planes'][index]
        data, signature = _read(self._root.parent / e['file'], m['shape'][1]*m['shape'][2]*8)
        if hashlib.sha256(data).hexdigest() != e['sha256']:
            raise ValueError('Phase SHA256 mismatch')
        a = np.frombuffer(data, dtype='<f8').reshape(m['shape'][1:])
        # Row-wise check avoids an additional plane-sized boolean allocation.
        if any(not np.isfinite(row).all() for row in a):
            raise ValueError('Nonfinite phase')
        return a, signature

    def phase(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index < len(self._signatures):
            raise IndexError('Phase index outside coverage')
        a, sig = self._load(index)
        if sig != self._signatures[index]:
            raise ValueError('Phase file modified/replaced')
        return a

    def validate_for(self, request, *, required_cells):
        m = json.loads(self._raw)
        if request.backend.backend not in ('numpy', 'cupy') or request.backend.precision != 'float64':
            raise ValueError('Disk phases require float64 numpy/cupy')
        if request.solver.partition_independent_scattering is not None:
            raise ValueError('Canonical scattering conflicts with external phases')
        if m['context'] != scattering_context(request) or m['shape'][1:] != [request.grid.Nx, request.grid.Ny]:
            raise ValueError('Physical phase context mismatch')
        if isinstance(required_cells, bool) or not isinstance(required_cells, Integral) or not 1 <= required_cells <= len(self._signatures):
            raise ValueError('Incomplete phase coverage')
        for i in range(len(self._signatures)):
            a = self.phase(i)
            del a

    def apply_to(self, field, index, *, xp):
        """Apply at the existing scattering step; release buffers before return."""
        host = self.phase(index)
        phase = xp.asarray(host)
        field *= xp.exp(1j * phase)[None, :, :]
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()

    def provenance(self):
        return dict(mode='disk_frozen_phase_input', manifest_sha256=self._digest,
                    manifest=json.loads(self._raw), retained_phase_bytes=0)


def _validate_generation(m):
    """Validate physical crop and complete original seed sequence independently."""
    gen = m['generation']
    c = m['context']
    grid = c['grid']
    nx, ny = gen['original_shape']
    ax, ay = gen['original_aperture_um']
    (x0,x1),(y0,y1) = gen['crop']
    if (any(type(v) is not int for v in (nx,ny,x0,x1,y0,y1)) or
        not 0 <= x0 < x1 <= nx or not 0 <= y0 < y1 <= ny or
        not all(math.isfinite(v) and v > 0 for v in (ax,ay))):
        raise ValueError('Invalid original coordinates')
    if (m['shape'][1:] != [x1-x0,y1-y0] or
        [grid['Nx'],grid['Ny']] != m['shape'][1:] or
        not np.allclose([grid['x_aperture_um']/grid['Nx'],grid['y_aperture_um']/grid['Ny']],
                        [ax/nx,ay/ny], rtol=0,atol=1e-15)):
        raise ValueError('Original sampling/crop mismatch')
    nz = gen['full_Nz']
    seeds = gen['complete_seeds']
    if (type(nz) is not int or nz < 1 or len(seeds) != nz or
        any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or
        grid['z_length_um'] != nz*grid['dz_um'] or m['shape'][0] > nz or
        hashlib.sha256(json.dumps(seeds,separators=(',', ':')).encode()).hexdigest() != c['seeds_sha256']):
        raise ValueError('Original longitudinal grid/seed identity mismatch')
    eps, corr = c['epsilon'], c['correlation_um']
    if not all(math.isfinite(v) and v > 0 for v in (eps,corr)):
        raise ValueError('Invalid scattering normalization')
    sigma = [corr*nx/ax,corr*ny/ay]
    if (gen['sigma_pixels'] != sigma or gen['raw_scale'] != math.sqrt(eps/nz*4*math.pi) or
        gen['post_filter_scale'] != math.sqrt(sigma[0]*sigma[1]) or
        gen['coordinate_convention'] != '-aperture/2 + index*spacing' or
        gen['filter'] != 'gaussian axis0 then axis1; reflect; order0; truncate4' or
        gen['rng'] != 'numpy.RandomState normal, original C order'):
        raise ValueError('Original filter/normalization mismatch')
