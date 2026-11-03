"""Small immutable external phase input for bounded float64 CPU experiments.

The NPZ contains final phase increments, not samples to regenerate/filter.
This is deliberately not a general cache or full-resolution storage API.
"""
from dataclasses import asdict
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import zipfile
import numpy as np

_MAX_BYTES = 64 * 1024 * 1024

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def scattering_context(request):
    """Return the exact physical/generation context bound by a phase manifest."""
    s = request.solver
    seeds = s.volume_noise_seeds
    return dict(grid=asdict(request.grid), epsilon=float(s.volume_noise_epsilon),
                correlation_um=float(s.volume_noise_correlation_um),
                base_seed=s.volume_noise_seed,
                seeds_sha256=None if seeds is None else _sha(json.dumps(list(seeds), separators=(',', ':')).encode()),
                precision=request.backend.precision)

class FrozenScatteringPhases:
    """Read-only, indexed, hash-bound phase planes loaded once before execution.

    Construct only with ``from_npz``. Manifest schema 1 requires units=rad,
    layout=zxy, dtype=<f8, shape, contiguous zero-based indices, payload_sha256,
    context from ``scattering_context``, and nonempty producer/coordinates text.
    The manifest SHA-256 must come from a trusted external record. Both first
    pass and replay use immutable in-memory bytes, not a mutable file or iterator.
    At most 64 MiB of serialized/uncompressed input is supported.
    """
    __slots__ = ('_planes', '_manifest', '_manifest_hash')

    def __init__(self):
        raise TypeError('Use FrozenScatteringPhases.from_npz')

    def __setattr__(self, name, value):
        raise AttributeError('FrozenScatteringPhases is immutable')

    @classmethod
    def from_npz(cls, path, *, manifest_path, manifest_sha256):
        mp = Path(manifest_path)
        if mp.stat().st_size > 65536:
            raise ValueError('Phase manifest exceeds size limit')
        raw_manifest = mp.read_bytes()
        if _sha(raw_manifest) != manifest_sha256:
            raise ValueError('Phase manifest SHA-256 mismatch')
        m = json.loads(raw_manifest)
        if any(m.get(k) != v for k, v in dict(schema=1, units='rad', layout='zxy', dtype='<f8').items()):
            raise ValueError('Unsupported phase manifest')
        shape = m.get('shape')
        if (not isinstance(shape, list) or len(shape) != 3 or
            any(type(n) is not int or n < 1 for n in shape) or
            shape[0]*shape[1]*shape[2]*8 > _MAX_BYTES):
            raise ValueError('Invalid/bounded phase shape')
        indices = m.get('indices')
        if (indices != list(range(shape[0])) or any(type(i) is not int for i in indices)):
            raise ValueError('Phase indices must be contiguous from zero')
        if not isinstance(m.get('context'), dict) or any(not isinstance(m.get(k), str) or not m[k] for k in ('producer','coordinates')):
            raise ValueError('Missing phase context/provenance')
        p = Path(path)
        if p.stat().st_size > _MAX_BYTES:
            raise ValueError('Phase input exceeds size limit')
        payload = p.read_bytes()
        if _sha(payload) != m.get('payload_sha256'):
            raise ValueError('Phase payload SHA-256 mismatch')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            members = z.infolist()
            if len(members) != 1 or members[0].filename != 'phase.npy' or members[0].file_size > _MAX_BYTES:
                raise ValueError('Phase NPZ must contain only bounded phase.npy')
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            a = z['phase']
            if list(a.shape) != shape or a.dtype.str != '<f8' or not np.isfinite(a).all():
                raise ValueError('Invalid phase shape/dtype/nonfinite values')
            planes = tuple(a[i].tobytes(order='C') for i in range(shape[0]))
        obj = object.__new__(cls)
        object.__setattr__(obj, '_planes', planes)
        object.__setattr__(obj, '_manifest', raw_manifest)
        object.__setattr__(obj, '_manifest_hash', manifest_sha256)
        return obj

    def validate_for(self, request, *, required_cells):
        """Validate the complete execution coverage before any march begins."""
        m = json.loads(self._manifest)
        if request.backend.backend != 'numpy' or request.backend.precision != 'float64':
            raise ValueError('Frozen phases currently require float64 NumPy CPU')
        if request.solver.partition_independent_scattering is not None:
            raise ValueError('Frozen legacy phases conflict with canonical scattering')
        if m['context'] != scattering_context(request):
            raise ValueError('Frozen phase physical/seed context mismatch')
        if m['shape'][1:] != [request.grid.Nx, request.grid.Ny]:
            raise ValueError('Frozen phase transverse grid mismatch')
        if (isinstance(required_cells, bool) or not isinstance(required_cells, Integral)
                or not 1 <= required_cells <= len(self._planes)):
            raise ValueError('Frozen phase coverage is incomplete')

    def phase(self, z_index):
        if isinstance(z_index, (bool, np.bool_)) or not isinstance(z_index, Integral) or not 0 <= z_index < len(self._planes):
            raise IndexError('Frozen phase index outside coverage')
        shape = json.loads(self._manifest)['shape'][1:]
        return np.frombuffer(self._planes[z_index], dtype='<f8').reshape(shape)

    def provenance(self):
        m = json.loads(self._manifest)
        return dict(mode='frozen_phase_input', manifest_sha256=self._manifest_hash,
                    payload_sha256=m['payload_sha256'], manifest=m,
                    retained_phase_bytes=sum(map(len,self._planes)))
