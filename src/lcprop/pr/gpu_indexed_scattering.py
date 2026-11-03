"""Opt-in GPU realization; never an alias for historical CPU phase bytes.

CuPy is imported only when constructing/using this experimental source.
One original-grid event is generated per call; no source-owned phase cache.
"""
import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from . import indexed_scattering as historical
from . import disk_scattering, frozen_scattering
from .disk_scattering import _read, file_sha256
from .frozen_scattering import scattering_context

ALGORITHM = 'cupy-xorwow-full-grid-gaussian-reflect-f64-v1'
RNG = 'cupy.random.RandomState; CURAND_RNG_PSEUDO_XORWOW; normal float64; full original C shape'
# Prototype guard, not a change to the physical domain or filter.
MAX_ORIGINAL_PLANE_BYTES = 1024**3


def _gpu():
    import cupy as cp
    from cupy.cuda import curand
    from cupyx.scipy.ndimage import gaussian_filter1d
    return cp, curand, gaussian_filter1d


def _implementation():
    return {name: file_sha256(path) for name, path in {
        'gpu_source': __file__, 'physical_recipe_validation': historical.__file__,
        'disk_validation': disk_scattering.__file__,
        'physical_context': frozen_scattering.__file__,
    }.items()}


def gpu_runtime_fingerprint():
    """Bind the selected device/runtime; not a cross-device determinism claim."""
    cp, curand, filter1d = _gpu()
    import importlib
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props['name']
    modules = [cp.random.RandomState.__module__, filter1d.__module__, curand.__name__]
    return dict(host=historical.runtime_fingerprint(), cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(), curand=curand.getVersion(),
        rng_method=int(curand.CURAND_RNG_PSEUDO_XORWOW),
        device_index=cp.cuda.runtime.getDevice(),
        device_name=name.decode() if isinstance(name, bytes) else name,
        compute_capability=[int(props['major']), int(props['minor'])],
        modules={m: file_sha256(importlib.import_module(m).__file__) for m in modules})


def _physical_view(recipe):
    """Reuse unchanged historical geometry validation, not its RNG/runtime."""
    m = copy.deepcopy(recipe)
    m['algorithm'] = historical._ALGORITHM
    m['generation']['rng'] = 'numpy.RandomState normal, original C order'
    return m


def _validate(m):
    if m.get('algorithm') != ALGORITHM or m['generation']['rng'] != RNG:
        raise ValueError('Not the explicitly selected GPU realization')
    historical._validate(_physical_view(m))
    if m.get('implementation') != _implementation():
        raise ValueError('GPU implementation identity changed')
    if m['algorithm_sha256'] != file_sha256(__file__):
        raise ValueError('GPU algorithm identity mismatch')
    nx, ny = m['generation']['original_shape']
    if nx * ny * 8 > MAX_ORIGINAL_PLANE_BYTES:
        raise ValueError('Original phase plane exceeds prototype allocation guard')
    if m.get('reproducibility') != 'qualified same-runtime/device only; distinct from historical CPU':
        raise ValueError('Invalid reproducibility scope')


def gpu_indexed_phase_descriptor(request, **kwargs):
    """Original physical request plus the CPU builder's geometry keywords.

    Requires an available GPU to bind the recipe. Does not generate noise.
    Explicit GPU use is required; no automatic CPU/GPU substitution.
    """
    if request.backend.backend != 'cupy':
        raise ValueError('GPU indexed recipe requires explicit CuPy backend')
    m = historical.indexed_phase_descriptor(request, **kwargs)
    m.update(algorithm=ALGORITHM, algorithm_sha256=file_sha256(__file__),
             runtime=gpu_runtime_fingerprint(), implementation=_implementation(),
             reproducibility='qualified same-runtime/device only; distinct from historical CPU')
    m['generation']['rng'] = RNG
    _validate(m)
    return m


def gpu_execution_seed_schedule(recipe):
    """Original physical seeds with inert refinement placeholders."""
    _validate(recipe)
    return historical.execution_seed_schedule(_physical_view(recipe))


class GPUIndexedScatteringPhases(historical.IndexedScatteringPhases):
    """Read-only recipe; phase(j) returns a caller-owned mutable GPU array.

    GPU arrays cannot promise NumPy's immutable-bytes contract. Returned planes
    are never retained by this source or reused on replay. No CPU fallback.
    """
    __slots__ = ()

    def __init__(self, descriptor_path, *, descriptor_sha256, source_identities):
        p = Path(descriptor_path)
        if p.stat().st_size > historical._MAX_DESCRIPTOR:
            raise ValueError('Oversized GPU descriptor')
        raw, _ = _read(p, p.stat().st_size)
        if hashlib.sha256(raw).hexdigest() != descriptor_sha256:
            raise ValueError('GPU descriptor SHA256 mismatch')
        try:
            m = json.loads(raw)
            _validate(m)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError('Malformed GPU recipe') from error
        if m['source_identities'] != source_identities:
            raise ValueError('GPU source identities mismatch')
        object.__setattr__(self, '_path', p.absolute())
        object.__setattr__(self, '_raw', raw)
        object.__setattr__(self, '_digest', descriptor_sha256)
        self._checked()

    def _checked(self):
        raw, _ = _read(self._path, len(self._raw))
        if raw != self._raw:
            raise ValueError('GPU recipe modified')
        m = json.loads(raw)
        _validate(m)
        if m['runtime'] != gpu_runtime_fingerprint():
            raise ValueError('GPU runtime/device differs; separate qualification required')
        return m

    def validate_for(self, request, *, required_cells):
        m = self._checked()
        if request.backend.backend != 'cupy' or request.backend.precision != 'float64':
            raise ValueError('GPU indexed source requires CuPy float64')
        if request.solver.partition_independent_scattering is not None:
            raise ValueError('Canonical and GPU indexed scattering conflict')
        expected = copy.deepcopy(m['context'])
        expected['grid']['dz_um'] = m['execution']['dz_um']
        expected['seeds_sha256'] = historical._digest(list(gpu_execution_seed_schedule(m)))
        if scattering_context(request) != expected:
            raise ValueError('GPU physical phase context mismatch')
        if isinstance(required_cells, bool) or not isinstance(required_cells, Integral) or not 1 <= required_cells <= len(m['execution']['cell_to_event']):
            raise ValueError('Incomplete GPU event coverage')

    def phase(self, event):
        m = self._checked()
        if isinstance(event, (bool, np.bool_)) or not isinstance(event, Integral) or not 0 <= event < m['shape'][0]:
            raise IndexError('Physical GPU event outside coverage')
        cp, curand, filter1d = _gpu()
        g = m['generation']
        # Explicit method, isolated state, one full-shape call: no global seed.
        rng = cp.random.RandomState(int(g['complete_seeds'][event]),
                                   method=curand.CURAND_RNG_PSEUDO_XORWOW)
        raw = rng.normal(0., g['raw_scale'], size=tuple(g['original_shape']), dtype=cp.float64)
        # Explicit original-boundary filtering before any crop; two work planes.
        work = cp.empty_like(raw)
        filter1d(raw, g['sigma_pixels'][0], axis=0, output=work,
                 order=0, mode='reflect', truncate=4.)
        filter1d(work, g['sigma_pixels'][1], axis=1, output=raw,
                 order=0, mode='reflect', truncate=4.)
        (x0, x1), (y0, y1) = g['crop']
        phase = raw[x0:x1, y0:y1].copy(order='C')
        phase *= g['post_filter_scale']
        if not bool(cp.isfinite(phase).all()):
            raise ValueError('Nonfinite GPU phase')
        cp.cuda.get_current_stream().synchronize()
        self._checked()
        return phase

    def apply_to(self, field, index, *, xp):
        m = self._checked()
        cp, _, _ = _gpu()
        if xp is not cp or not isinstance(field, cp.ndarray):
            raise ValueError('GPU phases require a CuPy field and backend')
        if field.device.id != cp.cuda.runtime.getDevice():
            raise ValueError('Field is on a different GPU device')
        if field.ndim != 3 or field.shape[0] < 1 or list(field.shape[-2:]) != m['shape'][1:] or field.dtype != cp.complex128:
            raise ValueError('GPU application requires compatible complex128 field')
        event = self.event_for_cell(index)
        if event is None:
            return
        phase = self.phase(event)
        field *= cp.exp(1j * phase)[None, :, :]
        cp.cuda.get_current_stream().synchronize()

    def provenance(self):
        return dict(mode='gpu_indexed_recipe', descriptor_sha256=self._digest,
                    descriptor=self._checked(), retained_phase_bytes=0,
                    historical_cpu_realization=False,
                    qualification='prototype; requires separate H200 qualification')
