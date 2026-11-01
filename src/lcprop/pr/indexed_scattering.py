"""Opt-in historical, indexed CPU regeneration with one-plane backend delivery.

A recipe is not a cross-platform promise of identical floating-point bytes.
No full-grid raw file, persistent phase file, phase cache or volume is allocated.
"""
from collections import deque
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import sys

import numpy as np
from .disk_scattering import _read, _validate_generation, file_sha256
from .frozen_scattering import scattering_context

_ALGORITHM = 'historical-randomstate-full-grid-rolling-halo-v1'
_MAX_DESCRIPTOR = 4 * 1024 * 1024
_MAX_HALO = 16 * 1024 * 1024


def runtime_fingerprint():
    """Recorded execution identity, not a bound on libm/filter roundoff."""
    import scipy
    from scipy.ndimage import _nd_image, _filters
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                machine=platform.machine(), system=platform.system(),
                libc=list(platform.libc_ver()), byteorder=sys.byteorder,
                numerical_binaries={k: file_sha256(v.__file__) for k, v in
                    [("randomstate", np.random.mtrand), ("ndimage", _nd_image), ("filter_weights", _filters)]})


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _hashes(values):
    return (isinstance(values, dict) and bool(values) and
            all(isinstance(k, str) and isinstance(h, str) and len(h) == 64 and
                all(c in '0123456789abcdef' for c in h) for k, h in values.items()))


def indexed_phase_descriptor(request, *, original_shape, original_aperture_um,
                             crop, events, source_identities, execution_dz_um=None):
    """Build a JSON-serializable recipe from the ORIGINAL longitudinal request.

    ``request.grid`` describes the crop but retains original L/dz/seeds. A
    refined execution request must use ``execution_seed_schedule(descriptor)``.
    Callers persist/hash the descriptor and retain its digest independently.
    Source identities are externally verified input/provenance digests.
    """
    if request.backend.precision != 'float64' or request.solver.partition_independent_scattering is not None:
        raise ValueError('Only historical float64 regeneration is supported')
    g = request.grid
    g.validate()
    if any(type(v) is not int or v < 1 for v in original_shape) or any(not math.isfinite(v) or v <= 0 for v in original_aperture_um):
        raise ValueError("Invalid original geometry")
    if not math.isfinite(request.solver.volume_noise_epsilon) or request.solver.volume_noise_epsilon <= 0 or not math.isfinite(request.solver.volume_noise_correlation_um) or request.solver.volume_noise_correlation_um <= 0:
        raise ValueError("Positive finite historical scattering required")
    nz = round(g.z_length_um / g.dz_um)
    seeds = request.solver.volume_noise_seeds
    if seeds is None or request.solver.volume_noise_seed is not None:
        raise ValueError('Complete explicit original seed list required')
    nx, ny = original_shape
    ax, ay = original_aperture_um
    sigma = [request.solver.volume_noise_correlation_um*nx/ax,
             request.solver.volume_noise_correlation_um*ny/ay]
    m = dict(schema=1, algorithm=_ALGORITHM, dtype='<f8', units='rad',
             shape=[events, g.Nx, g.Ny], context=scattering_context(request),
             source_identities=dict(source_identities),
             algorithm_sha256=file_sha256(__file__), runtime=runtime_fingerprint(),
             generation=dict(original_shape=list(original_shape), original_aperture_um=list(original_aperture_um),
                 crop=[list(c) for c in crop], full_Nz=nz, complete_seeds=list(seeds),
                 sigma_pixels=sigma, raw_scale=math.sqrt(request.solver.volume_noise_epsilon/nz*4*math.pi),
                 post_filter_scale=math.sqrt(sigma[0]*sigma[1]),
                 coordinate_convention='-aperture/2 + index*spacing',
                 filter='gaussian axis0 then axis1; reflect; order0; truncate4',
                 rng='numpy.RandomState normal, original C order'))
    dz = g.dz_um if execution_dz_um is None else execution_dz_um
    if not math.isfinite(dz) or dz <= 0:
        raise ValueError('Invalid execution spacing')
    ratio = Fraction(str(g.dz_um))/Fraction(str(dz))
    if ratio.denominator != 1 or ratio < 1:
        raise ValueError('Execution spacing must exactly divide original event spacing')
    factor = int(ratio)
    if type(events) is not int or not 1 <= events <= nz or events*factor > 100000:
        raise ValueError('Invalid/bounded physical-event coverage')
    m['event_positions_um'] = [float(Fraction(str(g.dz_um))*(j+1)) for j in range(events)]
    m['execution'] = dict(dz_um=float(dz), refinement_factor=factor,
        cell_to_event=[(i+1)//factor-1 if (i+1)%factor == 0 else None for i in range(events*factor)])
    _validate(m)
    return m


def execution_seed_schedule(descriptor):
    """Original event seeds with inert zero fillers for refined identity cells.

    Satisfies existing numerical-grid seed validation. Fillers are never used
    for RNG; normalization always uses descriptor.generation.full_Nz.
    """
    _validate(descriptor)
    f = descriptor['execution']['refinement_factor']
    return tuple(v for seed in descriptor['generation']['complete_seeds']
                 for v in ([0]*(f-1)+[seed]))


def _validate(m):
    if (m.get('schema') != 1 or m.get('algorithm') != _ALGORITHM or
        m.get('dtype') != '<f8' or m.get('units') != 'rad'):
        raise ValueError('Unsupported indexed recipe')
    shape = m['shape']
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('Invalid phase shape')
    if not _hashes(m['source_identities']):
        raise ValueError('Invalid source identities')
    _validate_generation(m)
    c = m['context'];g = m['generation'];e = m['execution']
    if any(not math.isfinite(g[k]) or g[k] <= 0 for k in ('raw_scale', 'post_filter_scale')):
        raise ValueError('Nonfinite/nonpositive generation scale')
    if c['precision'] != 'float64' or c['base_seed'] is not None:
        raise ValueError('Explicit float64 historical context required')
    factor = e['refinement_factor']
    if type(factor) is not int or not 1 <= factor <= 100000 or shape[0]*factor > 100000 or g['full_Nz']*factor > 100000:
        raise ValueError('Invalid refinement factor/coverage')
    dz = e['dz_um']
    if not math.isfinite(dz) or dz <= 0 or Fraction(str(dz))*factor != Fraction(str(c['grid']['dz_um'])):
        raise ValueError('Incompatible physical event spacing')
    expected = [(i+1)//factor-1 if (i+1)%factor == 0 else None for i in range(shape[0]*factor)]
    if e['cell_to_event'] != expected or any(v is not None and type(v) is not int for v in e['cell_to_event']):
        raise ValueError('Event mapping shifted/duplicated/rescaled')
    positions = [float(Fraction(str(c['grid']['dz_um']))*(j+1)) for j in range(shape[0])]
    if m['event_positions_um'] != positions or any(type(v) not in (int, float) for v in m['event_positions_um']):
        raise ValueError('Physical endpoint placement mismatch')
    for sigma in g['sigma_pixels']:
        if not math.isfinite(sigma) or sigma < 1e-15 or (2*int(4*sigma+.5)+1)*8 > _MAX_HALO:
            raise ValueError('Unsupported/bounded Gaussian filter weights')
    radius = int(4*g['sigma_pixels'][0]+.5)
    if min(g['original_shape'][0], 2*radius+1)*g['original_shape'][1]*8 > _MAX_HALO:
        raise ValueError('Filter halo exceeds bounded workspace')


def _generate(m, event):
    """Preserve full-grid sample positions; retain only the complete x halo."""
    from scipy.ndimage import gaussian_filter1d
    g = m['generation'];nx, ny = g['original_shape']
    (x0, x1), (y0, y1) = g['crop'];sx, sy = g['sigma_pixels']
    radius = int(4*sx+.5)
    rng = np.random.RandomState(g['complete_seeds'][event])
    rows = deque();next_row = 0;first_row = 0
    output = bytearray(m['shape'][1]*m['shape'][2]*8)
    offset = 0
    for x in range(x0, x1):
        lo, hi = max(0, x-radius), min(nx, x+radius+1)
        while next_row < hi:
            row = rng.normal(0, g['raw_scale'], size=ny).astype('<f8')
            if next_row >= lo:
                rows.append(row)
            next_row += 1
        while first_row < lo and rows and len(rows) > hi-lo:
            rows.popleft();first_row += 1
        first_row = lo
        block = np.stack(rows)
        if block.shape != (hi-lo, ny):
            raise ValueError('Incomplete historical halo')
        filtered = gaussian_filter1d(block, sx, axis=0, mode='reflect', order=0, truncate=4.)
        final = gaussian_filter1d(filtered[x-lo], sy, axis=0, mode='reflect', order=0, truncate=4.)
        values = (final[y0:y1]*g['post_filter_scale']).astype('<f8')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite generated phase')
        data = values.tobytes()
        memoryview(output)[offset:offset+len(data)] = data
        offset += len(data)
    return np.frombuffer(bytes(output), dtype='<f8').reshape(m['shape'][1:])


class IndexedScatteringPhases:
    """Immutable recipe; phase(j) addresses a physical event, not a cell.

    apply_to(field, cell, xp=...) uses the descriptor's explicit physical map.
    Reads regenerate only the selected event, retaining no phase cache. The
    descriptor's runtime fingerprint must match; cross-runtime use is an
    explicit, separately qualified new recipe, never a silent substitution.
    """
    __slots__ = ('_path', '_raw', '_digest')

    def __setattr__(self, name, value):
        raise AttributeError('IndexedScatteringPhases is immutable')

    def __init__(self, descriptor_path, *, descriptor_sha256, source_identities):
        p = Path(descriptor_path)
        if p.stat().st_size > _MAX_DESCRIPTOR:
            raise ValueError('Oversized generation descriptor')
        raw, _ = _read(p, p.stat().st_size)
        if hashlib.sha256(raw).hexdigest() != descriptor_sha256:
            raise ValueError('Descriptor SHA256 mismatch')
        try:
            m = json.loads(raw);_validate(m)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError('Malformed generation descriptor') from error
        if m['source_identities'] != source_identities:
            raise ValueError('Source identities mismatch')
        object.__setattr__(self, '_path', p.absolute())
        object.__setattr__(self, '_raw', raw)
        object.__setattr__(self, '_digest', descriptor_sha256)
        self._checked()

    def _checked(self):
        raw, _ = _read(self._path, len(self._raw))
        m = json.loads(raw)
        if raw != self._raw or m['algorithm_sha256'] != file_sha256(__file__):
            raise ValueError('Recipe/algorithm identity changed')
        if m['runtime'] != runtime_fingerprint():
            raise ValueError('Regeneration runtime differs; separate qualification required')
        return m

    def phase(self, event):
        m = self._checked()
        if isinstance(event, (bool, np.bool_)) or not isinstance(event, Integral) or not 0 <= event < m['shape'][0]:
            raise IndexError('Physical event outside coverage')
        phase = _generate(m, event)
        self._checked()
        return phase

    def event_for_cell(self, cell):
        m = self._checked();mapping = m['execution']['cell_to_event']
        if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, Integral) or not 0 <= cell < len(mapping):
            raise IndexError('Numerical cell outside coverage')
        return mapping[cell]

    def validate_for(self, request, *, required_cells):
        m = self._checked()
        if request.backend.backend not in ('numpy', 'cupy') or request.backend.precision != 'float64':
            raise ValueError('Indexed phases require float64 NumPy/CuPy')
        if request.solver.partition_independent_scattering is not None:
            raise ValueError('Canonical and historical scattering conflict')
        expected = json.loads(json.dumps(m['context']))
        expected['grid']['dz_um'] = m['execution']['dz_um']
        expected['seeds_sha256'] = _digest(list(execution_seed_schedule(m)))
        if scattering_context(request) != expected:
            raise ValueError('Execution geometry/physical seed context mismatch')
        if isinstance(required_cells, bool) or not isinstance(required_cells, Integral) or not 1 <= required_cells <= len(m['execution']['cell_to_event']):
            raise ValueError('Incomplete event coverage')

    def apply_to(self, field, index, *, xp):
        m = self._checked()
        if field.ndim != 3 or field.shape[0] < 1 or list(field.shape[-2:]) != m['shape'][1:] or np.dtype(field.dtype).str != '<c16':
            raise ValueError('Indexed application requires compatible complex128 field')
        event = self.event_for_cell(index)
        if event is None:
            return
        host = self.phase(event)
        phase = xp.asarray(host)
        field *= xp.exp(1j*phase)[None, :, :]
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()

    def provenance(self):
        return dict(mode='deterministic_indexed_recipe', descriptor_sha256=self._digest,
                    descriptor=self._checked(), retained_phase_bytes=0,
                    cross_runtime_numerical_qualification='not implied by recipe identity')


def compare_regenerated_phase(reference, candidate):
    """Report cross-runtime differences without inventing an acceptance budget.

    Caller must first establish identical recipe/scientific inputs. Equality
    is an observation for these arrays, never a general runtime guarantee.
    Nonzero differences require separately justified numerical qualification.
    """
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or a.dtype.str != '<f8' or b.dtype.str != '<f8':
        raise ValueError('Incompatible phase comparison')
    maximum = square = norm = 0.
    equal = True
    for x, y in zip(a, b):
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite phase comparison')
        d = x-y
        maximum = max(maximum, float(np.max(abs(d))))
        square += float(np.dot(d, d));norm += float(np.dot(x, x))
        equal = equal and x.tobytes() == y.tobytes()
    if not all(math.isfinite(v) for v in (maximum, square, norm)):
        raise ValueError('Nonfinite comparison metrics')
    return dict(bitwise=equal, max_abs=maximum,
                relative_l2=math.sqrt(square/norm) if norm else (0. if not square else None),
                qualification='observed_equal' if equal else 'requires_independent_error_budget',
                tolerance_applied=None)
