"""Standalone, versioned in-memory codec; no GUI/transport/saved-state dispatch.

Explicit boundary/material coordinates are preserved byte-for-byte. This package
is a result product, not a continuation checkpoint. No pickle is accepted.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import io
import json
import math

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.farfield import DirectionCosineSpectrum
from lcprop.pr.local_plane_products import LocalPlaneProductResult, LocalPlaneProductSelection
from lcprop.pr.published_static_products import PUBLISHED_STATIC_PRODUCTS_SCHEMA, PublishedStaticProducts
from lcprop.pr.local_plane_workflow import LocalPlaneRunResult
from lcprop.pr.published_static import PR_PUBLISHED_STATIC_WORKFLOW, PR_PUBLISHED_STATIC_ARITHMETIC


def validate_published_products(result):
    s, p = result.scientific, result.products
    if s.workflow_identity != PR_PUBLISHED_STATIC_WORKFLOW or s.arithmetic_identity != PR_PUBLISHED_STATIC_ARITHMETIC:
        raise ValueError('wrong workflow/arithmetic identity')
    if p.metadata.get('schema') != PUBLISHED_STATIC_PRODUCTS_SCHEMA:
        raise ValueError('wrong published-order product schema')
    if s.status not in ('completed', 'cancelled', 'failed') or s.completed_cells < 0:
        raise ValueError('invalid scientific status/count')
    if len(s.ledger) != s.completed_cells:
        raise ValueError('accepted ledger/count mismatch')
    for a in (p.boundary_z_um, p.material_z_um, p.x_um, p.y_um):
        if a.dtype != np.float64 or a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0):
            raise ValueError('coordinates require explicit increasing binary64 arrays')
    nb, nc, nx, ny = len(p.boundary_z_um), len(p.material_z_um), len(p.x_um), len(p.y_um)
    if nx < 3 or ny < 1 or nc != s.completed_cells or nb not in ((0, 1) if nc == 0 else (nc+1,)):
        raise ValueError('boundary/material sample count mismatch')
    if nb and (p.boundary_z_um[0] != 0.0 or p.boundary_z_um[-1] != s.reached_z_um):
        raise ValueError('boundary endpoint mismatch')
    if nc:
        if not np.array_equal(p.material_z_um, p.boundary_z_um[1:]):
            raise ValueError('material samples must equal right endpoints')
        if (p.material_z_um.tolist() != [r['material_plane_um'] for r in s.ledger]
                or p.boundary_z_um[1:].tolist() != [r['z_end_um'] for r in s.ledger]):
            raise ValueError('coordinates disagree with accepted ledger')
        # Adjacent index*h starts and start+h endpoints need not round to the
        # same binary64 number. Preserve both recorded values; do not resnap.
        for boundary, record in zip(p.boundary_z_um[:-1], s.ledger):
            start = record['z_start_um']
            if not math.isfinite(start) or abs(start-boundary) > 2*max(math.ulp(start), math.ulp(float(boundary))):
                raise ValueError('cell start disagrees with preceding boundary')
    selected = p.metadata['selection']
    LocalPlaneProductSelection(**selected).validate()
    if set(selected) != {f.name for f in fields(LocalPlaneProductSelection)}:
        raise ValueError('incomplete product selection')
    nch = p.metadata['channel_count']
    real_dtype = np.dtype(p.metadata['real_dtype'])
    if type(nch) is not int or nch < 1 or real_dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('invalid channel/precision metadata')
    complex_dtype = np.dtype('complex64' if real_dtype == np.float32 else 'complex128')
    if s.launch_identity is not None and (s.launch_identity['shape'] != [nch, nx, ny]
            or s.launch_identity['dtype'] != str(complex_dtype)):
        raise ValueError('launch/product geometry mismatch')
    if nb and (selected['intensity_cuts'] or selected['preview']):
        if p.presentation_reference is None or not np.isfinite(p.presentation_reference) or p.presentation_reference <= 0:
            raise ValueError('missing presentation reference')
        if p.metadata['scientific_reference'] != s.launch_identity['peak_intensity_reference']:
            raise ValueError('scientific reference mismatch')
    allowed = {'optical', 'intensity', 'material', 'source', 'residual'}
    if set(p.cuts) != {k for k in allowed if selected[k+'_cuts']}:
        raise ValueError('cut selection mismatch')
    for name, pair in p.cuts.items():
        count = nb if name in ('optical', 'intensity') else nc
        prefix = (nch,) if name == 'optical' else ()
        if len(pair) != 2:
            raise ValueError('cuts require x-z and y-z pair')
        for a, size in zip(pair, (nx, ny)):
            if a.shape != (count,) + prefix + (size,) or a.dtype != (complex_dtype if name == 'optical' else real_dtype):
                raise ValueError('cut shape/type mismatch')
            if not np.all(np.isfinite(a)):
                raise ValueError('nonfinite cut')
    if (p.preview is not None) != (bool(nb) and selected['preview']):
        raise ValueError('preview selection mismatch')
    if p.preview is not None:
        for coordinate in (p.preview_x_um, p.preview_y_um):
            if (coordinate is None or coordinate.ndim != 1 or coordinate.dtype != np.float64
                    or not np.all(np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0)):
                raise ValueError('invalid preview coordinates')
        if (p.preview.dtype != np.float32 or p.preview.shape != (nb, len(p.preview_x_um), len(p.preview_y_um))
                or p.preview.shape[1]*p.preview.shape[2] > nx+ny or not np.all(np.isfinite(p.preview))):
            raise ValueError('invalid bounded preview')
    elif p.preview_x_um is not None or p.preview_y_um is not None:
        raise ValueError('preview coordinates without preview')
    if p.metadata['preview_coordinate_axis'] != 'boundary_z_um':
        raise ValueError('preview must use optical boundary coordinates')
    if s.boundary_field is not None:
        if (not selected['endpoint'] or s.boundary_field.shape != (nch, nx, ny)
                or s.boundary_field.dtype != complex_dtype):
            raise ValueError('endpoint selection/shape/type mismatch')
    if s.status != 'completed' and (s.far_field is not None or s.far_field_z_um is not None):
        raise ValueError('partial result cannot advertise completed far field')
    if s.status == 'completed' and (s.completed_cells != s.requested_cells or not nb):
        raise ValueError('completed result lacks complete coordinates')
    if s.far_field is not None:
        if (not selected['far_field'] or s.far_field_z_um != s.reached_z_um
                or s.far_field.intensity.shape != (nx, ny) or s.far_field.intensity.dtype != real_dtype
                or s.far_field.s_x.shape != (nx,) or s.far_field.s_y.shape != (ny,)):
            raise ValueError('far field endpoint/shape mismatch')
    elif s.far_field_z_um is not None or (s.status == 'completed' and selected['far_field']):
        raise ValueError('missing selected final far field')


def encode_published_products(result) -> bytes:
    validate_published_products(result)
    s, p = result.scientific, result.products
    arrays = {name: getattr(p, name) for name in
              ('boundary_z_um', 'material_z_um', 'x_um', 'y_um', 'preview', 'preview_x_um', 'preview_y_um')
              if getattr(p, name) is not None}
    for name, pair in p.cuts.items():
        arrays[name+'_xz'], arrays[name+'_yz'] = pair
    if s.boundary_field is not None:
        arrays['endpoint'] = asnumpy(s.boundary_field)
    if s.far_field is not None:
        for name in ('intensity', 's_x', 's_y'):
            arrays['far_field_'+name] = asnumpy(getattr(s.far_field, name))
    scientific = {f.name: getattr(s, f.name) for f in fields(s)
                  if f.name not in ('boundary_field', 'far_field')}
    header = dict(schema=PUBLISHED_STATIC_PRODUCTS_SCHEMA, scientific=scientific,
        metadata=p.metadata, presentation_reference=p.presentation_reference,
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                       sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k, v in arrays.items()})
    payload = io.BytesIO()
    np.savez(payload, header=np.frombuffer(json.dumps(header, allow_nan=False).encode(), dtype=np.uint8), **arrays)
    return payload.getvalue()


def decode_published_products(payload: bytes) -> LocalPlaneProductResult:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        header = json.loads(archive['header'].tobytes())
        if header['schema'] != PUBLISHED_STATIC_PRODUCTS_SCHEMA or set(archive.files) != set(header['arrays']) | {'header'}:
            raise ValueError('wrong package schema/array inventory')
        arrays = {}
        for name, identity in header['arrays'].items():
            a = archive[name]
            if (list(a.shape) != identity['shape'] or str(a.dtype) != identity['dtype']
                    or hashlib.sha256(a.tobytes(order='C')).hexdigest() != identity['sha256']):
                raise ValueError('product array identity mismatch')
            arrays[name] = a
    values = header['scientific']
    values['ledger'] = tuple(values['ledger'])
    values['boundary_field'] = arrays.pop('endpoint', None)
    values['far_field'] = None
    if 'far_field_intensity' in arrays:
        values['far_field'] = DirectionCosineSpectrum(**{name: arrays.pop('far_field_'+name)
            for name in ('intensity', 's_x', 's_y')})
    cuts = {name: (arrays.pop(name+'_xz'), arrays.pop(name+'_yz'))
            for name in ('optical', 'intensity', 'material', 'source', 'residual') if name+'_xz' in arrays}
    products = PublishedStaticProducts(**{name: arrays.pop(name, None) for name in
        ('boundary_z_um', 'material_z_um', 'x_um', 'y_um', 'preview', 'preview_x_um', 'preview_y_um')},
        cuts=cuts, presentation_reference=header['presentation_reference'], metadata=header['metadata'])
    if arrays:
        raise ValueError('unrecognized product arrays')
    result = LocalPlaneProductResult(LocalPlaneRunResult(**values), products)
    validate_published_products(result)
    return result
