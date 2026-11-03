"""Session-only spatial registry. No reconstruction, device transfer or retention."""
from dataclasses import replace
import numpy as np
from lcprop.products.data_model import FieldCollection


def spatial_coordinates(field, geometry):
    """Validate retained coordinates, using geometry only when lengths agree."""
    if field.axes != ('z', 'x', 'y') or getattr(field.data, 'ndim', None) != 3:
        raise ValueError('No retained z,x,y volume for this product.')
    if hasattr(field.data, '__cuda_array_interface__'):
        raise ValueError('Device fields require an explicit Product export before viewing.')
    if np.dtype(field.data.dtype).kind not in 'fiu':
        raise ValueError('Linked slices require a real scalar field.')
    result = {}
    for axis, size in zip(field.axes, field.data.shape):
        values = field.coordinates.get(axis, getattr(geometry, axis, None))
        if values is None:
            raise ValueError(f'Missing {axis} coordinates for linked slices.')
        values = np.asarray(values)
        if (values.shape != (size,) or not size or not np.isfinite(values).all()
                or (size > 1 and not np.all(np.diff(values) > 0))):
            raise ValueError(f'Invalid {axis} coordinates for linked slices.')
        result[axis] = values
    return result


def field_identity(field):
    # Source key distinguishes face/node/projected fields, even with equal labels.
    return (field.source_volume_key or field.key, field.quantity, field.value_unit,
            field.kind, field.content_revision)


def spatial_registry(run_data):
    available, unavailable = {}, {}
    for key, field in run_data.fields.items():
        if getattr(field.data, 'ndim', None) != 3:
            continue
        try:
            available[key] = spatial_coordinates(field, run_data.geometry)
        except ValueError as exc:
            unavailable[key] = str(exc)
    return available, unavailable


def independent_plane_label(field):
    """Only explicit plane coordinates authorize a z label; never guess an index."""
    if field.axes != ('x', 'y'):
        return field.display_name
    coordinates = field.coordinates
    name = field.display_name
    if field.key == 'optical_intensity_xy':
        name = 'Retained optical intensity plane'
    value = coordinates.get('selected_z_um')
    unit = 'µm'
    if value is None:
        value = coordinates.get('z')
        unit = field.units.get('z', 'um')
        unit = 'µm' if unit == 'um' else unit
    if value is not None:
        array = np.asarray(value)
        if array.size == 1 and np.isfinite(array).all():
            return f'{name} (z={float(array.reshape(-1)[0]):g} {unit})'
    return f'{name} (z unspecified)'


def with_transverse_aliases(run_data):
    """Borrow one plane; never add a persisted product or copy a volume."""
    available, _ = spatial_registry(run_data)
    fields = FieldCollection([(key, replace(field, display_name=independent_plane_label(field)))
                              if getattr(field.data, 'ndim', None) == 2 else (key, field)
                              for key, field in run_data.fields.items()])
    for key, coordinates in available.items():
        source = fields[key]
        # Existing endpoint planes stay independently selectable; generated aliases
        # alone track the arbitrary z slider. No source record is mutated.
        for k, f in list(fields.items()):
            if f.source_volume_key == key and f.axes == ('x', 'y'):
                fields.add(k, replace(f, source_volume_key=None))
        alias = '__linked_xy__:' + key
        fields.add(alias, replace(source, key=alias,
            data=source.data[source.data.shape[0] // 2], axes=('x', 'y'),
            source_volume_key=key,
            coordinates={**source.coordinates, **coordinates}, initially_selected=False))
    return replace(run_data, fields=fields)
