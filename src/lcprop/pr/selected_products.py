"""Operational selected Static products, independent of scientific requests.

Intensity previews are contiguous block means of the *executed* fields.
Only bounded row blocks of intensity are formed when exact planes are absent.
The far-field FFT is necessarily full resolution, but its intensity is reduced
in blocks using the canonical transform and density normalization.
"""
from __future__ import annotations

import numpy as np
from lcprop.optics.splitstep import total_intensity
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr.visualization import _bin_edges, _block_centers
from lcprop.transport.result_policy import static_product_selection

DISPLAY_MAX_SIDE = 1024
# Historical reduced-Static optical catalog; unified material IDs belong to a
# different workflow and must not change this persisted availability mapping.
OPTICAL_PRODUCTS = ("input_intensity", "output_intensity", "far_field_intensity", "complex_input", "complex_output")


def display_shape(shape):
    scale = min(1.0, DISPLAY_MAX_SIDE / max(shape))
    return tuple(max(1, int(n * scale)) for n in shape)


def reduce_rows(shape, row_values, *, xp, asnumpy):
    """Return block means; callback creates at most one x-bin of intensity."""
    mx, my = display_shape(shape)
    xe, ye = _bin_edges(shape[0], mx), _bin_edges(shape[1], my)
    out = np.empty((mx, my), dtype=np.float64)
    for i, (start, stop) in enumerate(zip(xe[:-1], xe[1:])):
        # Splitting oversized bins also bounds intermediates on huge metadata grids.
        sums = xp.zeros(shape[1], dtype=xp.float64)
        rows = max(1, 262144 // shape[1])
        for a in range(int(start), int(stop), rows):
            b = min(a + rows, int(stop))
            sums += xp.sum(row_values(a, b), axis=0, dtype=xp.float64)
        # Transfer only a row summary, never an endpoint or intensity plane.
        host = np.asarray(asnumpy(sums))
        out[i] = np.add.reduceat(host, ye[:-1]) / ((stop - start) * np.diff(ye))
    return out, xe, ye


def preview_record(data, xe, ye, axes, coordinates, shape, *, far_field=False):
    quantity = "direction_cosine_power_density" if far_field else "normalized_intensity"
    unit = "normalized power / direction-cosine²" if far_field else "1/µm²"
    return {
        "data": np.asarray(data, dtype=np.float32),
        "metadata": {
            "visualization_only": True, "quantity": quantity, "value_unit": unit,
            "normalization": ("fftshift(fft2(A))*dx*dy; squared modulus times (n/lambda)^2"
                              if far_field else "coherence-group sum of squared field modulus"),
            "original_shape": list(shape), "preview_shape": list(data.shape),
            "axes": list(axes), "dtype": "float32",
            "reduction": "contiguous_block_arithmetic_mean",
            "block_bounds": {axes[0]: xe.tolist(), axes[1]: ye.tolist()},
            "coordinates": {axis: _block_centers(coord, edges).tolist()
                            for axis, coord, edges in zip(axes, coordinates, (xe, ye))},
            "original_extent": [float(coordinates[0][0]), float(coordinates[0][-1]),
                                float(coordinates[1][0]), float(coordinates[1][-1])],
        },
    }


def construct_selected_products(initial, final, *, policy, grid, launch, xp, asnumpy):
    kind, selection = static_product_selection(policy)
    if set(selection)-set(OPTICAL_PRODUCTS):
        raise ValueError('product is unavailable for legacy reduced Static')
    shape = initial.shape[-2:]
    coords = [(np.arange(n) - (n - 1) / 2) * float(grid[d])
              for n, d in zip(shape, ("dx_um", "dy_um"))]
    groups = tuple(launch["coherence_groups"])
    previews, exact = {}, {}
    for name, field in (("input_intensity", initial), ("output_intensity", final)):
        def rows(a, b):
            return total_intensity(field[:, a:b], coherence_groups=groups, xp=xp)
        if name in selection:
            value = total_intensity(field, coherence_groups=groups, xp=xp)
            exact[name] = np.asarray(asnumpy(value)).copy()
            reduced, xe, ye = reduce_rows(shape, lambda a, b: value[a:b], xp=xp, asnumpy=asnumpy)
            del value
        else:
            reduced, xe, ye = reduce_rows(shape, rows, xp=xp, asnumpy=asnumpy)
        previews[name] = preview_record(reduced, xe, ye, ("x", "y"), coords, shape)
    options = dict(dx_um=float(grid["dx_um"]), dy_um=float(grid["dy_um"]),
                   wavelength_um=float(launch["wavelengths_um"][0]),
                   refractive_index=float(launch["refractive_index"]),
                   coherence_groups=groups, xp=xp)
    if "far_field_intensity" in selection:
        spectrum = direction_cosine_spectrum(final, **options)
        exact["far_field_intensity"] = np.asarray(asnumpy(spectrum.intensity)).copy()
        reduced, xe, ye = reduce_rows(shape, lambda a, b: spectrum.intensity[a:b], xp=xp, asnumpy=asnumpy)
    else:
        def reducer(transformed, scale):
            return reduce_rows(shape, lambda a, b: xp.abs(transformed[a:b]) ** 2 * scale,
                               xp=xp, asnumpy=asnumpy)[0]
        spectrum = direction_cosine_spectrum(final, density_reducer=reducer, **options)
        reduced = spectrum.intensity
    angular = [np.asarray(asnumpy(spectrum.s_x)), np.asarray(asnumpy(spectrum.s_y))]
    previews["far_field_intensity"] = preview_record(reduced, xe, ye, ("s_x", "s_y"), angular, shape, far_field=True)
    return {"version": 1, "policy": kind, "selection": list(selection),
            "previews": previews, "exact": exact,
            "exact_coordinates": {"x": coords[0], "y": coords[1], "s_x": angular[0], "s_y": angular[1]},
            "availability": {name: ("selected" if name in selection else "not_selected")
                             for name in OPTICAL_PRODUCTS}}


def validate_selected_products(products, *, shape, policy):
    ANALYSIS_PRODUCTS = OPTICAL_PRODUCTS
    if products["version"] != 1 or products["policy"] != policy or policy not in ("interactive", "analysis"):
        raise ValueError("selected product contract version/policy mismatch")
    selection = products["selection"]
    if not isinstance(selection, list) or len(set(selection)) != len(selection) or not set(selection) <= set(ANALYSIS_PRODUCTS):
        raise ValueError("invalid exact selection")
    if policy == "interactive" and selection:
        raise ValueError("Interactive cannot include exact selections")
    names = {"input_intensity", "output_intensity", "far_field_intensity"}
    if set(products["previews"]) != names or set(products["exact"]) != set(selection) & names:
        raise ValueError("missing or unexpected products")
    if products["availability"] != {key: ("selected" if key in selection else "not_selected") for key in ANALYSIS_PRODUCTS}:
        raise ValueError("inconsistent availability")
    for key in names:
        record = products["previews"][key]
        value, meta = record["data"], record["metadata"]
        axes = ("s_x", "s_y") if key == "far_field_intensity" else ("x", "y")
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != display_shape(shape)
                or meta["visualization_only"] is not True or meta["original_shape"] != list(shape)
                or meta["preview_shape"] != list(value.shape) or meta["dtype"] != str(value.dtype)
                or meta["axes"] != list(axes) or meta["reduction"] != "contiguous_block_arithmetic_mean"):
            raise ValueError("invalid display preview")
        expected_quantity = "direction_cosine_power_density" if key == "far_field_intensity" else "normalized_intensity"
        expected_unit = "normalized power / direction-cosine²" if key == "far_field_intensity" else "1/µm²"
        expected_normalization = ("fftshift(fft2(A))*dx*dy; squared modulus times (n/lambda)^2"
                                  if key == "far_field_intensity" else "coherence-group sum of squared field modulus")
        if (meta["quantity"] != expected_quantity or meta["value_unit"] != expected_unit
                or meta["normalization"] != expected_normalization):
            raise ValueError("invalid preview quantity/normalization")
        coordinates = products["exact_coordinates"]
        extent = [float(coordinates[axes[0]][0]), float(coordinates[axes[0]][-1]),
                  float(coordinates[axes[1]][0]), float(coordinates[axes[1]][-1])]
        if meta["original_extent"] != extent:
            raise ValueError("invalid preview extent")
        for axis, n, m in zip(axes, shape, value.shape):
            edges = _bin_edges(n, m)
            original = np.asarray(products["exact_coordinates"][axis])
            if original.shape != (n,) or not np.all(np.isfinite(original)) or (n > 1 and not np.all(np.diff(original) > 0)):
                raise ValueError("invalid original coordinates")
            if meta["block_bounds"][axis] != edges.tolist() or meta["coordinates"][axis] != _block_centers(original, edges).tolist():
                raise ValueError("invalid preview coordinate mapping")
        if key in products["exact"]:
            exact = products["exact"][key]
            if not isinstance(exact, np.ndarray) or exact.shape != shape or exact.dtype.kind != "f":
                raise ValueError("invalid exact intensity")
    if not isinstance(products.get("carrier_power"), dict):
        raise ValueError("missing carrier diagnostic")


def add_selected_fields(fields, products):
    from lcprop.products.data_model import make_field
    for key, record in products["previews"].items():
        meta = record["metadata"]
        axes = tuple(meta["axes"])
        coords = {axis: np.asarray(meta["coordinates"][axis]) for axis in axes}
        coords["preview_metadata"] = meta
        kind = "far_field_intensity" if key == "far_field_intensity" else "intensity"
        units = {axis: "1" if axis.startswith("s_") else "um" for axis in axes}
        title = key.replace("_", " ").title()
        fields.add(key, make_field(key, title + " (Display Preview)", record["data"], axes, kind, units,
                                  quantity=meta["quantity"], value_unit=meta["value_unit"], coordinates=coords))
        if key in products["exact"]:
            fields.add(key + "_exact", make_field(key + "_exact", title + " (Exact)", products["exact"][key], axes, kind, units,
                       quantity=meta["quantity"], value_unit=meta["value_unit"],
                       coordinates={axis: products["exact_coordinates"][axis] for axis in axes}))
