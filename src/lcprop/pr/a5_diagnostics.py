"""Exact selected-product A5 diagnostics; never propagation or a preview estimate.

Working storage is O(Nx + Ny + number of carriers), not O(Nx*Ny): one row's
ellipse/histogram temporaries, axis marginals, row scalar sums and fixed bins.
The caller's exact plane is read-only input. No masked image is constructed.
Float64 row reductions can differ in rounding from the original full-plane
sum; definitions, support and thresholds are unchanged.
"""
from copy import deepcopy
import math

import numpy as np

from lcprop.products.data_model import DiagnosticData
from lcprop.pr.a5_artifacts import compact_artifacts
from lcprop.pr.far_field_mask import carrier_ellipses, ellipse_exclusion

EXACT_KEY = "far_field_intensity_exact"
ADVICE = "Analysis → Far Field Intensity is required (exact selected product; no complex endpoints required)."
FROZEN_ELLIPSE = {
    "channel_index": 0, "center_s": [-0.041597256936178405, -0.0],
    "covariance_s": [[4.397495647870098e-08, 0.0], [0.0, 4.405117963497312e-08]],
}
IDENTITIES = {
    "comparison_contract_sha256": "590a3cb0102158e40427b063a3aa12dcbaa493f4efbdbf22cd935d69b629fd9c",
    "historical_mask_source_sha256": "861220c4c244665035471580e9765d524100ace1d6ab69d1d6d288ec13175ab7",
    "artifact_source_sha256": "788ff423aa3f73b42dd8baefb6a01fb22d0121c04e0cd93b339891b0f13a5b23",
    "edge_source_sha256": "e36e6ce87363c89c68ad11430a4594932a609f1e85790a3b4817c868753dd196",
    "rectangle_source_sha256": "a96582d6feaf8637c7847c80c28551e7e9e919006b2e1d30734a8b5ac838c6dd",
    "historical_product_commit": "d9a6e436e1eae1deb15700a1024d8a6afee26919",
    "preserved_jobs": [4394756, 4414263],
    "identity_kind": "definition/source identity, not a generated-array hash",
}
QUALIFICATION = (
    "Sampled field norm, not physical mW. Complete FFT support; no fitting, peak "
    "normalization, recentering or support clipping of primary fractions. Applying "
    "this diagnostic does not make a calculation an A5 reproduction: historical "
    "L50/L2 streaming A5, entrance-only pinned experiment, and production canonical-V2 "
    "Static are distinct. Neither ellipse nor rectangle is published 81% efficiency. "
    "This does not validate sideways/backward experimental propagation."
)


def _exact_input(field, grid, launch):
    if field.key != EXACT_KEY or "preview_metadata" in field.coordinates:
        raise ValueError(ADVICE)
    if field.quantity != "direction_cosine_power_density":
        raise ValueError("Exact selected far-field normalization is unavailable")
    data = np.asarray(field.data)
    sx, sy = (np.asarray(field.coordinates[k]) for k in ("s_x", "s_y"))
    if (field.axes != ("s_x", "s_y") or sx.ndim != 1 or sy.ndim != 1
            or data.shape != (sx.size, sy.size) or min(data.shape) < 2
            or data.dtype.kind != "f"):
        raise ValueError("Exact canonical FFT coordinates and real intensity are required")
    if any(key in grid and grid[key] != size for key, size in zip(("Nx", "Ny"), data.shape)):
        raise ValueError("Exact intensity shape disagrees with retained grid")
    wavelengths = launch["wavelengths_um"]
    wavelength, n = float(wavelengths[0]), float(launch["refractive_index"])
    if (not wavelengths or not np.isfinite([wavelength, n]).all() or min(wavelength, n) <= 0
            or any(float(w) != wavelength for w in wavelengths)):
        raise ValueError("Single-wavelength exact direction-cosine normalization is required")
    for axis, key in ((sx, "dx_um"), (sy, "dy_um")):
        spacing = float(grid[key])
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Invalid retained grid spacing")
        expected = np.fft.fftshift(np.fft.fftfreq(axis.size, d=spacing)) * wavelength / n
        if not np.isfinite(axis).all() or not np.allclose(axis, expected, rtol=1e-12, atol=1e-15):
            raise ValueError("Exact coordinates disagree with retained FFT grid/normalization")
    # Row validation avoids a full-plane finite/comparison temporary.
    for row in data:
        if not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError("Finite nonnegative exact intensity is required")
    return data, sx, sy


def _primary(data, sx, sy, ellipses):
    totals, off = [], []
    count = 0
    for i, row in enumerate(data):
        excluded = ellipse_exclusion(sx[i:i+1], sy, ellipses)[0]
        count += int(excluded.sum())
        totals.append(float(np.sum(row, dtype=np.float64)))
        off.append(float(np.sum(row[~excluded], dtype=np.float64)))
    total, outside = math.fsum(totals), math.fsum(off)
    cell = float((sx[1]-sx[0]) * (sy[1]-sy[0]))
    if not np.isfinite([total, outside, total*cell, outside*cell]).all():
        raise ValueError("Far-field diagnostic reduction overflow")
    return {
        "status": "available", "carriers": deepcopy(ellipses),
        "mahalanobis_radius": 4.0, "excluded_boundary": "distance_squared <= 16",
        "excluded_pixel_count": count, "coordinate_cell_measure": cell,
        "surviving_norm": total*cell, "off_carrier_norm": outside*cell,
        "off_carrier_fraction": outside/total if total > 0 else None,
        "fraction_status": "available" if total > 0 else "unavailable",
        "fraction_unavailable_reason": None if total > 0 else "Zero surviving norm",
        "support": "complete sampled FFT grid", "qualification": QUALIFICATION,
    }


def _edge_occupancy(data):
    """a5_spectral_edge.py definition, with row-bounded marginal accumulation."""
    mx = np.empty(data.shape[0], dtype=np.float64)
    my = np.zeros(data.shape[1], dtype=np.float64)
    for i, row in enumerate(data):
        mx[i] = np.sum(row, dtype=np.float64)
        my += row
    total = float(np.sum(mx, dtype=np.float64))
    result = {}
    for band in (0.05, 0.10):
        axes = {}
        for marginal, label in ((mx, "x"), (my, "y")):
            frequencies = np.fft.fftshift(np.fft.fftfreq(marginal.size))
            selected = np.abs(frequencies) >= (1-band)*0.5
            power = float(np.sum(marginal[selected], dtype=np.float64))
            axes[label] = {"fraction": power/total if total else None,
                           "selected_bins": int(selected.sum()), "axis_bins": marginal.size}
        result[str(float(band))] = axes
    fractions = [v["fraction"] for axes in result.values() for v in axes.values()]
    return {"bands": result, "threshold": 0.001,
            "interpretation_blocker": any(v > .001 for v in fractions) if total else None,
            "qualification": "Frozen A5 interpretation threshold (>0.001 in any band/axis); not a convergence proof"}


def _historical_rectangle(data, sx, sy):
    """Frozen save_spectrum integer bins; only applicable on its original FFT grid.

    Preserve int-to-bin rounding, half-open rectangle and strict free-space
    diagnostic support. Row accumulation replaces only full-plane masks.
    """
    expected_x = np.fft.fftshift(np.fft.fftfreq(8192, d=1000/8192))*.633/2.4
    expected_y = np.fft.fftshift(np.fft.fftfreq(4096, d=1000/4096))*.633/2.4
    if (data.shape != (8192, 4096) or not np.allclose(sx, expected_x, rtol=1e-12, atol=1e-15)
            or not np.allclose(sy, expected_y, rtol=1e-12, atol=1e-15)):
        return {"status": "unavailable", "reason": "Frozen rectangle requires the A5 8192×4096, 1×1 mm, lambda=0.633 µm, n=2.4 FFT grid"}
    # Recovered authoritative bins: int(1000*(-sin(.1)/.633 ± .025))+Nx//2,
    # int(±1000*.025)+Ny//2. Not the analytic Gaussian ellipse.
    xlo, xhi, ylo, yhi = 3914, 3964, 2023, 2073
    totals, outside_sums, supported_sums, supported_outside = [], [], [], []
    for i, row in enumerate(data):
        outside = np.ones(sy.size, dtype=bool)
        if xlo <= i < xhi:
            outside[ylo:yhi] = False
        support = (sx[i]*sx[i] + sy*sy) < 1/2.4**2
        totals.append(float(np.sum(row, dtype=np.float64)))
        outside_sums.append(float(np.sum(row[outside], dtype=np.float64)))
        supported_sums.append(float(np.sum(row[support], dtype=np.float64)))
        supported_outside.append(float(np.sum(row[outside & support], dtype=np.float64)))
    total, outside, supported, clipped = map(math.fsum, (totals, outside_sums, supported_sums, supported_outside))
    return {"status": "available", "bins_half_open": [xlo, xhi, ylo, yhi],
            "historical_rectangle_fraction": outside/total if total else None,
            "support_clipped_rectangle_fraction": clipped/supported if supported else None,
            "rectangle_numerator": outside, "rectangle_denominator": total,
            "support_clipped_numerator": clipped, "support_clipped_denominator": supported,
            "support_clipped_definition": "strict sx²+sy² < 1/2.4²; both numerator and denominator clipped",
            "qualification": "DFT sums; distinct from ellipse; not published 81% identity"}


def exact_far_field_diagnostics(field, grid, launch):
    """Return distinct current and frozen diagnostics without changing the field."""
    data, sx, sy = _exact_input(field, grid, launch)
    try:
        current = _primary(data, sx, sy, carrier_ellipses(launch))
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        current = {"status": "unavailable", "reason": str(exc)}
    current["mask_mode"] = "current_launch"
    historical = _primary(data, sx, sy, [FROZEN_ELLIPSE])
    historical["mask_mode"] = "frozen_historical_A5"
    historical["provenance"] = deepcopy(IDENTITIES)
    historical["rectangle"] = _historical_rectangle(data, sx, sy)
    center = FROZEN_ELLIPSE["center_s"][0]
    historical["canonical_artifacts"] = compact_artifacts(data, sx, sy, center)
    historical["masked_artifacts"] = compact_artifacts(
        data, sx, sy, center,
        excluded_row=lambda i: ellipse_exclusion(sx[i:i+1], sy, [FROZEN_ELLIPSE])[0])
    historical["edge_occupancy"] = _edge_occupancy(data)
    current["provenance"] = {"geometry": "retained resolved launch; Q/k², full covariance, channel union",
                             "mask_definition_source_sha256": IDENTITIES["historical_mask_source_sha256"]}
    for value in (current, historical):
        value["input_field"] = EXACT_KEY
        value["normalization"] = "fftshift(fft2(A))*dx*dy; squared modulus times (n/lambda)^2"
        value["retained_grid"] = deepcopy(grid)
    return current, historical


def add_static_a5_diagnostics(fields, diagnostics, result):
    """Publish through the shared Diagnostics and Samples/Tables product APIs."""
    field = fields.get(EXACT_KEY)
    try:
        if field is None:
            raise ValueError(ADVICE)
        values = exact_far_field_diagnostics(field, result.grid_summary, result.launch_summary)
    except (KeyError, IndexError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        values = tuple({"status": "unavailable", "reason": str(exc), "required_product": ADVICE}
                       for _ in range(2))
    for mode, label, value in zip(("current_launch", "frozen_historical_A5"),
            ("Current-launch carrier exclusion", "Historical A5 comparison mask (frozen)"), values):
        value["mask_mode"] = mode
        # Tables expose each definition/value, including unavailable reasons.
        value["rows"] = [{"property": key, "value": item} for key, item in value.items()]
        key = "a5_" + mode
        diagnostics.add(key, DiagnosticData(key, label, value))
