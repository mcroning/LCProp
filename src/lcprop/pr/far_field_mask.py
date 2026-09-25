"""Completed-result carrier exclusions; never an optical screen or power estimate."""
from dataclasses import replace
import math

import numpy as np

from lcprop.products.data_model import DiagnosticData

MASK_SIGMAS = 4.0
MASK_KEY = "far_field_zero_order_masked"


def carrier_ellipses(launch, *, sigmas=MASK_SIGMAS):
    """Use retained resolver geometry, also used by launch-sampling qualification.

    For entrance amplitude exp(-r.T Q r), intensity in k has covariance Q.
    Each resolved channel is enabled: disabled editor beams never enter launch.
    """
    if not math.isfinite(sigmas) or sigmas <= 0:
        raise ValueError("mask extent must be finite and positive")
    beams = launch.get("resolved_beams")
    if not beams:
        raise ValueError("retained resolved Gaussian launch geometry is unavailable")
    if len(beams) != int(launch.get("Nch", len(beams))):
        raise ValueError("resolved launch geometry does not cover every channel")
    ellipses = []
    for index, beam in enumerate(beams):
        kx, ky, kz = (float(beam[key]) for key in (
            "kx_rad_per_um", "ky_rad_per_um", "kz_internal_rad_per_um"))
        k = math.hypot(kx, ky, kz)
        q = np.asarray(beam["interface_quadratic_per_um2"], dtype=float)
        if (not math.isfinite(k) or k <= 0 or kz <= 0 or q.shape != (2, 2)
                or not np.isfinite(q).all() or not np.allclose(q, q.T)
                or np.min(np.linalg.eigvalsh(q)) <= 0):
            raise ValueError("invalid retained resolved Gaussian geometry")
        ellipses.append({"channel_index": index, "center_s": [kx/k, ky/k],
                         "covariance_s": (q/k**2).tolist()})
    return ellipses


def mask_diagnostic(field, launch, *, sigmas=MASK_SIGMAS):
    """Exclude the union of Mahalanobis-radius ellipses on the canonical FFT grid.

    NaNs explicitly denote excluded/unavailable diagnostic pixels. The generic
    Results stack converts arrays with asarray, so numpy.ma masks would be lost.
    Existing finite-only Auto limits and Matplotlib invalid masking support NaNs.
    """
    ellipses = carrier_ellipses(launch, sigmas=sigmas)
    sx, sy = (np.asarray(field.coordinates[key], dtype=float) for key in ("s_x", "s_y"))
    data = np.asarray(field.data)
    if field.axes != ("s_x", "s_y") or data.shape != (sx.size, sy.size):
        raise ValueError("canonical direction-cosine grid is required")
    spacings = []
    for axis in (sx, sy):
        d = np.diff(axis)
        if not d.size or not np.isfinite(axis).all() or d[0] <= 0 or not np.allclose(d, d[0], rtol=1e-10, atol=0):
            raise ValueError("uniform canonical FFT coordinates with at least two samples are required")
        spacings.append(float(d[0]))
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("finite nonnegative canonical intensity is required")
    excluded = np.zeros(data.shape, dtype=bool)
    for ellipse in ellipses:
        x = sx[:, None] - ellipse["center_s"][0]
        y = sy[None, :] - ellipse["center_s"][1]
        inverse = np.linalg.inv(ellipse["covariance_s"])
        distance = inverse[0, 0]*x*x + 2*inverse[0, 1]*x*y + inverse[1, 1]*y*y
        excluded |= distance <= sigmas**2
    cell = spacings[0]*spacings[1]
    total = float(np.sum(data, dtype=np.float64)*cell)
    off = float(np.sum(data[~excluded], dtype=np.float64)*cell)
    provenance = {
        "status": "available", "mask_convention": "union of incident Gaussian spectral covariance ellipses",
        "spectral_standard_deviations": float(sigmas),
        "ideal_single_lobe_enclosed_fraction": -math.expm1(-sigmas**2/2),
        "carrier_count": len(ellipses), "carriers": ellipses,
        "coordinate_convention": "absolute in-medium s_x=kx/k_internal, s_y=ky/k_internal",
        "excluded_pixel_count": int(excluded.sum()), "coordinate_cell_measure": cell,
        "off_carrier_field_norm": off, "total_field_norm": total,
        "off_carrier_fraction": off/total if total > 0 else None,
        "fraction_unavailable_reason": None if total > 0 else "total field norm is zero",
        "qualification": "discrete sampled field norm, not physical mW or a fanning-success metric; analytic unscreened/untruncated incident Gaussian widths; no output fitting",
    }
    masked = np.array(data, dtype=float, copy=True)
    masked[excluded] = np.nan
    product = replace(field, key=MASK_KEY,
                      display_name="Output Far-Field Intensity — zero-order masked",
                      data=masked, quantity="direction_cosine_field_norm_density",
                      value_unit="normalized field norm / direction-cosine²",
                      coordinates={**field.coordinates, "excluded_pixel_semantics": "NaN means carrier-excluded/unavailable, never measured zero",
                                   "carrier_mask": provenance}, initially_selected=False)
    return product, provenance


def add_completed_mask(fields, diagnostics, result):
    """Missing historical geometry disables this optional diagnostic explicitly."""
    canonical = fields.get("far_field_intensity")
    if canonical is None:
        values = {"status": "unavailable", "reason": "canonical far field is unavailable"}
    else:
        try:
            product, values = mask_diagnostic(canonical, result.launch_summary)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            values = {"status": "unavailable", "reason": str(exc)}
        else:
            fields.add(product.key, product)
    diagnostics.add("off_carrier", DiagnosticData("off_carrier", "Off-carrier field norm (Gaussian carrier exclusion)", values))
