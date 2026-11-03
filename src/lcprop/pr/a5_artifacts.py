"""Promoted frozen A5 compact_artifacts definition, with lazy row exclusion.

Source: preserved a5_package.py SHA256
788ff423aa3f73b42dd8baefb6a01fb22d0121c04e0cd93b339891b0f13a5b23.
Only array coercion/row-mask input handling differs from that reference function.
No full-plane masked copy is required. All histogram definitions are unchanged.
"""
import json
import math
import numpy as np


def compact_artifacts(power, sx, sy, carrier_sx, *, excluded_row=None):
    """Finite-support diagnostics; excluded pixels never become measured zeros.

    Radial means divide by valid pixel counts. Roughness uses only consecutive
    triples of fully observed annuli, excluding partial annuli and gap edges.
    """
    sx, sy = np.asarray(sx), np.asarray(sy)
    if power.shape != (sx.size, sy.size) or sx.ndim != 1 or sy.ndim != 1:
        raise ValueError('spectrum and coordinate shapes differ')
    if not np.isfinite(carrier_sx):
        raise ValueError('finite carrier coordinate required')
    edges = np.linspace(0, 2, 129)
    angles = np.linspace(-math.pi, math.pi, 73)
    radial, angular = np.zeros(128), np.zeros(72)
    radial_count, angular_count = np.zeros(128, dtype=int), np.zeros(72, dtype=int)
    radial_possible = np.zeros(128, dtype=int)
    total = high = overflow = 0.
    valid_count = 0
    for i, x in enumerate(sx):
        xx = x - carrier_sx
        r, angle = np.hypot(xx, sy), np.arctan2(sy, xx)
        row = power[i]
        if excluded_row is not None:
            row = np.ma.array(row, mask=excluded_row(i), copy=False)
        w = np.asarray(np.ma.getdata(row), dtype=np.float64)
        coordinates_valid = np.isfinite(r) & np.isfinite(angle)
        valid = coordinates_valid & np.isfinite(w) & ~np.ma.getmaskarray(row)
        if np.any(w[valid] < 0):
            raise ValueError('nonnegative spectrum required')
        radial_possible += np.histogram(r[coordinates_valid], bins=edges)[0]
        rv, av, wv = r[valid], angle[valid], w[valid]
        valid_count += int(valid.sum())
        total += float(wv.sum())
        high += float(wv[rv >= .25].sum())
        overflow += float(wv[rv > 2].sum())
        radial += np.histogram(rv, bins=edges, weights=wv)[0]
        angular += np.histogram(av, bins=angles, weights=wv)[0]
        radial_count += np.histogram(rv, bins=edges)[0]
        angular_count += np.histogram(av, bins=angles)[0]
    if not (np.isfinite([total, high, overflow]).all()
            and np.isfinite(radial).all() and np.isfinite(angular).all()):
        raise ValueError('spectrum diagnostic reduction overflow')
    means = np.full(128, np.nan)
    np.divide(radial, radial_count, out=means, where=radial_count > 0)
    complete = (radial_count > 0) & (radial_count == radial_possible)
    centers = np.flatnonzero(complete[:-2] & complete[1:-1] & complete[2:]) + 1
    roughness = None
    reasons = {}
    if np.any(radial_count == 0) or np.any(angular_count == 0):
        reasons['empty_bins'] = 'Null bin values have no valid samples; see valid pixel counts'
    if not centers.size:
        reasons['roughness'] = 'No consecutive triple of fully observed nonempty annuli'
    else:
        denominator = float(np.abs(means[centers]).sum())
        if denominator > 0:
            roughness = float(np.abs(means[centers-1] - 2*means[centers]
                                     + means[centers+1]).sum() / denominator)
        else:
            reasons['roughness'] = 'Zero intensity on eligible roughness centers'
    if total <= 0:
        reasons['power_fractions'] = ('No finite valid support' if not valid_count
                                      else 'Zero total power on finite valid support')
    def fractions(values, counts):
        return [float(v/total) if total > 0 and n > 0 else None
                for v, n in zip(values, counts)]
    result = {
        'radial_edges_direction_cosine': edges.tolist(),
        'radial_power_fraction': fractions(radial, radial_count),
        'radial_mean_per_pixel': [float(v) if n else None for v, n in zip(means, radial_count)],
        'radial_valid_pixel_count': radial_count.tolist(),
        'radial_available_coordinate_count': radial_possible.tolist(),
        'angular_edges_rad': angles.tolist(),
        'angular_power_fraction': fractions(angular, angular_count),
        'angular_valid_pixel_count': angular_count.tolist(),
        'valid_pixel_count': valid_count,
        'excluded_or_invalid_pixel_count': int(power.size) - valid_count,
        'valid_power_denominator': total if valid_count else None,
        'high_frequency_rho_gt_0_25_fraction': high/total if total > 0 else None,
        'radial_overflow_rho_gt_2_fraction': overflow/total if total > 0 else None,
        'radial_roughness_L1_second_difference_over_L1': roughness,
        'roughness_eligible_center_bins': centers.tolist(),
        'unavailable_reasons': reasons,
        'definition': ('Only finite unmasked pixels with finite coordinates contribute. '
                       'Fractions divide by total power on that same valid support; '
                       'empty bins are null (no valid samples), not measured zero. '
                       'Radial means divide by valid pixel counts. rho around carrier; '
                       '128 bins[0,2],72 angular bins; high frequency rho>=0.25 includes overflow. '
                       'Roughness sums absolute second differences only on contiguous triples '
                       'of fully observed nonempty annuli, normalized by absolute mean intensity '
                       'on those center bins. Partial annuli and gaps are never bridged. '
                       'Null scalars have reasons; null bin values mean no valid support '
                       'or unavailable power denominator. No pass/fail threshold; later metrics exploratory.')}
    # Reject arithmetic overflow or accidental nonfinite output; never emit NaN/Infinity.
    json.dumps(result, allow_nan=False)
    return result
