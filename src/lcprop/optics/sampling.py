"""Analytic launch qualification; no fields, FFTs or material evolution.

For A(r)=exp(-r.T Q r), spectral power is proportional to
exp(-delta_k.T inv(Q) delta_k/2): its covariance is Q. Marginal
4-sigma intervals leave erfc(4/sqrt(2)) = 6.334e-5 power outside
an axis interval before aperture truncation or screens.
"""
from dataclasses import dataclass
import math

import numpy as np

from lcprop.optics.physical_launch import resolve_beam_geometry
from lcprop.optics.boundaries import TransverseBoundarySpec

SPECTRAL_SIGMAS = 4.0


@dataclass(frozen=True)
class LaunchSamplingReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def require_valid(self):
        if self.errors:
            raise ValueError("Optical launch sampling invalid; correct Grid Nx/Ny or aperture. "
                             + "\n".join(self.errors))

    def summary(self):
        return "\n".join(("Launch sampling: " + ("INVALID" if self.errors else "central carriers representable"),
                          *self.errors, *("Warning: " + w for w in self.warnings)))


def strict_minimum_samples(carrier, width):
    """Smallest integer N>=2 satisfying pi*N/L > |k|, without FFT rounding."""
    n = max(2, math.floor(abs(carrier)*width/math.pi) + 1)
    while math.pi/(width/n) <= abs(carrier):
        n += 1
    while n > 2 and math.pi/(width/(n-1)) > abs(carrier):
        n -= 1
    return n


def qualify_launch_sampling(beams, grid, n_ref, *, boundary=TransverseBoundarySpec(),
                            launch_elements=()):
    """GridSpec or RuntimeGrid; BeamStack already contains only enabled beams.

    Reject equality at either signed Nyquist edge, including even-grid -Nyquist:
    that bin cannot distinguish the two physical propagation directions.
    Boundary estimate is the unwrapped central ray plus a rigid two-radius
    entrance envelope, not a nonlinear/diffracting propagation prediction.
    """
    spec = getattr(grid, "spec", grid)
    spec.validate(); beams.validate(); boundary.validate()
    errors, warnings = [], []
    for number, beam in enumerate(beams.channels, 1):
        g = resolve_beam_geometry(beam, n_ref)
        q = g.interface_quadratic
        invq = np.linalg.inv(q)
        for axis, k, n, width, center, j in (
            ('x', g.kx, int(spec.Nx), float(spec.x_aperture_um), beam.x0_um, 0),
            ('y', g.ky, int(spec.Ny), float(spec.y_aperture_um), beam.y0_um, 1),
        ):
            spacing = width/n
            limit = math.pi/spacing
            margin = SPECTRAL_SIGMAS*math.sqrt(q[j, j])
            minimum = strict_minimum_samples(k, width)
            practical = strict_minimum_samples(abs(k)+margin, width)
            label = f"Beam {number} ({beam.name}), {axis}"
            if abs(k) >= limit:
                errors.append(
                    f"{label}: signed k{axis}={k:.9g}, |k{axis}|={abs(k):.9g} rad/µm; "
                    f"Nyquist={limit:.9g} rad/µm (strict inequality required), "
                    f"N{axis}={n}, L{axis}={width:g} µm, d{axis}={spacing:.9g} µm. "
                    f"Carrier-only minimum N{axis}={minimum}; Gaussian 4-sigma "
                    f"margin recommendation N{axis}>={practical} (not a power-of-two requirement).")
            if abs(k)+margin >= limit:
                warnings.append(f"{label}: Gaussian spectral 4-sigma interval approaches/exceeds Nyquist; "
                                f"|k|+4σ={abs(k)+margin:.9g} rad/µm, limit={limit:.9g}; "
                                f"recommended N{axis}>={practical}. Analytic unscreened, untruncated profile only.")
            exit_center = center + float(spec.z_length_um)*k/g.kz_internal
            extent = max(abs(center), abs(exit_center)) + 2*math.sqrt(invq[j, j])
            edge = width/2
            if boundary.mode == 'sponge':
                edge *= 1-boundary.width_fraction
            elif boundary.mode == 'tukey':
                edge *= 1-boundary.tukey_alpha
            if extent >= edge:
                consequence = ('content may wrap to the opposite side' if boundary.mode == 'periodic'
                               else 'content may be attenuated in the edge region')
                warnings.append(f"{label}: unwrapped ray/two-radius entrance-envelope estimate "
                                f"approaches/crosses {boundary.mode} boundary; {consequence}. "
                                "This is advisory, not a prediction of diffracting/nonlinear envelope evolution.")
    if launch_elements:
        warnings.append("Screens/launch elements may add spectral bandwidth; analytic Gaussian margin "
                        "does not certify the post-screen spectrum.")
    return LaunchSamplingReport(tuple(errors), tuple(warnings))
