"""Scalar, collimated physical launch geometry owned by LCProp.

All geometry uses vacuum wavelength and external beam-normal radii. The
interface is ideal unit *power* transmission, not a Fresnel coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Qualification thresholds, not fit parameters or renormalization targets.
MINIMUM_FORWARD_COSINE = 1.0e-8
CAPTURE_TOLERANCE = 1.0e-3
SPECTRAL_COSINE_RELATIVE_TOLERANCE = 1.0e-2
NONPROPAGATING_NORM_TOLERANCE = 1.0e-6
NYQUIST_EDGE_NORM_TOLERANCE = 1.0e-6


def transverse_frame(theta: float, phi: float, psi: float = 0.0) -> np.ndarray:
    """Rows are rolled transverse unit vectors under the zero-roll convention."""
    axis = np.array([-math.sin(phi), math.cos(phi), 0.0])
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = (np.eye(3) + math.sin(theta) * cross
                + (1.0 - math.cos(theta)) * (cross @ cross))
    e1, e2 = rotation[:, 0], rotation[:, 1]
    return np.stack((math.cos(psi)*e1 + math.sin(psi)*e2,
                     -math.sin(psi)*e1 + math.cos(psi)*e2))


@dataclass(frozen=True)
class ResolvedBeamGeometry:
    kx: float
    ky: float
    kz_external: float
    kz_internal: float
    k_internal: float
    theta_internal: float
    cosine_external: float
    cosine_internal: float
    external_frame: np.ndarray
    interface_quadratic: np.ndarray
    internal_quadratic: np.ndarray

    def summary(self) -> dict:
        return {
            "interface_model": "isotropic_scalar_unit_power_transmission",
            "kx_rad_per_um": self.kx, "ky_rad_per_um": self.ky,
            "kz_external_rad_per_um": self.kz_external,
            "kz_internal_rad_per_um": self.kz_internal,
            "theta_internal_rad": self.theta_internal,
            "internal_slopes": [self.kx/self.kz_internal, self.ky/self.kz_internal],
            "cosine_external": self.cosine_external,
            "cosine_internal": self.cosine_internal,
            "interface_quadratic_per_um2": self.interface_quadratic.tolist(),
            "internal_quadratic_per_um2": self.internal_quadratic.tolist(),
            "internal_principal_radii_um": (
                1/np.sqrt(np.linalg.eigvalsh(self.internal_quadratic))).tolist(),
            "qualification": "scalar reference; no vector or anisotropic interface solution",
        }


def resolve_beam_geometry(channel, n_internal: float) -> ResolvedBeamGeometry:
    channel.validate()
    if not math.isfinite(n_internal) or n_internal <= 0:
        raise ValueError("internal optical index must be finite and positive")
    if math.cos(channel.theta_ext_rad) <= MINIMUM_FORWARD_COSINE:
        raise ValueError("numerically grazing external launch is unsupported")
    k0 = 2*math.pi/channel.wavelength_um
    kext, kint = channel.n_ext*k0, n_internal*k0
    theta, phi = channel.theta_ext_rad, channel.phi_rad
    kx = kext*math.sin(theta)*math.cos(phi)
    ky = kext*math.sin(theta)*math.sin(phi)
    kz2 = kint*kint-kx*kx-ky*ky
    if kz2 <= 0:
        raise ValueError("physical launch has no forward propagating transmitted mode")
    kz = math.sqrt(kz2)
    if not math.isfinite(kint) or kz/kint <= MINIMUM_FORWARD_COSINE:
        raise ValueError("numerically grazing internal launch is unsupported")
    ti = math.atan2(math.hypot(kx, ky), kz)
    frame = transverse_frame(theta, phi, channel.psi_rad)
    mext = frame[:, :2]
    qface = mext.T @ np.diag([1/channel.w1_um**2, 1/channel.w2_um**2]) @ mext
    mint_inverse = np.linalg.inv(transverse_frame(ti, phi)[:, :2])
    qint = mint_inverse.T @ qface @ mint_inverse
    return ResolvedBeamGeometry(kx, ky, kext*math.cos(theta), kz, kint, ti,
                                math.cos(theta), kz/kint, frame, qface, qint)


def external_direction_for_exit(channel, n_internal: float, length_um: float,
                                exit_x_um: float, exit_y_um: float) -> tuple[float, float]:
    """Invert the scalar interface for a requested forward physical ray endpoint.

    Choose the equivalent signed-theta branch with azimuth closest to the
    current azimuth; ties retain the current theta sign (positive at +0).
    At zero displacement retain phi and signed zero. Thus crossing zero along
    the current meridian changes theta sign rather than jumping phi by pi.
    An unreachable/grazing external direction is rejected, never clamped.
    """
    channel.validate()
    if not all(math.isfinite(v) for v in (n_internal, length_um, exit_x_um, exit_y_um)):
        raise ValueError("inverse ray geometry must be finite")
    if n_internal <= 0 or length_um <= 0:
        raise ValueError("inverse ray requires positive internal index and interaction length")
    sx = (exit_x_um-channel.x0_um)/length_um
    sy = (exit_y_um-channel.y0_um)/length_um
    if not math.isfinite(sx) or not math.isfinite(sy):
        raise ValueError("inverse ray slopes must be finite")
    norm = math.hypot(1., sx, sy)
    k0 = 2*math.pi/channel.wavelength_um
    kx, ky = n_internal*k0*sx/norm, n_internal*k0*sy/norm
    sine = math.hypot(kx, ky)/(channel.n_ext*k0)
    if sine >= 1 or math.sqrt(max(0., 1-sine*sine)) <= MINIMUM_FORWARD_COSINE:
        raise ValueError("requested exit requires unreachable or grazing external incidence")
    if 1/norm <= MINIMUM_FORWARD_COSINE:
        raise ValueError("numerically grazing internal launch is unsupported")
    if sine == 0:
        return math.copysign(0., channel.theta_ext_rad), channel.phi_rad
    theta = math.asin(sine)
    phi = math.atan2(ky, kx) % (2*math.pi)
    alternatives = ((theta, phi), (-theta, (phi+math.pi) % (2*math.pi)))
    distances = [abs(math.remainder(p-channel.phi_rad, 2*math.pi)) for _, p in alternatives]
    if math.isclose(*distances, abs_tol=1e-12, rel_tol=0):
        return alternatives[1 if math.copysign(1., channel.theta_ext_rad) < 0 else 0]
    return alternatives[0 if distances[0] < distances[1] else 1]


def sample_resolved_beam(channel, geometry, grid, power_fraction, complex_dtype):
    """Analytic infinite-plane normalization, followed by finite-grid sampling."""
    xp = grid.xp
    x = grid.x_um[:, None]-channel.x0_um
    y = grid.y_um[None, :]-channel.y0_um
    q = geometry.interface_quadratic
    envelope = xp.exp(-(q[0, 0]*x*x + 2*q[0, 1]*x*y + q[1, 1]*y*y))
    scale = math.sqrt(power_fraction * geometry.cosine_external /
                      (geometry.cosine_internal * math.pi*channel.w1_um*channel.w2_um/2))
    phase = geometry.kx*x + geometry.ky*y + channel.phase_rad
    return (scale*envelope*xp.exp(1j*phase)).astype(complex_dtype, copy=False)


def scalar_flux_diagnostic(field, grid, geometry, power_scale):
    """Only scalar reductions cross to the host; no volume host transfer.

    The weighted flux is a homogeneous isotropic scalar reference, not a vector
    Poynting claim. Qualification tests the whole sampled spectrum after screens.
    """
    xp = grid.xp
    spectrum = xp.abs(xp.fft.fft2(field))**2
    kx = 2*np.pi*xp.fft.fftfreq(grid.Nx, d=grid.dx_um)[:, None]
    ky = 2*np.pi*xp.fft.fftfreq(grid.Ny, d=grid.dy_um)[None, :]
    radial = (kx*kx+ky*ky)/geometry.k_internal**2
    propagating = radial < 1
    cosine = xp.sqrt(xp.maximum(0, 1-radial))
    total = float(xp.sum(spectrum))
    factor = power_scale*grid.dx_um*grid.dy_um/(grid.Nx*grid.Ny)
    weighted = float(xp.sum(cosine*spectrum))*factor
    if total == 0:
        return {"scalar_axial_current_mW": 0., "narrow_band_available": True,
                "reasons": [], "cosine_relative_rms": 0.,
                "nonpropagating_norm_fraction": 0., "nyquist_edge_norm_fraction": 0.}
    nonprop = float(xp.sum(xp.where(propagating, 0, spectrum)))/total
    relative = float(xp.sum(spectrum*(cosine/geometry.cosine_internal-1)**2)/total)**0.5
    edge = ((xp.abs(kx) >= .9*np.pi/grid.dx_um) |
            (xp.abs(ky) >= .9*np.pi/grid.dy_um))
    edge_fraction = float(xp.sum(xp.where(edge, spectrum, 0)))/total
    reasons = []
    if abs(geometry.kx) >= np.pi/grid.dx_um or abs(geometry.ky) >= np.pi/grid.dy_um:
        reasons.append("carrier exceeds grid Nyquist limit; sampled spectrum is aliased")
    if relative > SPECTRAL_COSINE_RELATIVE_TOLERANCE:
        reasons.append("spectrum exceeds constant-carrier cosine tolerance")
    if nonprop > NONPROPAGATING_NORM_TOLERANCE:
        reasons.append("nonpropagating spectral support")
    if edge_fraction > NYQUIST_EDGE_NORM_TOLERANCE:
        reasons.append("spectrum approaches grid Nyquist boundary; refine grid")
    return {"scalar_axial_current_mW": weighted,
            "narrow_band_available": not reasons, "reasons": reasons,
            "cosine_relative_rms": relative,
            "nonpropagating_norm_fraction": nonprop,
            "nyquist_edge_norm_fraction": edge_fraction}
