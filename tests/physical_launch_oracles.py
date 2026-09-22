"""Independent formulas for physical-launch workflow regression assertions."""
import numpy as np


def scalar_lineage_flux(field, grid, beams, n_internal):
    """Homogeneous scalar reference; no coherence cross terms or vector claim."""
    nx, ny = field.shape[-2:]
    dx, dy = grid.x_aperture_um/nx, grid.y_aperture_um/ny
    kx = 2*np.pi*np.fft.fftfreq(nx, dx)[:, None]
    ky = 2*np.pi*np.fft.fftfreq(ny, dy)[None, :]
    total = sum(b.power_mW for b in beams.channels)
    flux = 0.
    for plane, beam in zip(field, beams.channels):
        k = 2*np.pi*n_internal/beam.wavelength_um
        flux += total*dx*dy/(nx*ny)*np.sum(
            np.sqrt(np.maximum(0., 1-(kx*kx+ky*ky)/k**2))*abs(np.fft.fft2(plane))**2)
    return float(flux)


def normal_gaussian_norm(grid, beams):
    """Exact sampled quadrature of analytically normalized normal Gaussian beams."""
    dx, dy = grid.x_aperture_um/grid.Nx, grid.y_aperture_um/grid.Ny
    x = ((np.arange(grid.Nx)+.5)*dx-grid.x_aperture_um/2)[:, None]
    y = ((np.arange(grid.Ny)+.5)*dy-grid.y_aperture_um/2)[None, :]
    total = sum(b.power_mW for b in beams.channels)
    norm = 0.
    for b in beams.channels:
        assert b.theta_ext_rad == b.psi_rad == 0
        norm += b.power_mW/total*2/(np.pi*b.w1_um*b.w2_um)*np.sum(
            np.exp(-2*((x-b.x0_um)/b.w1_um)**2-2*((y-b.y0_um)/b.w2_um)**2))*dx*dy
    return float(norm)
