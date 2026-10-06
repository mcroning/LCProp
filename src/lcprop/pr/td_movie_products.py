"""Bounded observations of accepted TD source maps and optical endpoints.

No propagation, material response or scientific state mutation occurs here.
Each channel retains at most 36 x 128 x 128 float32 samples, including Fast.
"""
import numpy as np
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr.visualization import downsample_td_movie_frame
from lcprop.pr.trajectory import pack_frames


def _axis_preview(values, *, xp, asnumpy):
    edges = np.linspace(0, len(values), min(128, len(values)) + 1, dtype=int)
    reduced = xp.stack([xp.mean(values[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return np.asarray(asnumpy(reduced)).tolist()


class AcceptedMovieProducts:
    """Host retention is bounded; reduction precedes device-to-host export."""

    def __init__(self):
        self.frames = {key: [] for key in ('xz', 'yz', 'far_field')}
        self.metadata = {}

    def append(self, field, source, *, grid, request, reference, groups, asnumpy):
        if len(self.frames['xz']) >= 36:
            raise ValueError('TD movie frame bound exceeded')
        xp = grid.xp
        summary = grid.summary()
        nx, ny = field.shape[-2:]
        x = (np.arange(nx) - .5 * (nx - 1)) * summary['dx_um']
        y = (np.arange(ny) - .5 * (ny - 1)) * summary['dy_um']
        ix, iy = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
        published = request.optical_coupling == 'frozen_material_published_optical_first_v1'
        # Historical reduced uses entrance samples; historical transverse uses centers.
        offset = 1 if published else (.5 if hasattr(request, 'resolved_optical_coupling') else 0)
        z = (np.arange(source.shape[0]) + offset) * summary['dz_um']
        inverse = (0.0 if getattr(reference, 'optical_scale_W_cm2', None) == 0 else float(reference))
        background = request.material.background_intensity
        for key, cut, axes, coordinates in (
            ('xz', source[:, :, iy].T, ('x', 'z'), (x, z)),
            ('yz', source[:, ix, :].T, ('y', 'z'), (y, z)),
        ):
            optical = (cut - background) * inverse
            self.frames[key].append(downsample_td_movie_frame(optical, xp=xp, asnumpy=asnumpy))
            self.metadata[key] = dict(
                axes=list(axes), coordinates={a: _axis_preview(v, xp=np, asnumpy=np.asarray)
                                             for a, v in zip(axes, coordinates)},
                quantity='optical_intensity', value_unit='1/um^2',
                observation='accepted-state arriving source samples' if published else 'accepted-state historical source samples',
                x_cut_um=float(x[ix]), y_cut_um=float(y[iy]),
                display_name=f'Accepted optical {key} trajectory',
            )
        spectrum = direction_cosine_spectrum(
            field, dx_um=summary['dx_um'], dy_um=summary['dy_um'],
            wavelength_um=request.beams.channels[0].wavelength_um,
            refractive_index=request.material.refractive_index, coherence_groups=groups, xp=xp)
        self.frames['far_field'].append(downsample_td_movie_frame(spectrum.intensity, xp=xp, asnumpy=asnumpy))
        self.metadata['far_field'] = dict(
            axes=['s_x', 's_y'],
            coordinates={a: _axis_preview(v, xp=xp, asnumpy=asnumpy)
                         for a, v in (('s_x', spectrum.s_x), ('s_y', spectrum.s_y))},
            quantity='direction_cosine_field_norm_density',
            value_unit='normalized field norm / direction-cosine²',
            display_name='Accepted output far-field trajectory',
            observation='actual accepted-state output boundary',
            normalization='fftshift(fft2(A))*dx*dy; squared modulus times (n/lambda0)^2',
            reduction='full-resolution trusted spectrum, then contiguous block mean',
        )

    def attach(self, metadata):
        if not self.frames['xz']:
            return
        metadata['additional_movies'] = {
            key: dict(self.metadata[key], frames=pack_frames(np.stack(frames)),
                      fixed_color_limits=[min(0.0, float(np.min(frames))),
                                          max(float(np.max(frames)), 1.0) if np.max(frames) <= 0 else float(np.max(frames))])
            for key, frames in self.frames.items()
        }
