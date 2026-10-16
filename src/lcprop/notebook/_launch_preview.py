"""Shared launch artists; no widget backend or independent scientific model."""

def ray_endpoint(channel, request):
    """Same physical internal-ray endpoint used by the desktop LC host."""
    from lcprop.optics.physical_launch import resolve_beam_geometry
    geometry = resolve_beam_geometry(channel, float(request.material.no))
    length = float(request.grid.z_length_um)
    return (channel.x0_um + length * geometry.kx / geometry.kz_internal,
            channel.y0_um + length * geometry.ky / geometry.kz_internal)


def update_decorations(self, request, active_beam):
    for artist in self.markers + self.labels + self.arrows + self.tips:
        artist.remove()
    self.markers, self.labels, self.arrows, self.tips = [], [], [], []
    for i, channel in enumerate(request.beams.channels):
        x, y = channel.x0_um, channel.y0_um
        color = ('white', 'red')[i]
        active = i == active_beam
        marker, = self.ax.plot([x], [y], marker='o', linestyle='none',
            markerfacecolor='none', markeredgecolor=color, markersize=10,
            markeredgewidth=2, scalex=False, scaley=False)
        self.markers.append(marker)
        self.labels.append(self.ax.annotate(f'Beam {i+1}', (x, y), xytext=(7, 7),
                                           textcoords='offset points', color=color))
        try:
            tip = ray_endpoint(channel, request)
        except ValueError:
            tip = None
        arrow = self.ax.annotate('', xy=tip or (x, y), xytext=(x, y),
            arrowprops=dict(arrowstyle='->', color=color, linewidth=2.5 if active else 1,
                            alpha=1 if active else .5))
        endpoint, = self.ax.plot([tip[0] if tip else x], [tip[1] if tip else y],
            marker='s', linestyle='none', color=color, markersize=7 if active else 5,
            markerfacecolor='none', scalex=False, scaley=False)
        arrow.set_visible(tip is not None)
        endpoint.set_visible(tip is not None)
        self.arrows.append(arrow)
        self.tips.append(endpoint)
    # Moving decorations must never drive constrained-layout geometry.
    for artist in self.labels + self.arrows + self.tips + self.markers:
        artist.set_in_layout(False)

def update_intensity(self, grid, intensity):
    if self.mesh is None:
        self.mesh = self.ax.pcolormesh(grid.x_um, grid.y_um, intensity.T, shading='nearest')
        self.ax.set(aspect='equal', xlabel='x (µm)', ylabel='y (µm)',
                    title='Draft launch\nInternal rays to exit')
        self.ax.title.set_fontsize(10)
        self.ax.tick_params(labelsize=8)
        self.ax.xaxis.label.set_fontsize(9)
        self.ax.yaxis.label.set_fontsize(9)
        self.colorbar = self.figure.colorbar(self.mesh, ax=self.ax, pad=.035,
                                            label=r'$1/\mu\mathrm{m}^{2}$')
        self.colorbar.ax.tick_params(labelsize=8)
        self.colorbar.ax.yaxis.label.set_fontsize(9)
    else:
        self.mesh.set_array(intensity.T)
        self.mesh.set_clim(float(intensity.min()), float(intensity.max()))
