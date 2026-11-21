"""Frontend launch artists for any channel count; scientific geometry is reused.

Adapted from the pinned engine's two-beam decoration helper. Keep this local
so notebook presentation does not modify engine or desktop behavior.
"""
from lcprop.notebook._launch_preview import ray_endpoint

BEAM_COLORS = ('white', 'red', 'cyan', 'yellow', 'magenta', 'lime', 'orange')


def update_decorations(self, request, active_beam, n_internal=None, *, labels=None):
    for artist in self.markers + self.labels + self.arrows + self.tips:
        artist.remove()
    self.markers, self.labels, self.arrows, self.tips = [], [], [], []
    for i, channel in enumerate(request.beams.channels):
        x, y = channel.x0_um, channel.y0_um
        color = BEAM_COLORS[i % len(BEAM_COLORS)]
        active = i == active_beam
        marker, = self.ax.plot([x], [y], marker='o', linestyle='none',
            markerfacecolor='none', markeredgecolor=color, markersize=10,
            markeredgewidth=2, scalex=False, scaley=False)
        self.markers.append(marker)
        label = f'Beam {i+1}' if labels is None else labels[i]
        self.labels.append(self.ax.annotate(label, (x, y), xytext=(7, 7),
                                           textcoords='offset points', color=color))
        try:
            tip = ray_endpoint(channel, request, n_internal)
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

