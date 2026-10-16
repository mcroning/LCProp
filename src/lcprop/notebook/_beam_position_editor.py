"""Optional launch-only canvas. Numeric controls remain the scientific model."""
import math
from ._launch_preview import ray_endpoint, update_decorations, update_intensity


def widget_canvas():
    # ipympl's package import selects its backend. Require explicit user setup first.
    import matplotlib
    if matplotlib.get_backend().lower() not in ('widget', 'ipympl', 'module://ipympl.backend_nbagg'):
        raise RuntimeError('First run %matplotlib widget in a fresh kernel before creating the controller')
    from matplotlib.figure import Figure
    from ipympl.backend_nbagg import Canvas, FigureManager
    figure = Figure(figsize=(3.75, 3.75), layout='constrained')
    canvas = Canvas(figure)
    FigureManager(canvas, id(figure))
    canvas.layout.width = '375px'
    canvas.layout.max_width = '100%'
    canvas.toolbar_visible = False
    canvas.header_visible = False
    canvas.footer_visible = False
    return figure, canvas


def gesture_request(request, index, action, x, y):
    """Change only the numeric fields addressed by the selected action."""
    from dataclasses import replace
    from lcprop.optics.physical_launch import external_direction_for_exit
    channel = request.beams.channels[index]
    if action == 'Move':
        channel = replace(channel, x0_um=x, y0_um=y)
    elif action == 'Tilt':
        theta, phi = external_direction_for_exit(channel, float(request.material.no),
                                                float(request.grid.z_length_um), x, y)
        channel = replace(channel, theta_ext_rad=theta, phi_rad=phi)
    else:
        raise ValueError('Unknown graphical action')
    channels = list(request.beams.channels)
    channels[index] = channel
    return replace(request, beams=replace(request.beams, channels=tuple(channels)))


class BeamPositionEditor:
    LIVE_INTERVAL = .25  # At most four launch samples/second, no queued timer work.
    SLOW_SECONDS = .1
    MAX_LIVE_UPDATES = 20  # Per gesture; final release is never omitted.
    DRAW_INTERVAL = 1/30

    def __init__(self, owner):
        from time import monotonic
        self.clock = monotonic
        self.owner = owner
        self.figure, self.canvas = widget_canvas()
        self.ax = self.figure.subplots()
        self.active = True
        self.gesture = None
        self.mesh = self.colorbar = None
        self.markers, self.labels, self.arrows, self.tips = [], [], [], []
        self.committed_intensity = None
        self.live_count = 0
        self.live_disabled = False
        self._layout_frozen = False
        self._frontend_initialized = False
        # Public widget message observation; never write backend handshake state.
        self.canvas.on_msg(self._canvas_message)
        self.ids = [self.canvas.mpl_connect(name, handler) for name, handler in (
            ('button_press_event', self.press), ('motion_notify_event', self.motion),
            ('button_release_event', self.release), ('figure_leave_event', self.cancel),
            ('key_press_event', self.key), ('resize_event', self.resize))]

    def _canvas_message(self, widget, content, buffers):
        if self.active and content.get('type') == 'initialized':
            self._frontend_initialized = True

    def _draw_committed(self):
        if not self._frontend_initialized:
            # Resolve layout locally without premature image transmission.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg.draw(self.figure.canvas)
        # Let the widget frontend schedule its normal draw response. Forced
        # refresh/direct draws reproduced corruption in the minimal browser test.
        self.canvas.draw_idle()

    def positions(self):
        o = self.owner
        points = [(o.controls['x'].value, o.controls['y'].value)]
        if o.second_enabled.value:
            points.append((o.beam2_controls['x0_um'].value, o.beam2_controls['y0_um'].value))
        return points

    def synchronize(self, *, immediate=False, redraw=True):
        self.gesture = None
        try:
            request = self.owner.request()
        except (ValueError, TypeError):
            if redraw:
                self.canvas.draw_idle()
            return
        update_decorations(self, request, self.owner.active_beam.value)
        if redraw:
            if immediate:
                self._draw_committed()
            else:
                self.canvas.draw_idle()

    def show_intensity(self, intensity):
        self.mesh.set_array(intensity.T)
        self.mesh.set_clim(float(intensity.min()), float(intensity.max()))

    def update(self, grid, intensity, request):
        self.grid = grid
        self.bounds = (-request.grid.x_aperture_um/2, request.grid.x_aperture_um/2,
                       -request.grid.y_aperture_um/2, request.grid.y_aperture_um/2)
        self.committed_intensity = intensity
        update_intensity(self, grid, intensity)
        # Solve the layout once, then freeze it. Dynamic colorbar tick text and
        # moving beam labels must not resize the plotting area on every draw.
        first_layout = not self._layout_frozen
        self.synchronize(immediate=True)
        if first_layout:
            self._freeze_layout()

    def _freeze_layout(self):
        positions = [ax.get_position().frozen() for ax in self.figure.axes]
        self.figure.set_layout_engine('none')
        self._layout_frozen = True
        for ax, position in zip(self.figure.axes, positions):
            ax.set_position(position)
        self._draw_committed()

    def resize(self, event):
        # A real canvas resize may need new margins. Ordinary draws/gestures do not.
        if not self.active or self.mesh is None or not self._layout_frozen:
            return
        self.cancel()
        for ax in self.figure.axes:
            ax.set_in_layout(True)
        self.figure.set_layout_engine('constrained')
        self._draw_committed()
        self._freeze_layout()

    def ready(self):
        toolbar = getattr(self.canvas, 'toolbar', None)
        return (self.active and self.owner._position_editor is self and
                not self.owner.disposed and not self.owner.busy and not self.owner.loading and
                not getattr(toolbar, 'mode', '') and not self.ax.get_navigate_mode())

    def valid(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return False
        x, y = event.xdata, event.ydata
        return (math.isfinite(x) and math.isfinite(y) and
                self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3])

    def press(self, event):
        if not self.ready() or event.button != 1 or not self.valid(event):
            self.cancel()
            return
        self.cancel()
        i = self.owner.active_beam.value
        request = self.owner.request()
        if i >= len(request.beams.channels):
            return
        action = self.owner.action.value
        channel = request.beams.channels[i]
        try:
            x, y = ray_endpoint(channel, request) if action == 'Tilt' else (channel.x0_um, channel.y0_um)
        except ValueError:
            return
        px, py = self.ax.transData.transform((x, y))
        if math.hypot(event.x-px, event.y-py) <= 12:
            self.gesture = (i, x-event.xdata, y-event.ydata, action, request)
            self.next_live = self.clock() + self.LIVE_INTERVAL
            self.next_draw = self.clock()
            self.live_count = 0
            self.live_disabled = False

    def proposed(self, event):
        if not self.gesture or not self.ready() or not self.valid(event):
            return None
        i, dx, dy, action, request = self.gesture
        x, y = event.xdata+dx, event.ydata+dy
        if not (self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]):
            return None
        return i, x, y, action, request

    def live(self, point):
        if not self.owner.live_preview.value or self.live_disabled:
            return
        now = self.clock()
        if now < self.next_live:
            return
        if self.live_count >= self.MAX_LIVE_UPDATES:
            self.live_disabled = True
            self.owner.editor_status.value = 'Live update cap reached; last provisional image retained until release.'
            return
        i, x, y, action, request = point
        from .interactive_lc import launch_preview
        self.live_count += 1
        try:
            proposed = gesture_request(request, i, action, x, y)
            _, _, intensity = launch_preview(proposed)
            self.show_intensity(intensity)
            self.owner.editor_status.value = (f'Provisional canonical launch: {action} target ({x:.2f}, {y:.2f}) µm; '
                                              'numeric controls commit only on release.')
        except (ValueError, TypeError) as error:
            from html import escape
            self.live_disabled = True
            self.owner.editor_status.value = 'Live launch unavailable; marker-only fallback: ' + escape(str(error))
        elapsed = self.clock()-now
        self.next_live = self.clock()+self.LIVE_INTERVAL
        if elapsed > self.SLOW_SECONDS:
            self.live_disabled = True
            self.owner.editor_status.value = 'Live launch exceeded 100 ms; last provisional image retained, marker-only until release.'

    def motion(self, event):
        point = self.proposed(event)
        if point is None:
            if self.gesture:
                self.cancel()
            return
        i, x, y, action, request = point
        channel = request.beams.channels[i]
        if action == 'Move':
            self.markers[i].set_data([x], [y])
            self.labels[i].xy = (x, y)
            tip = ray_endpoint(channel, request)
            tip = (tip[0]+x-channel.x0_um, tip[1]+y-channel.y0_um)
            self.arrows[i].set_position((x, y))
        else:
            tip = (x, y)
        self.arrows[i].xy = tip
        self.tips[i].set_data([tip[0]], [tip[1]])
        if not self.live_count:
            self.owner.editor_status.value = 'Proposed geometry only; numeric controls unchanged until release.'
        self.live(point)
        now = self.clock()
        if now >= self.next_draw:
            self.canvas.draw_idle()
            self.next_draw = now+self.DRAW_INTERVAL

    def release(self, event):
        point = self.proposed(event) if event.button == 1 else None
        self.cancel(redraw=point is None)
        if point is None:
            return
        i, x, y, action, request = point
        try:
            changed = gesture_request(request, i, action, x, y).beams.channels[i]
        except (ValueError, TypeError) as error:
            from html import escape
            self.canvas.draw_idle()
            self.owner.editor_status.value = 'Gesture rejected; controls unchanged: '+escape(str(error))
            return
        o = self.owner
        if action == 'Move':
            controls = (o.controls['x'], o.controls['y']) if i == 0 else (
                o.beam2_controls['x0_um'], o.beam2_controls['y0_um'])
            values = changed.x0_um, changed.y0_um
        else:
            group = o.beam1_controls if i == 0 else o.beam2_controls
            controls = group['theta_ext_rad'], group['phi_rad']
            values = changed.theta_ext_rad, changed.phi_rad
        o.loading = True
        try:
            controls[0].value, controls[1].value = values
        finally:
            o.loading = False
        o._changed()

    def cancel(self, event=None, *, redraw=True):
        if self.active:
            if self.mesh is not None and self.committed_intensity is not None:
                self.show_intensity(self.committed_intensity)
            self.synchronize(redraw=redraw)
            self.owner.editor_status.value = 'Select Active Beam and Move/Tilt; arrows show internal rays to exit.'

    def key(self, event):
        if event.key == 'escape':
            self.cancel()

    def close(self):
        self.active = False
        self.canvas.on_msg(self._canvas_message, remove=True)
        self.gesture = None
        for connection in self.ids:
            self.canvas.mpl_disconnect(connection)
        toolbar = getattr(self.canvas, 'toolbar', None)
        if toolbar is not None and hasattr(toolbar, 'close'):
            toolbar.close()
        close = getattr(self.canvas, 'close', None)
        if close:
            close()
        self.figure.clear()
