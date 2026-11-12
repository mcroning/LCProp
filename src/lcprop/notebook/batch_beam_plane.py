"""Shared notebook BeamPlane using canonical channel edits and draft samples."""
from dataclasses import replace
from html import escape
from matplotlib.patches import Circle
import math

import ipywidgets as W

from lcprop.core.beams import BeamChannel, LEGACY_COHERENCE_GROUP
from lcprop.notebook._beam_position_editor import widget_canvas
from lcprop.notebook._launch_preview import ray_endpoint, update_decorations
from lcprop.notebook.batch_beams import objects
from lcprop.notebook.batch_requests import BEAM_CONTROLS, refractive_index
from lcprop.notebook.batch_results import draw_plane, retained_plane
from lcprop.optics.physical_launch import external_direction_for_exit


class BatchBeamPlane:
    def __init__(self, owner):
        self.owner = owner
        self.beam = W.Dropdown(description="Active beam")
        self.mode = W.ToggleButtons(options=["Move", "Tilt"])
        self.extent = W.ToggleButtons(options=["Fit", "Full aperture"], value="Full aperture")
        self.enabled = W.Checkbox(description="Enabled")
        self.coherence = W.Dropdown(description="Coherence", options=[
            ("Preserve saved groups", None), ("Coherent", "coherent"),
            ("Incoherent", "incoherent"),
        ])
        self.preview_status = W.HTML()
        self.fields = {
            name: W.FloatText(description=label, continuous_update=False,
                              style={"description_width": "initial"})
            for name, label in BEAM_CONTROLS.items()
        }
        self.add = W.Button(description="Add beam")
        self.duplicate = W.Button(description="Duplicate")
        self.delete = W.Button(description="Delete")
        self.inputs = (self.beam, self.mode, self.extent, self.enabled,
                       self.coherence, *self.fields.values(), self.add, self.duplicate, self.delete)
        self.figure = self.canvas = self.ax = None
        self.connections = ()
        self.gesture = self.drag_marker = None
        self.tilt_error = ""
        self.syncing = False
        self.generation = None
        try:
            self.figure, self.canvas = widget_canvas()
        except RuntimeError:
            canvas = W.HTML("Enable the notebook widget backend for graphical beam editing.")
        else:
            self.ax = self.figure.subplots()
            self.connections = tuple(self.canvas.mpl_connect(name, handler) for name, handler in (
                ("button_press_event", self.press), ("motion_notify_event", self.motion),
                ("button_release_event", self.release),
                ("figure_leave_event", self.cancel), ("key_press_event", self.key),
            ))
            canvas = self.canvas
        self.widget = W.VBox([
            W.HTML("<b>BeamPlane · draft launch</b>"),
            self.beam, W.HBox([self.add, self.duplicate, self.delete]),
            self.enabled, self.coherence, self.mode, self.extent, self.preview_status, canvas,
            *self.fields.values(),
        ])
        self.beam.observe(self._selection, names="value")
        self.mode.observe(self._change_mode, names="value")
        self.extent.observe(lambda _: self.draw(), names="value")
        self.enabled.observe(self._enabled, names="value")
        self.coherence.observe(self._regroup, names="value")
        for name, widget in self.fields.items():
            widget.observe(lambda change, name=name: self._numeric(name, change["new"]),
                           names="value")
        self.add.on_click(lambda _: self._run(lambda: self._add(False)))
        self.duplicate.on_click(lambda _: self._run(lambda: self._add(True)))
        self.delete.on_click(lambda _: self._run(self._delete))

    @property
    def session(self):
        return self.owner.session

    def _objects(self):
        return objects(self.session.draft.beams, self.session.presentation)

    def _run(self, action):
        if self.syncing or self.session.busy or self.session.disposed:
            return
        try:
            self.owner._guard(self.generation, action)
        finally:
            self.sync()

    def _selection(self, _):
        if not self.syncing:
            self.sync()

    def _current(self, items):
        return next(i for i, (channel, _) in enumerate(items) if channel.name == self.beam.value)

    def sync(self, *, redraw=True):
        if self.session.disposed:
            return
        self.cancel()
        self.generation = self.session.generation
        items = self._objects()
        self.syncing = True
        try:
            active = self.beam.value
            self.beam.options = [(channel.name + ("" if enabled else " (disabled)"), channel.name)
                                 for channel, enabled in items]
            names = [channel.name for channel, _ in items]
            self.beam.value = active if active in names else names[0]
            channel, enabled = items[self._current(items)]
            self.enabled.value = enabled
            for name, field in self.fields.items():
                field.value = getattr(channel, name)
            self.coherence.value = None
            for widget in self.inputs:
                widget.disabled = self.session.busy
            count = len(self.session.draft.beams.channels)
            self.enabled.disabled |= (enabled and count == 1) or (not enabled and count == 2)
            self.delete.disabled |= enabled and count == 1
        finally:
            self.syncing = False
        if redraw:
            self.draw()

    def _commit(self, items, coherence=None, select=None):
        if self.session.edit_beams(items, self.generation, coherence):
            self.owner.notice.value = ""
            self.owner._refresh_preview()
            if self.owner.source.value == "Draft preview":
                self.owner._update_selection()
            if select is not None:
                # Set after the updated list is populated by sync().
                self.sync(redraw=False)
                self.beam.value = select

    def _replace(self, **changes):
        items = list(self._objects())
        index = self._current(items)
        channel, enabled = items[index]
        items[index] = (replace(channel, **changes), enabled)
        self._commit(items)

    def _numeric(self, name, value):
        self._run(lambda: self._replace(**{name: value}))

    def _enabled(self, change):
        def apply():
            items = list(self._objects())
            index = self._current(items)
            items[index] = (items[index][0], change["new"])
            self._commit(items)

        self._run(apply)

    def _regroup(self, change):
        if change["new"] is not None:
            self._run(lambda: self._commit(self._objects(), change["new"]))

    def _delete(self):
        items = list(self._objects())
        del items[self._current(items)]
        self._commit(items)

    def _add(self, duplicate):
        items = list(self._objects())
        names = {channel.name for channel, _ in items}
        number = 2
        while f"beam-{number}" in names:
            number += 1
        name = f"beam-{number}"
        active = items[self._current(items)][0]
        group = (LEGACY_COHERENCE_GROUP if
                 self.session.draft.beams.channels[0].coherence_group == LEGACY_COHERENCE_GROUP
                 else f"{name}:laser")
        channel = (replace(active, name=name) if duplicate else
                   BeamChannel(name=name, x0_um=20., w1_um=10., w2_um=10.,
                               coherence_group=group))
        if ((channel.coherence_group == LEGACY_COHERENCE_GROUP)
                != (group == LEGACY_COHERENCE_GROUP)):
            channel = replace(channel, coherence_group=group)
        self._commit([*items, (channel, len(self.session.draft.beams.channels) < 2)], select=name)

    def _change_mode(self, change):
        self.tilt_error = ""
        self.draw()

    def draw(self):
        if self.session.disposed:
            return
        message = self.owner._preview_message
        self.preview_status.value = f"Intensity unavailable: {escape(message)}" if message else ""
        if self.ax is None:
            return
        self.cancel()
        ax = self.ax
        ax.clear()
        request = self.session.draft
        grid = request.grid
        index = refractive_index(self.session.route, request)
        x_half, y_half = grid.x_aperture_um / 2, grid.y_aperture_um / 2
        if self.owner._preview is not None:
            plane = retained_plane(self.owner._preview, None, ("x", "y"), None)
            draw_plane(ax, plane, False)
        ax.set(title="Draft input intensity\nInternal rays to exit",
               xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        self.markers, self.labels, self.arrows, self.tips = [], [], [], []
        selected = next((i for i, channel in enumerate(request.beams.channels)
                         if channel.name == self.beam.value), None)
        labels = [channel.name for channel in request.beams.channels]
        update_decorations(self, request, selected, index, labels=labels)
        for channel, enabled in self._objects():
            if not enabled:
                ax.plot(channel.x0_um, channel.y0_um, marker="x", color="gray")
        ax.set(xlim=(-x_half, x_half), ylim=(-y_half, y_half))
        if self.mode.value == "Tilt" and selected is not None:
            channel = request.beams.channels[selected]
            # The grazing external ray is the limiting internal exit radius.
            # The canonical inverse still validates every proposed endpoint.
            radius = (grid.z_length_um * channel.n_ext /
                      math.sqrt(index**2 - channel.n_ext**2)
                      if index > channel.n_ext else math.inf)
            if radius < min(x_half, y_half):
                ax.add_patch(Circle((channel.x0_um, channel.y0_um), radius,
                                    fill=False, color="cyan", linestyle="--"))
                self.preview_status.value += (
                    f"<br>Drag the square inside the dashed circle "
                    f"(exit radius &lt; {radius:.2f} µm).")
                if self.tilt_error:
                    self.preview_status.value += f"<br><b>{escape(self.tilt_error)}</b>"
                if self.extent.value == "Fit":
                    span = radius * 1.12
                    ax.set(xlim=(channel.x0_um - span, channel.x0_um + span),
                           ylim=(channel.y0_um - span, channel.y0_um + span))
                    self.canvas.draw_idle()
                    return
        if self.extent.value == "Fit":
            points = [(channel.x0_um, channel.y0_um) for channel, _ in self._objects()]
            for channel in request.beams.channels:
                try:
                    points.append(ray_endpoint(channel, request, index))
                except ValueError:
                    pass
            for setter, coords, half in zip((ax.set_xlim, ax.set_ylim), zip(*points), (x_half, y_half)):
                low, high = min(coords), max(coords)
                pad = max((high - low) * .12, half * .16)
                setter(low - pad, high + pad)
        self.canvas.draw_idle()

    def _point(self, event):
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            if math.isfinite(event.xdata) and math.isfinite(event.ydata):
                return event.xdata, event.ydata
        return None

    def press(self, event):
        self.cancel()
        if self.session.busy or self.session.disposed or event.button != 1 or self._point(event) is None:
            return
        if getattr(getattr(self.canvas, "toolbar", None), "mode", ""):
            return
        request = self.session.draft
        index = refractive_index(self.session.route, request)
        items = self._objects()
        channel, enabled = items[self._current(items)]
        target = (channel.x0_um, channel.y0_um)
        if self.mode.value == "Tilt":
            if not enabled:
                return
            try:
                target = ray_endpoint(channel, request, index)
            except ValueError:
                return
        px, py = self.ax.transData.transform(target)
        if math.hypot(event.x - px, event.y - py) <= 12:
            self.gesture = (self.generation, request, channel.name, self.mode.value,
                            target[0] - event.xdata, target[1] - event.ydata)

    def motion(self, event):
        point = self._point(event)
        if self.gesture is None or point is None or self.session.busy:
            return
        _, _, _, _, dx, dy = self.gesture
        x, y = point[0] + dx, point[1] + dy
        if self.drag_marker is None:
            self.drag_marker, = self.ax.plot(x, y, marker="+", color="yellow",
                                             scalex=False, scaley=False)
        else:
            self.drag_marker.set_data([x], [y])
        self.canvas.draw_idle()

    def release(self, event):
        gesture = self.gesture
        self.cancel()
        point = self._point(event)
        if gesture is None or point is None or event.button != 1:
            return
        generation, request, name, mode, dx, dy = gesture
        if (generation != self.session.generation or request is not self.session.draft
                or name != self.beam.value or mode != self.mode.value):
            return
        x, y = point[0] + dx, point[1] + dy
        if mode == "Move":
            self._run(lambda: self._replace(x0_um=x, y0_um=y))
        else:
            def tilt():
                items = self._objects()
                channel, _ = items[self._current(items)]
                try:
                    theta, phi = external_direction_for_exit(
                        channel, refractive_index(self.session.route, request),
                        request.grid.z_length_um, x, y)
                except ValueError as error:
                    self.tilt_error = (
                        "Tilt target is outside the reachable exit region. "
                        "Release the square closer to the beam center."
                    )
                    raise ValueError(self.tilt_error) from error
                self.tilt_error = ""
                self._replace(theta_ext_rad=theta, phi_rad=phi)

            self._run(tilt)

    def cancel(self, _=None):
        self.gesture = None
        if self.drag_marker is not None:
            self.drag_marker.remove()
            self.drag_marker = None
            self.canvas.draw_idle()

    def key(self, event):
        if event.key == "escape":
            self.cancel()

    def close(self):
        self.cancel()
        for connection in self.connections:
            self.canvas.mpl_disconnect(connection)
