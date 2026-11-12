"""Interactive notebook for bounded LC static and reduced PR TD requests."""
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import ipywidgets as W
from IPython.display import display
from matplotlib import pyplot as plt

from lcprop.notebook.batch_beam_plane import BatchBeamPlane
from lcprop.notebook.batch_files import archive_completed, selected_upload, unpack_completed
from lcprop.notebook.batch_requests import (
    PR_MATERIAL_CONTROLS, preset, preview, validate,
)
from lcprop.notebook.batch_results import draw_plane, field_coordinates, retained_plane
from lcprop.notebook.batch_session import BatchSession

LC_VOLUME_FIELDS = {"intensity_stack": "Intensity", "delta_theta_stack": "Director response (Δθ)"}

PLANES = {"x–y": ("x", "y"), "x–z": ("z", "x"), "y–z": ("z", "y")}


def disclosure(label, content):
    toggle = W.ToggleButton(description=label, layout=W.Layout(width="auto"))
    content.layout.display = "none"

    def show(change):
        content.layout.display = "" if change["new"] else "none"

    toggle.observe(show, names="value")
    return W.VBox([toggle, content])


def close_widgets(widget):
    for child in getattr(widget, "children", ()):
        close_widgets(child)
    widget.close()


class InteractiveBatch:
    """One controller instance. Run requires an active notebook event loop."""

    fixed_route = None

    def __init__(self, *, runner=None, validator=validate, preview_factory=preview, gpu_probe=None):
        self.preview_factory = preview_factory
        self.directory = Path(tempfile.mkdtemp(prefix="lcprop-notebook-"))
        self.session = BatchSession(runner=runner, validator=validator, notify=self._sync,
                                    fixed_route=self.fixed_route, gpu_probe=gpu_probe)
        self.editor = W.VBox()
        self.beam_plane = BatchBeamPlane(self)
        self.status = W.HTML()
        self.progress_view = W.HTML()
        self.notice = W.HTML()
        self.stop_button = W.Button(description="Stop", disabled=True)
        self.source = W.Dropdown(description="View", options=["Draft preview"])
        self.field = W.Dropdown(description="Field")
        if self.fixed_route == "lc":
            self.cut = W.ToggleButtons(
                description="Plane", options=[(plane.replace("–", "").upper(), plane)
                                              for plane in PLANES],
            )
        else:
            self.cut = W.Dropdown(description="Plane", options=list(PLANES))
        self.aspect = W.Dropdown(description="Aspect", options=["Physical", "Square"])
        self.position = (W.Dropdown(description="Cut")
                         if self.fixed_route == "lc" else
                         W.IntSlider(description="Cut sample", continuous_update=False))
        self.plot = W.Output()
        self.canvas_host = W.VBox()
        self.plot_message = W.HTML()
        self._figure = None
        self.details = W.HTML()
        self.files = W.VBox()
        self.extra = W.VBox([
            disclosure("Save / load / export", self.files),
            disclosure("Executed request and diagnostics", self.details),
        ])
        self.refresh_button = W.Button(description="Refresh status")
        self.refresh_button.on_click(self._refresh_status)
        self.actions = W.HBox([self.stop_button, self.refresh_button])
        self.widget = W.VBox([
            self.editor, self.beam_plane.widget, self.actions, self.status, self.progress_view, self.notice,
            W.HBox([self.source, self.field]), W.HBox([self.cut, self.aspect]),
            self.position, self.plot_message, self.canvas_host, self.plot, self.extra,
        ])
        self._colab_callback = None
        self._generation = None
        self._record = None
        self._updating_results = False
        self._preview = None
        self._preview_message = ""
        self._view_key = None
        for control in (self.source, self.field, self.cut):
            control.observe(self._update_selection, names="value")
        for control in (self.aspect, self.position):
            control.observe(self._draw, names="value")
        self._make_files()
        self._refresh_preview()
        self._sync()

    def _guard(self, generation, action):
        if self.session.disposed or generation != self.session.generation:
            return
        try:
            action()
        except Exception as error:
            self.notice.value = f"<b>{escape(type(error).__name__)}</b>: {escape(str(error))}"

    def _refresh_preview(self):
        try:
            self._preview = self.preview_factory(self.session.route, self.session.draft)
            self._preview_message = ""
        except (TypeError, ValueError) as error:
            self._preview = None
            self._preview_message = str(error)

    def _make_editor(self):
        session = self.session
        generation = session.generation
        request = session.draft
        route = session.route
        route_control = W.Dropdown(description="Workflow", options=[
            ("LC static", "lc"), ("PR reduced nonlinear TD", "pr")
        ], value=route)
        inputs = []

        def number(label, value, cls=W.FloatText):
            widget = cls(description=label, value=value,
                         style={"description_width": "initial"})
            inputs.append(widget)
            return widget

        if route == "lc":
            voltage = number("Applied voltage (V)", request.bias.V_bias)
            material_panel = W.VBox([voltage])
        else:
            material = {name: number(label, getattr(request.material, name))
                        for name, label in PR_MATERIAL_CONTROLS.items()}
            steps = number("Material steps", request.solver.Nt, W.IntText)
            timestep = number("Normalized dt", request.solver.dt_normalized)
            device = W.Dropdown(description="Device", options=[
                ("Auto", "auto"), ("CPU", "numpy"), ("GPU", "cupy"),
            ], value=request.backend.backend)
            inputs.append(device)
            material_panel = W.VBox([*material.values(), steps, timestep, device])
        grid = request.grid
        grid_summary = W.HTML(
            f"Fixed grid: {grid.Nx} × {grid.Ny}, aperture "
            f"{grid.x_aperture_um:g} × {grid.y_aperture_um:g} µm, "
            f"length {grid.z_length_um:g} µm, dz {grid.dz_um:g} µm."
            + (" CPU only." if route == "lc" else "")
        )

        def update():
            if session.busy:
                return
            draft = session.draft
            if route == "lc":
                draft = replace(draft, bias=replace(request.bias, V_bias=voltage.value))
            else:
                draft = replace(
                    draft,
                    material=replace(request.material, **{
                        name: widget.value for name, widget in material.items()}),
                    solver=replace(request.solver, Nt=steps.value, dt_normalized=timestep.value),
                    backend=replace(draft.backend, backend=device.value),
                )
            if session.set_draft(route, draft, generation):
                self.notice.value = ""
                self._refresh_preview()
                self._update_selection()

        def changed(_):
            self._guard(generation, update)

        def select_route(_):
            if session.set_draft(route_control.value, preset(route_control.value), generation):
                session.generation += 1
                self._refresh_preview()
                self._sync()

        def run():
            update()
            session.start(generation)

        button = W.Button(description="Run", button_style="primary")
        button.on_click(lambda _: self._guard(generation, run))
        reset = W.Button(description="Restore preset")

        def restore():
            if session.set_draft(route, preset(route), generation, reset_presentation=True):
                session.generation += 1
                self._refresh_preview()
                self._sync()

        reset.on_click(lambda _: self._guard(generation, restore))
        controls = (route_control, *inputs, button, reset)
        for control in controls:
            control.disabled = session.busy
        for control in inputs:
            control.observe(changed, names="value")
        route_control.observe(select_route, names="value")
        self.run_button = button
        old_controls = self.editor.children
        if self.fixed_route is None:
            workflow = route_control
        else:
            workflow = W.HTML(f"<b>{escape(route_control.label)}</b>")
            route_control.close()
        self.editor.children = (workflow, grid_summary, material_panel, W.HBox([button, reset]))
        for control in old_controls:
            close_widgets(control)

    def _make_files(self):
        session = self.session
        upload = W.FileUpload(accept=".json,.zip", multiple=False, disabled=session.busy)
        load = W.Button(description="Load / reopen", disabled=session.busy)
        save = W.Button(description="Download draft", disabled=session.busy)
        export = W.Button(description="Download completed")
        self.upload = upload
        self.load_button = load
        self.save_button = save
        self.export_button = export
        self.upload_receipt = W.HTML(value="No file received by the kernel.")

        def receipt(change):
            if not change["new"]:
                self.upload_receipt.value = "No file received by the kernel."
                return
            try:
                name, data = selected_upload(change["new"])
            except ValueError as error:
                self.upload_receipt.value = escape(str(error))
            else:
                self.upload_receipt.value = f"Received <b>{escape(name)}</b> ({len(data):,} bytes)."

        upload.observe(receipt, names="value")

        def editable(action):
            if session.busy:
                return
            self._guard(session.generation, action)

        def read():
            name, data = selected_upload(upload.value)
            self._open_uploaded(name, data)

        def download(path):
            # Colab transport is imported only on explicit download.
            try:
                from google.colab import files
            except ImportError:
                self.notice.value = f"Saved to <code>{escape(str(path))}</code>"
            else:
                files.download(str(path))

        def download_completed():
            path = session.export_completed(self.directory / f"completed-{uuid4().hex}")
            download(archive_completed(path))

        load.on_click(lambda _: editable(read))
        save.on_click(lambda _: editable(lambda: download(
            session.save_draft(self.directory / f"draft-{uuid4().hex}.lcprop.json"))))
        export.on_click(lambda _: self._guard(session.generation, download_completed))
        self.files.children = (upload, self.upload_receipt, W.HBox([load, save, export]))

    def _open_uploaded(self, name, data):
        self.session._editable()
        if name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory(prefix="reopen-", dir=self.directory) as staging:
                self.session.reopen(unpack_completed(data, Path(staging)))
            self.notice.value = f"Reopened <b>{escape(name)}</b>."
            return
        if not name.lower().endswith(".json"):
            raise ValueError("Select an experiment JSON or completed ZIP")
        path = self.directory / f"upload-{uuid4().hex}.lcprop.json"
        path.write_bytes(data)
        self.session.load_draft(path)
        self._refresh_preview()
        self._sync()
        self.notice.value = f"Loaded <b>{escape(name)}</b>."

    def open_from_colab(self):
        """Run in a notebook cell to use Colab's native blocking picker."""
        self.session._editable()
        from google.colab import files

        with tempfile.TemporaryDirectory(dir=self.directory) as staging:
            uploaded = files.upload(target_dir=staging)
            if not uploaded:
                return
            name, data = selected_upload({
                path: {"content": content} for path, content in uploaded.items()
            })
            self._open_uploaded(Path(name).name, data)

    def display_in_colab(self):
        """Display the controller and pull updates through Colab's browser callback."""
        from google.colab import output
        from IPython.display import JSON, Javascript

        if self._colab_callback is not None:
            output.unregister_callback(self._colab_callback)
        name = self._colab_callback = f"lcprop.status.{uuid4().hex}"

        def poll(last_state):
            session = self.session
            state = [session.generation, session.status]
            if state != last_state:
                self._refresh_status(None)
                self.stop_button.send_state("disabled")
                for control in (*self.beam_plane.inputs, self.upload, self.load_button, self.save_button,
                                self.export_button):
                    control.send_state("disabled")
            elif session.busy:
                self.progress_view.send_state("value")
            return JSON({"state": state, "busy": session.busy})

        output.register_callback(name, poll)
        display(self.widget)
        display(Javascript("""
(() => {
  const mount = document.createElement('div');
  element.appendChild(mount);
  let state = null;
  async function poll() {
    if (!mount.isConnected) return;
    try {
      const response = await google.colab.kernel.invokeFunction(CALLBACK, [state], {});
      const update = response.data['application/json'];
      state = update.state;
      setTimeout(poll, update.busy ? 500 : 1500);
    } catch (error) {
      if (mount.isConnected) {
        mount.textContent = 'Automatic progress connection ended. Redisplay the notebook controls to reconnect.';
      }
    }
  }
  setTimeout(poll, 0);
})();
""".replace("CALLBACK", json.dumps(name))))


    def _refresh_status(self, _):
        """Resend state from a live frontend callback after a missed update."""
        if self.session.disposed:
            return
        self._generation = None
        self._sync()
        self.status.send_state("value")
        self.progress_view.send_state("value")
        for control in (self.editor, self.actions, self.files, self.source):
            control.send_state()

    def _sync(self):
        session = self.session
        self.status.value = f"<b>{escape(session.status)}</b> {escape(session.message)}"
        if session.last_run_id is None:
            self.progress_view.value = "CPU · No admitted run" if session.route == "lc" else "No admitted run"
        else:
            done, total, stage = session.progress or (None, None, "")
            stage = stage or ("Preparing / executing" if session.busy else session.status)
            known = done is not None and total is not None and 0 <= done <= total and total > 0
            meter = ""
            if known:
                meter = f'<progress max="{total}" value="{done}"></progress> {done}/{total}'
            elif session.busy:
                meter = "<progress></progress>"
            stop = {"stopping": " · Stop requested", "stopped": " · Stop acknowledged"}
            device = session.last_device
            label = f"Auto → {device.label}" if device.requested == "auto" else device.label
            fallback = f"<br>{escape(device.fallback_reason)}" if device.fallback_reason else ""
            self.progress_view.value = (
                f"Run {session.last_run_id} · {escape(label)} · {session.elapsed:.1f} s"
                f"{stop.get(session.status, '')}<br>{escape(stage)} {meter}{fallback}"
            )
        changed = self._generation != session.generation
        if changed:
            self._generation = session.generation
            self._make_editor()
            self.beam_plane.sync()
            self.stop_button.close()
            self.stop_button = W.Button(description="Stop")
            generation = session.generation
            self.stop_button.on_click(lambda _: self._guard(generation, session.stop))
            self.actions.children = (self.stop_button, self.refresh_button)
        self.stop_button.disabled = session.status != "running"
        self.upload.disabled = session.busy
        self.load_button.disabled = session.busy
        self.save_button.disabled = session.busy
        self.export_button.disabled = session.completed is None
        records = (session.completed, session.partial)
        if changed or self._record != tuple(id(record) for record in records):
            completed_changed = session.completed is not None and (
                self._record is None or self._record[0] != id(session.completed))
            self._record = tuple(id(record) for record in records)
            options = ["Draft preview"]
            if session.completed is not None:
                options.append("Completed")
            if session.partial is not None:
                options.append("Stopped / partial")
            self._updating_results = True
            try:
                old = self.source.value
                self.source.options = options
                self.source.value = "Completed" if completed_changed else (
                    old if old in options else options[0])
            finally:
                self._updating_results = False
            self._update_selection()

    def _products(self):
        if self.source.value == "Draft preview":
            fields = {} if self._preview is None else {self._preview.key: self._preview}
            return fields, None, None
        record = (self.session.completed if self.source.value == "Completed"
                  else self.session.partial)
        if record is None or record.products is None:
            return {}, None, record
        return dict(record.products.fields.items()), record.products.geometry, record

    def _update_selection(self, _=None):
        if self._updating_results or self.session.disposed:
            return
        self._updating_results = True
        try:
            fields, geometry, record = self._products()
            options = [(field.display_name, key) for key, field in fields.items()
                       if getattr(field.data, "ndim", 0) in (2, 3)]
            if self.fixed_route == "lc" and self.source.value == "Completed":
                options = [(label if key in fields else f"{label} (unavailable)", key)
                           for key, label in LC_VOLUME_FIELDS.items()] + [
                    (label, key) for label, key in options if key not in LC_VOLUME_FIELDS
                ]
            old = self.field.value
            self.field.options = options
            keys = [value for _, value in options]
            self.field.value = old if old in keys else (keys[0] if keys else None)
            field = fields.get(self.field.value)
            missing = set(field.axes) - set(PLANES[self.cut.value]) if field else set()
            coordinate = None
            axis = None
            if field is not None and field.data.ndim == 3 and len(missing) == 1:
                try:
                    axis = next(iter(missing))
                    coordinate, unit = field_coordinates(field, geometry, axis)
                except ValueError:
                    pass
            key = (self.source.value, self.field.value, self.cut.value, id(record))
            index = 0 if coordinate is None or self.cut.value == "x–y" else len(coordinate) // 2
            self.position.disabled = coordinate is None
            if self.fixed_route == "lc":
                previous = self.position.value
                self.position.description = f"{axis} cut" if coordinate is not None else "Cut"
                self.position.layout.display = "" if coordinate is not None else "none"
                self.position.options = [] if coordinate is None else [
                    (f"{value:.6g} {unit}", i) for i, value in enumerate(coordinate)
                ]
                if coordinate is not None:
                    self.position.value = previous if key == self._view_key else index
            else:
                self.position.max = 0 if coordinate is None else len(coordinate) - 1
                if key != self._view_key:
                    self.position.value = index
            self._view_key = key
            self._show_identity(record)
        finally:
            self._updating_results = False
        self._draw()

    def _show_identity(self, record):
        if record is None:
            self.details.value = "Draft preview. No executed request for this view."
            return
        diagnostics = ""
        if record.products is not None:
            diagnostics = "\n".join(
                f"{name}.{key}: {value}"
                for name, diagnostic in record.products.diagnostics.items()
                for key, value in diagnostic.values.items()
                if isinstance(value, (str, float, int, bool))
            )
        metadata = record.provenance or {}
        if record.device is not None:
            metadata = {**metadata, "device": record.device.metadata()}
        provenance = json.dumps(metadata, indent=2)
        self.details.value = (
            f"<b>{escape(self.source.value)} · {record.run_id}</b>"
            f"<p>Request SHA-256: {record.digest}</p>"
            f"<pre>{escape(provenance)}\n{escape(diagnostics)}\n{escape(record.request_json)}</pre>"
        )

    def _draw(self, _=None):
        if self._updating_results or self.session.disposed:
            return
        fields, geometry, _ = self._products()
        field = fields.get(self.field.value)
        try:
            if self.source.value == "Draft preview" and self.cut.value != "x–y":
                raise ValueError("The entrance preview has XY data only. Select a completed result for longitudinal cuts.")
            if field is None:
                label = LC_VOLUME_FIELDS.get(self.field.value)
                if label is not None:
                    raise ValueError(f"The completed result has no retained {label} volume.")
                raise ValueError(self._preview_message or "No retained field")
            plane = retained_plane(field, geometry, PLANES[self.cut.value],
                                   None if self.position.disabled else self.position.value)
            fig = self._figure
            if fig is None:
                with plt.ioff():
                    size = (8, 5) if self.fixed_route == "lc" else (6, 4)
                    fig = plt.figure(figsize=size, layout="constrained")
                if isinstance(fig.canvas, W.Widget):
                    self._figure = fig
                    self.canvas_host.children = (fig.canvas,)
            try:
                fig.clear()
                ax = fig.add_subplot()
                artist = draw_plane(ax, plane, self.aspect.value == "Square")
                fig.colorbar(artist, ax=ax, label=plane.value_unit)
                self.plot_message.value = ""
                self.canvas_host.layout.display = ""
                if self._figure is not None:
                    fig.canvas.draw_idle()
                else:
                    with self.plot:
                        self.plot.clear_output(wait=True)
                        display(fig)
            finally:
                if self._figure is None:
                    plt.close(fig)
        except ValueError as error:
            self.canvas_host.layout.display = "none"
            self.plot.clear_output()
            self.plot_message.value = f"Unavailable: {escape(str(error))}"

    async def dispose(self):
        if self._colab_callback is not None:
            from google.colab import output

            output.unregister_callback(self._colab_callback)
            self._colab_callback = None
        self.beam_plane.close()
        close_widgets(self.widget)
        self.session.notify = lambda: None
        if getattr(self, "_figure", None) is not None:
            plt.close(self._figure)
        await self.session.dispose()


class LCNotebook(InteractiveBatch):
    """LC static Colab entry controller."""

    fixed_route = "lc"


class PRNotebook(InteractiveBatch):
    """PR reduced nonlinear TD Colab entry controller."""

    fixed_route = "pr"
