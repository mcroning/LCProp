"""Interactive CPU notebook for bounded LC static and reduced PR TD requests."""
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import ipywidgets as W
from IPython.display import display
from matplotlib import pyplot as plt

from lcprop.core.beams import BeamChannel
from lcprop.notebook.batch_files import archive_completed, selected_upload, unpack_completed
from lcprop.notebook.batch_requests import (
    BEAM_CONTROLS, PR_MATERIAL_CONTROLS, beam_stack, preset, preview, validate,
)
from lcprop.notebook.batch_results import draw_plane, field_coordinates, retained_plane
from lcprop.notebook.batch_session import BatchSession

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
    """Single controller per kernel. Run requires an active notebook event loop."""

    def __init__(self, *, runner=None, validator=validate, preview_factory=preview):
        self.preview_factory = preview_factory
        self.directory = Path(tempfile.mkdtemp(prefix="lcprop-notebook-"))
        self.session = BatchSession(runner=runner, validator=validator, notify=self._sync)
        self.editor = W.VBox()
        self.status = W.HTML()
        self.notice = W.HTML()
        self.stop_button = W.Button(description="Stop", disabled=True)
        self.source = W.Dropdown(description="View", options=["Draft preview"])
        self.field = W.Dropdown(description="Field")
        self.cut = W.Dropdown(description="Plane", options=list(PLANES))
        self.aspect = W.Dropdown(description="Aspect", options=["Physical", "Square"])
        self.position = W.IntSlider(description="Cut sample", continuous_update=False)
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
            self.editor, self.actions, self.status, self.notice,
            W.HBox([self.source, self.field]), W.HBox([self.cut, self.aspect]),
            self.position, self.plot_message, self.canvas_host, self.plot, self.extra,
        ])
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
        count = W.Dropdown(description="Beams", options=[1, 2],
                           value=len(request.beams.channels))
        coherence = W.Dropdown(description="Coherence", options=[
            ("Preserve saved groups", None), ("Coherent", "coherent"),
            ("Incoherent", "incoherent"),
        ])
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
            material_panel = W.VBox([*material.values(), steps, timestep])
        channels = list(request.beams.channels)
        if len(channels) == 1:
            names = {channel.name for channel in channels}
            editor = session.presentation.get("beam_editor")
            if editor is not None:
                names.update(beam["name"] for beam in editor["beam_stack"]["beams"])
            number_suffix = 2
            while f"beam-{number_suffix}" in names:
                number_suffix += 1
            name = f"beam-{number_suffix}"
            channels.append(BeamChannel(name=name, x0_um=20., w1_um=10., w2_um=10.,
                                        coherence_group="beam-2"))
        beam_inputs = [
            {name: number(label, getattr(channel, name))
             for name, label in BEAM_CONTROLS.items()} for channel in channels
        ]
        second = W.VBox(list(beam_inputs[1].values()))
        second.layout.display = "" if count.value == 2 else "none"
        beams = W.Accordion(children=[W.VBox(list(beam_inputs[0].values())), second])
        beams.set_title(0, "Beam 1")
        beams.set_title(1, "Beam 2")
        beams.selected_index = None
        grid = request.grid
        grid_summary = W.HTML(
            f"Fixed grid: {grid.Nx} × {grid.Ny}, aperture "
            f"{grid.x_aperture_um:g} × {grid.y_aperture_um:g} µm, "
            f"length {grid.z_length_um:g} µm, dz {grid.dz_um:g} µm. CPU only."
        )

        def update():
            if session.busy:
                return
            selected = [replace(channel, **{name: widget.value for name, widget in controls.items()})
                        for channel, controls in zip(channels[:count.value], beam_inputs)]
            draft = replace(request, beams=beam_stack(request.beams, selected, coherence.value))
            if route == "lc":
                draft = replace(draft, bias=replace(request.bias, V_bias=voltage.value))
            else:
                draft = replace(
                    draft,
                    material=replace(request.material, **{
                        name: widget.value for name, widget in material.items()}),
                    solver=replace(request.solver, Nt=steps.value, dt_normalized=timestep.value),
                )
            if session.set_draft(route, draft, generation):
                self.notice.value = ""
                self._refresh_preview()
                self._update_selection()

        def changed(_):
            second.layout.display = "" if count.value == 2 else "none"
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
            if session.set_draft(route, preset(route), generation):
                session.generation += 1
                self._refresh_preview()
                self._sync()

        reset.on_click(lambda _: self._guard(generation, restore))
        controls = (route_control, count, coherence, *inputs, button, reset)
        for control in controls:
            control.disabled = session.busy
        for control in (*inputs, count, coherence):
            control.observe(changed, names="value")
        route_control.observe(select_route, names="value")
        self.run_button = button
        old_controls = self.editor.children
        self.editor.children = (route_control, grid_summary, material_panel, count,
                                coherence, beams, W.HBox([button, reset]))
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
            if name.lower().endswith(".zip"):
                session.reopen(unpack_completed(data, self.directory))
                self.notice.value = f"Reopened <b>{escape(name)}</b>."
                return
            path = self.directory / f"upload-{uuid4().hex}.lcprop.json"
            path.write_bytes(data)
            session.load_draft(path)
            self._refresh_preview()
            self._sync()

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

    def _refresh_status(self, _):
        """Resend state from a live frontend callback after a missed update."""
        if self.session.disposed:
            return
        self._generation = None
        self._sync()
        self.status.send_state("value")
        for control in (self.editor, self.actions, self.files, self.source):
            control.send_state()

    def _sync(self):
        session = self.session
        progress = ""
        if session.progress:
            done, total, message = session.progress
            progress = f" · {done}/{total} · {escape(message)}"
        self.status.value = f"<b>{escape(session.status)}</b>{progress} {escape(session.message)}"
        changed = self._generation != session.generation
        if changed:
            self._generation = session.generation
            self._make_editor()
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
            old = self.field.value
            self.field.options = options
            keys = [value for _, value in options]
            self.field.value = old if old in keys else (keys[0] if keys else None)
            field = fields.get(self.field.value)
            missing = set(field.axes) - set(PLANES[self.cut.value]) if field else set()
            coordinate = None
            if field is not None and field.data.ndim == 3 and len(missing) == 1:
                try:
                    coordinate, _ = field_coordinates(field, geometry, next(iter(missing)))
                except ValueError:
                    pass
            self.position.disabled = coordinate is None
            self.position.max = 0 if coordinate is None else len(coordinate) - 1
            key = (self.source.value, self.field.value, self.cut.value)
            if key != self._view_key:
                self.position.value = 0 if coordinate is None or self.cut.value == "x–y" else len(coordinate) // 2
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
        provenance = json.dumps(record.provenance or {}, indent=2)
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
            if field is None:
                raise ValueError(self._preview_message or "No retained field")
            if self.source.value == "Draft preview" and self.cut.value != "x–y":
                raise ValueError("Longitudinal intensity is awaiting propagation results")
            plane = retained_plane(field, geometry, PLANES[self.cut.value],
                                   None if self.position.disabled else self.position.value)
            fig = self._figure
            if fig is None:
                with plt.ioff():
                    fig = plt.figure(figsize=(6, 4), layout="constrained")
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
        close_widgets(self.widget)
        self.session.notify = lambda: None
        if getattr(self, "_figure", None) is not None:
            plt.close(self._figure)
        await self.session.dispose()
