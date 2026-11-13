"""Interactive notebooks for LC and reduced PR workflows."""
from collections.abc import Mapping
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import ipywidgets as W
from IPython.display import display
from matplotlib import pyplot as plt

from lcprop.notebook.batch_beam_plane import BatchBeamPlane
from lcprop.notebook.batch_files import archive_completed, save_verified_copy, selected_upload, unpack_completed
from lcprop.notebook.batch_requests import (
    PR_MATERIAL_CONTROLS, material_for, preset, preview, validate,
)
from lcprop.core.grid import grid_metadata
from lcprop.notebook.batch_resources import MIB, estimate, require_work
from lcprop.notebook.batch_results import draw_plane, field_coordinates, retained_plane
from lcprop.notebook.batch_session import BatchSession

LC_VOLUME_FIELDS = {"intensity_stack": "Intensity", "delta_theta_stack": "Director response (Δθ)"}

PR_SPACE_CHARGE_SUFFIXES = {
    "initial_E": "XY slice",
    "final_E": "XY slice",
    "initial_E_stack": "Volume",
    "final_E_stack": "Volume",
}

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

    def __init__(self, *, runner=None, validator=validate, preview_factory=preview, gpu_probe=None, **session_options):
        self.preview_factory = preview_factory
        self.directory = Path(tempfile.mkdtemp(prefix="lcprop-notebook-"))
        self.session = BatchSession(runner=runner, validator=validator, notify=self._sync,
                                    fixed_route=self.fixed_route, gpu_probe=gpu_probe,
                                    **session_options)
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
        self.position = W.Dropdown(description="Cut")
        self.plot = W.Output()
        self.canvas_host = W.VBox()
        self.canvas_host.layout.width = "100%"
        self.canvas_host.layout.min_width = "0"
        self.canvas_host.layout.overflow = "auto"
        self.canvas_host.layout.align_items = "flex-start"
        self.plot.layout.overflow = "auto"
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
        self._route_views = {}
        self._view_route = self.session.route
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
            self.session.validator(self.session.route, self.session.draft)
            self.session.draft.grid.validate()
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
        labels = {
            "lc": "LC static", "lc_td": "LC time dependent",
            "pr_static": "PR static — reduced x-only",
            "pr": "PR time dependent — reduced x-only",
        }
        lc = material_for(route) == "lc"
        route_control = W.Dropdown(description="Workflow", options=[
            (label, key) for key, label in labels.items()
            if self.fixed_route is None or material_for(key) == self.fixed_route
        ], value=route)
        inputs = []

        def number(label, value, cls=W.FloatText):
            widget = cls(description=label, value=value,
                         style={"description_width": "initial"})
            inputs.append(widget)
            return widget

        grid_labels = {
            "Nx": "Nx", "Ny": "Ny", "x_aperture_um": "X aperture (µm)",
            "y_aperture_um": "Y aperture (µm)", "z_length_um": "Optical length (µm)",
            "dz_um": "Optical dz (µm)",
        }
        grid_inputs = {key: number(label, getattr(request.grid, key),
                                   W.IntText if key in ("Nx", "Ny") else W.FloatText)
                       for key, label in grid_labels.items()}
        solver_labels = {
            "lc": {"max_iterations": "Iteration limit",
                   "static_max_coupled_passes": "Coupled passes",
                   "static_residual_rms_tol": "Residual RMS tolerance",
                   "static_residual_max_tol": "Residual max tolerance"},
            "lc_td": {"Nt": "Material steps", "dt": "Normalized dt", "gamma_z": "Gamma z",
                      "max_picard_iter": "Picard iterations", "tolerance_update": "Update tolerance"},
            "pr_static": {"max_coupled_passes": "Coupled passes", "optical_substeps": "Optical substeps",
                          "residual_rms_tolerance": "Coupled RMS override",
                          "residual_max_tolerance": "Coupled max override"},
            "pr": {"Nt": "Material steps", "dt_normalized": "Normalized dt",
                   "optical_substeps": "Optical substeps"},
        }[route]
        solver_inputs = {}
        optional_tolerances = {"residual_rms_tolerance", "residual_max_tolerance",
                               "static_residual_rms_tol", "static_residual_max_tol"}
        for key, label in solver_labels.items():
            value = getattr(request.solver, key)
            if key in optional_tolerances:
                widget = W.Text(description=label, value="" if value is None else str(value),
                                placeholder="Engine default", style={"description_width": "initial"})
                inputs.append(widget)
            else:
                widget = number(label, value, W.IntText if isinstance(value, int) else W.FloatText)
            solver_inputs[key] = widget
        if route == "pr":
            integrator = W.Dropdown(description="Integrator", options=[
                ("Semi-implicit trapezoidal", "semi_implicit_trapezoidal"),
                ("Explicit Euler (reference)", "euler"),
            ], value=request.solver.integrator)
            solver_inputs["integrator"] = integrator
            inputs.append(integrator)
        if lc:
            voltage = number("Applied voltage (V)", request.bias.V_bias)
            material_panel = W.VBox([voltage])
        else:
            material = {name: number(label, getattr(request.material, name))
                        for name, label in PR_MATERIAL_CONTROLS.items()}
            options = [("CPU", "numpy"), ("Auto (CPU)", "auto")]
            if route == "pr" and (session.gpu_probe is not None or request.backend.backend == "cupy"):
                options.append(("Qualified T4 opt-in", "cupy"))
            if request.backend.backend not in dict(options).values():
                options.append(("Unsupported GPU intent", request.backend.backend))
            device = W.Dropdown(description="Device", options=options, value=request.backend.backend)
            inputs.append(device)
            material_panel = W.VBox([*material.values(), device])
        grid_summary = W.HTML()
        inspect = W.HTML()
        inspect_panel = disclosure("Inspect request", inspect)
        button = W.Button(description="Run", button_style="primary")
        reset = W.Button(description="Restore preset")
        copy_from = W.Dropdown(description="Copy grid from", options=[
            (label, key) for key, label in labels.items()
            if key != route and material_for(key) == material_for(route)])
        copy_grid = W.Button(description="Copy grid")
        release = W.Button(description="Release results in this workflow")

        def values(controls):
            return {key: widget.value for key, widget in controls.items()}

        def update_summary():
            try:
                draft = session.draft
                payload, warnings = session.validator(route, draft)
                meta = grid_metadata(draft.grid)
                charge = estimate(route, draft)
                grid = draft.grid
                packed = ("Unavailable" if charge.export is None else
                          f"{charge.export / MIB:.1f} MiB" +
                          (" (exceeds ZIP limit)" if charge.export > 128 * MIB else
                           " (large export)" if charge.export >= 64 * MIB else ""))
                warning = (" · High memory estimate" if charge.retained >= 192 * MIB
                           or charge.working + session.resident_bytes() >= 1024 * MIB else "")
                available = session.memory_available()
                available_text = "unknown" if available is None else f"{available / MIB:.1f} MiB"
                resident_results, resident_archives = session.resident_storage()
                grid_summary.value = (
                    f"dx {meta.dx_um:g}, dy {meta.dy_um:g} µm · Nz {meta.Nz} · "
                    f"Requested length {grid.z_length_um:g} µm · Marched {meta.Nz * grid.dz_um:g} µm · "
                    f"Preview at most {min(meta.Nx, 128)} × {min(meta.Ny, 128)} · "
                    f"Working {charge.working / MIB:.1f} MiB · Retained {charge.retained / MIB:.1f} MiB · "
                    f"Existing results {resident_results / MIB:.1f} MiB · "
                    f"Re-export archive bytes {resident_archives / MIB:.1f} MiB · ZIP {packed}{warning} · "
                    f"Runtime available {available_text} · Elapsed budget {120 if lc else 60} s (cooperative) · Runtime estimate unavailable"
                    + ("<br>" + escape("; ".join(warnings)) if warnings else "")
                )
                inspect.value = f"<pre>{escape(json.dumps(payload, indent=2))}</pre>"
                try:
                    session.require_admission(charge)
                except ValueError as error:
                    grid_summary.value += f"<br>{escape(str(error))}"
                    button.disabled = True
                else:
                    button.disabled = session.busy
                return True
            except (TypeError, ValueError, OverflowError) as error:
                grid_summary.value = escape(str(error))
                inspect.value = ""
                button.disabled = True
                return False

        def update():
            if session.busy:
                return
            try:
                solver_values = values(solver_inputs)
                for key in optional_tolerances & solver_values.keys():
                    text = solver_values[key].strip()
                    solver_values[key] = float(text) if text else None
                draft = replace(session.draft, grid=replace(session.draft.grid, **values(grid_inputs)),
                                solver=replace(session.draft.solver, **solver_values))
                if lc:
                    draft = replace(draft, bias=replace(draft.bias, V_bias=voltage.value))
                else:
                    draft = replace(draft, material=replace(draft.material, **values(material)),
                                    backend=replace(draft.backend, backend=device.value))
            except ValueError as error:
                button.disabled = True
                grid_summary.value = escape(str(error))
                self._preview = None
                self._preview_message = str(error)
                self._update_selection()
                return
            if session.set_draft(route, draft, generation):
                self.notice.value = ""
                if update_summary():
                    self._refresh_preview()
                else:
                    self._preview = None
                    self._preview_message = grid_summary.value
                self.beam_plane.draw()
                self._update_selection()
                return not button.disabled

        def transition(action):
            action()
            self._refresh_preview()
            self._sync()

        def restore():
            if session.set_draft(route, preset(route), generation, reset_presentation=True):
                session.generation += 1
                self._refresh_preview()
                self._sync()

        for control in (route_control, *inputs, button, reset, copy_from, copy_grid, release):
            control.disabled = session.busy
        for control in inputs:
            control.observe(lambda _: self._guard(generation, update), names="value")
        route_control.observe(lambda _: self._guard(generation, lambda: transition(
            lambda: session.switch(route_control.value))), names="value")
        button.on_click(lambda _: self._guard(generation, lambda: update() and session.start(generation)))
        reset.on_click(lambda _: self._guard(generation, restore))
        copy_grid.on_click(lambda _: self._guard(generation, lambda: transition(
            lambda: session.copy_grid(copy_from.value))))
        release.on_click(lambda _: self._guard(generation, lambda: transition(session.release_result)))
        self.run_button = button
        self.workflow_control = route_control
        self.grid_inputs = grid_inputs
        self.solver_inputs = solver_inputs
        self.material_panel = material_panel
        old_controls = self.editor.children
        guidance = ("LC time dependent starts from dark bias; Continue is unavailable."
                    if lc else "Full-transverse PR static and time dependent are unavailable in this batch.")
        self.editor.children = (
            route_control, W.HTML(guidance), disclosure("Grid", W.VBox([
                *grid_inputs.values(), W.HBox([copy_from, copy_grid])])), grid_summary,
            material_panel, disclosure("Solver", W.VBox(list(solver_inputs.values()))),
            inspect_panel, W.HBox([button, reset, release]),
        )
        self._update_editor_summary = update_summary
        update_summary()
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
        self.export_notice = W.HTML()
        drive_folder = W.Text(description="Drive folder", value="MyDrive")
        mount = W.Button(description="Connect Google Drive")
        drive_draft = W.Button(description="Save draft to Drive")
        drive_zip = W.Button(description="Save completed ZIP to Drive")
        self.drive_buttons = (drive_folder, mount, drive_draft, drive_zip)
        drive_root = Path("/content/drive")
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
            if name.lower().endswith(".zip"):
                # ipywidgets 7 uses a mapping and a change counter; 8 uses a tuple.
                upload.value = {} if isinstance(upload.value, Mapping) else ()
                if hasattr(upload, "_counter"):
                    upload._counter = 0

        def download(path):
            # Colab transport is imported only on explicit download.
            try:
                from google.colab import files
            except ImportError:
                self.notice.value = f"Saved to <code>{escape(str(path))}</code>"
            else:
                files.download(str(path))

        def draft_artifact():
            return session.save_draft(self.directory / f"draft-{uuid4().hex}.lcprop.json")

        def zip_artifact():
            reason = session.export_unavailable()
            if reason:
                raise ValueError(reason)
            if shutil.disk_usage(self.directory).free < 2 * session.export_size(session.completed):
                raise OSError("Insufficient local storage for staging the ZIP")
            name = f"completed-{uuid4().hex}"
            with tempfile.TemporaryDirectory(dir=self.directory) as staging:
                path = session.export_completed(Path(staging) / name)
                archive = archive_completed(path)
                return archive.rename(self.directory / archive.name)

        def mount_drive():
            from google.colab import drive
            drive.mount(str(drive_root))

        def save_drive(artifact):
            if not (drive_root / "MyDrive").is_dir():
                raise ValueError("Connect Google Drive first")
            folder = (drive_root / drive_folder.value.strip()).resolve()
            if folder == drive_root or not folder.is_relative_to(drive_root.resolve()):
                raise ValueError("Choose a folder inside mounted Google Drive")
            if not folder.is_dir():
                raise ValueError("Choose an existing Google Drive folder")
            path = save_verified_copy(artifact(), folder)
            self.notice.value = f"Verified Drive copy: <code>{escape(str(path))}</code>"

        mount.on_click(lambda _: editable(mount_drive))
        drive_draft.on_click(lambda _: editable(lambda: save_drive(draft_artifact)))
        drive_zip.on_click(lambda _: editable(lambda: save_drive(zip_artifact)))

        load.on_click(lambda _: editable(read))
        save.on_click(lambda _: editable(lambda: download(draft_artifact())))
        export.on_click(lambda _: editable(lambda: download(zip_artifact())))
        self.files.children = (
            upload, self.upload_receipt, W.HBox([load, save, export]), self.export_notice,
            disclosure("Google Drive (optional)", W.VBox(list(self.drive_buttons))),
        )

    def _open_uploaded(self, name, data):
        self.session._editable()
        if name.lower().endswith(".zip"):
            require_work(len(data) + 128 * MIB, self.session.resident_bytes(),
                         self.session.memory_available())
            with tempfile.TemporaryDirectory(prefix="reopen-", dir=self.directory) as staging:
                self.session.reopen(unpack_completed(data, Path(staging)), upload_bytes=len(data))
            self._update_editor_summary()
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
            self.progress_view.value = "CPU · No admitted run" if material_for(session.route) == "lc" else "No admitted run"
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
        route_changed = self._view_route != session.route
        if route_changed:
            self._route_views[self._view_route] = (
                self.source.value, self.field.value, self.cut.value,
                self.aspect.value, self.position.value,
            )
            self._view_route = session.route
            self._refresh_preview()
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
        self.export_button.disabled = session.busy or bool(session.export_unavailable())
        self.export_button.tooltip = session.export_unavailable()
        self.export_notice.value = escape(session.export_unavailable())
        for control in self.drive_buttons:
            control.disabled = session.busy
        self.drive_buttons[-1].disabled = self.export_button.disabled
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
                saved_view = self._route_views.get(session.route) if route_changed else None
                old = saved_view[0] if saved_view else self.source.value
                self.source.options = options
                self.source.value = "Completed" if completed_changed and not saved_view else (
                    old if old in options else options[0])
            finally:
                self._updating_results = False
            self._update_selection()
            if saved_view:
                _, field, cut, aspect, position = saved_view
                if field in dict(self.field.options).values():
                    self.field.value = field
                self.cut.value = cut
                self.aspect.value = aspect
                if position in dict(self.position.options).values():
                    self.position.value = position

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
            if record is not None and material_for(record.route) == "pr":
                options = [
                    (f"{label} ({PR_SPACE_CHARGE_SUFFIXES[key]})"
                     if key in PR_SPACE_CHARGE_SUFFIXES else label, key)
                    for label, key in options
                ]
            if self.session.route == "lc" and self.source.value == "Completed":
                options = [(label if key in fields else f"{label} (unavailable)", key)
                           for key, label in LC_VOLUME_FIELDS.items()] + [
                    (label, key) for label, key in options if key not in LC_VOLUME_FIELDS
                ]
            old = self.field.value
            self.field.options = options
            keys = [value for _, value in options]
            self.field.value = old if old in keys else (keys[0] if keys else None)
            field = fields.get(self.field.value)
            self.cut.disabled = bool(field is not None and set(field.axes) == {"s_x", "s_y"})
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
            previous = self.position.value
            self.position.description = f"{axis} cut" if coordinate is not None else "Cut"
            self.position.layout.display = "" if coordinate is not None else "none"
            self.position.options = [] if coordinate is None else [
                (f"{value:.6g} {unit}", i) for i, value in enumerate(coordinate)
            ]
            if coordinate is not None:
                self.position.value = previous if key == self._view_key else index
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
            axes = ("s_x", "s_y") if set(field.axes) == {"s_x", "s_y"} else PLANES[self.cut.value]
            plane = retained_plane(field, geometry, axes,
                                   None if self.position.disabled else self.position.value)
            fig = self._figure
            if fig is None:
                with plt.ioff():
                    size = (8, 5) if self.fixed_route == "lc" else (6, 4)
                    fig = plt.figure(figsize=size, layout="constrained")
                if isinstance(fig.canvas, W.Widget):
                    self._figure = fig
                    fig.canvas.layout.min_width = f"{fig.bbox.width:g}px"
                    self.canvas_host.children = (fig.canvas,)
            try:
                fig.clear()
                ax = fig.add_subplot()
                square = self.aspect.value == "Square"
                artist = draw_plane(ax, plane, square)
                # Use the displayed physical ratio, including axis unit conversion.
                wide_plot = not square and ax.get_aspect() * ax.get_data_ratio() <= 1 / 3
                fig.colorbar(artist, ax=ax, label=plane.value_unit,
                             location="bottom" if wide_plot else "right")
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
    """LC static and time-dependent entry controller."""

    fixed_route = "lc"


class PRNotebook(InteractiveBatch):
    """Reduced PR static and time-dependent entry controller."""

    fixed_route = "pr"
