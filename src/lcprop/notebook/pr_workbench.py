"""Optional reduced PR view. Reuses public sessions, limits, codecs and plots."""
from html import escape
from importlib import resources
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt

from .batch_beam_plane import BatchBeamPlane
from .batch_files import MAX_UPLOAD_BYTES
from .interactive_batch import PRNotebook, disclosure
from .pr_workbench_preview import editing_preview


class _EditingBeams(BatchBeamPlane):
    def draw(self):
        # A rejected scientific grid must not remove the independent editing image.
        if self.owner._preview is None:
            self.owner._refresh_preview()
        super().draw()
        self.preview_status.value += (
            "<br>Editing preview only — independently sampled. "
            "Run uses the scientific grid and its validation.")


class _Sections(W.VBox):
    """Plain widgets for Colab managers where nested Tab/Accordion views fail."""
    def __init__(self, panels):
        self.panels = tuple(panels)
        self.selector = W.ToggleButtons(options=[("Beams", 0), ("Results", 1), ("Files", 2)],
                                       value=0)
        super().__init__((self.selector, *self.panels))
        self.selector.observe(self._show, names="value")
        self._show()

    def _show(self, _=None):
        for index, panel in enumerate(self.panels):
            panel.layout.display = "" if index == self.selector.value else "none"


class PRWorkbench(PRNotebook):
    """Reduced Static/TD workbench; no new execution or resource policy."""
    beam_plane_type = _EditingBeams
    show_unavailable_routes = False

    def __init__(self, *, files_root=None, **options):
        self.files_root = Path(files_root) if files_root is not None else Path.cwd()
        if not self.files_root.is_dir():
            raise ValueError("Choose an existing session-files directory")
        self._workbench_ready = False
        options.setdefault("preview_factory", editing_preview)
        super().__init__(**options)
        self.scientific_status = W.HTML()
        self.refresh_button.description = "Refresh display"
        self.saved_file = W.Dropdown(description="Saved file", options=[("Choose a file", "")],
                                     layout=W.Layout(width="95%"))
        self.scan_button = W.Button(description="Refresh file list")
        self.open_button = W.Button(description="Open selected file")
        self.example_button = W.Button(description="Open synthetic example")
        self.scan_button.on_click(lambda _: self.scan_files())
        self.open_button.on_click(lambda _: self._guard(
            self.session.generation, lambda: self.open_path(self.saved_file.value)))
        self.example_button.on_click(lambda _: self._guard(
            self.session.generation, self.open_example))
        legacy_upload = W.VBox((self.upload, self.upload_receipt, self.load_button,
                                self.clear_upload_button), layout=W.Layout(display="none"))
        self.files.children = (
            W.HTML("<b>Saved experiments and results</b><br>In Colab, upload through the "
                   "Files sidebar, then Refresh file list and Open selected file. "
                   "No file picker or Drive connection opens automatically."),
            self.example_button, self.saved_file, W.HBox((self.scan_button, self.open_button)),
            W.HBox((self.save_button, self.export_button)), self.export_notice, self.drive_panel, legacy_upload)
        beam = self.beam_plane
        beam.widget.children = (
            W.HTML("<b>BeamPlane · draft launch</b>"), beam.beam,
            W.HBox((beam.add, beam.duplicate, beam.delete)), beam.enabled, beam.coherence,
            beam.mode, beam.extent, beam.preview_status, beam.canvas_widget,
            disclosure("Beam values", W.VBox(tuple(beam.fields.values()))))
        results = W.VBox((self.scientific_status,
                         W.HBox((self.source, self.field), layout=W.Layout(flex_flow="row wrap")),
                         W.HBox((self.cut, self.aspect), layout=W.Layout(flex_flow="row wrap")),
                         self.position, self.plot_message, self.canvas_host, self.plot))
        self.sections = _Sections((beam.widget, results, self.files))
        self.widget.children = (
            W.HTML("<h3>PR workbench</h3>Reduced x-only Static and TD. CPU is the default. "
                   "Full-transverse transport is not enabled here; existing GPU guards remain."),
            self.editor, self.actions, self.status, self.progress_view, self.notice,
            self.sections, disclosure("Request and diagnostics", self.details))
        self._workbench_ready = True
        self.scan_files()
        self._sync()

    def _make_editor(self):
        super()._make_editor()
        parts = self.editor_sections
        parts["guidance"].close()
        # Do not rewrite options after callbacks are attached: doing so resets
        # ipywidgets' selection and can silently switch the active draft.
        self.workflow_control.description = "Evolution"
        self.material_panel.children = tuple(
            child for child in self.material_panel.children if child is not self.device_control)
        self.advanced = disclosure("Advanced settings", W.VBox((
            parts["grid"], disclosure("Material", parts["material"]),
            parts["solver"], disclosure("Resource admission and limits", parts["summary"]),
            parts["inspect"])))
        compact = W.HTML()

        def summarize(_=None):
            grid = self.session.draft.grid
            compact.value = (f"Grid {grid.Nx} × {grid.Ny} · length {grid.z_length_um:g} µm · "
                             "60-second cooperative PR limit. Advanced settings show admission limits.")
        parts["summary"].observe(summarize, names="value")
        self.run_button.observe(summarize, names="disabled")
        summarize()
        self.editor.children = (W.HBox((self.workflow_control, self.device_control),
                                     layout=W.Layout(flex_flow="row wrap")),
                                parts["buttons"], compact, self.advanced)

    def _refresh_preview(self):
        try:
            self._preview = self.preview_factory(self.session.route, self.session.draft)
            self._preview_message = ""
        except (TypeError, ValueError) as error:
            self._preview = None
            self._preview_message = str(error)

    def _sync(self):
        super()._sync()
        if self._workbench_ready:
            for widget in (self.scan_button, self.open_button, self.example_button):
                widget.disabled = self.session.busy
            self._show_scientific_status()

    def _show_scientific_status(self):
        if not self._workbench_ready:
            return
        _, _, record = self._products()
        if record is None:
            text = "Editing preview — no executed result in this view."
        else:
            result = record.result
            if self.source.value == "Stopped / partial":
                state = "Stopped / partial — not a completed scientific result"
            elif record.route == "pr_static":
                state = ("Static converged" if getattr(result, "converged", False)
                         else "Static not converged — execution finished")
            else:
                state = (f"TD evolution: {getattr(result, 'completed_steps', '?')} steps completed "
                         "(not a global convergence claim)")
            text = f"Run {record.run_id} · {state}"
            if ((record.provenance or {}).get("synthetic")
                    or (record.provenance or {}).get("environment", {}).get("synthetic")):
                text += " · Synthetic example; not a propagation result"
        self.scientific_status.value = escape(text)

    def _update_selection(self, _=None):
        super()._update_selection(_)
        self._show_scientific_status()

    def _new_result_figure(self):
        # Fit the normal narrow Colab panel without changing physical aspect.
        with plt.ioff():
            return plt.figure(figsize=(3.6, 3.2), layout="constrained")

    def _draw(self, _=None):
        # Clear only this output model: clear_output inside a Colab callback may
        # otherwise erase the parent cell, including the workbench itself.
        output = self.plot
        original = output.clear_output
        output.clear_output = lambda *a, **k: setattr(output, "outputs", ())
        try:
            super()._draw(_)
            if self._figure is not None:
                self._figure.canvas.layout.min_width = "0"
                self._figure.canvas.layout.max_width = "100%"
                for axis in self._figure.axes:
                    axis.title.set_wrap(True)
                self._figure.canvas.draw_idle()
        finally:
            output.clear_output = original

    def _refresh_status(self, _=None):
        if self.session.disposed:
            return
        self._sync()
        self._update_selection()
        for control in (self.editor, self.actions, self.files, self.source, self.field,
                        self.cut, self.aspect, self.position, self.status, self.progress_view):
            control.send_state()

    def scan_files(self):
        previous = self.saved_file.value
        paths = sorted(p for p in self.files_root.iterdir()
                       if p.is_file() and p.suffix.lower() in {".json", ".zip"})
        self.saved_file.options = [("Choose a file", "")] + [(p.name, str(p)) for p in paths]
        if previous in {str(path) for path in paths}:
            self.saved_file.value = previous

    def open_path(self, path):
        if not path:
            raise ValueError("Upload in the Files sidebar, refresh the list, then select your file.")
        path = Path(path)
        if path.suffix.lower() not in {".json", ".zip"}:
            raise ValueError("Choose an experiment JSON or completed ZIP")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("File exceeds the unchanged 128 MiB upload limit")
        self._open_uploaded(path.name, path.read_bytes())
        if path.suffix.lower() == ".zip":
            self.sections.selector.value = 1
            self._update_selection()

    def open_example(self):
        """Open a tiny synthetic transport fixture; never compute a result."""
        data = resources.files("lcprop.notebook.assets").joinpath("synthetic_pr.zip").read_bytes()
        self._open_uploaded("synthetic_pr.zip", data)
        self.sections.selector.value = 1
        self._update_selection()
        self.notice.value = "Loaded synthetic example. No propagation was performed."

    def redisplay(self, *, hosted=True):
        if self.session.disposed:
            raise RuntimeError("Save needed files and restart before replacing a disposed workbench.")
        self._refresh_status()
        self.beam_plane.draw()
        if hosted:
            self.display_in_colab()
        else:
            from IPython.display import display
            display(self.widget)
        return self


def open_pr_workbench(namespace, *, hosted=True, **options):
    """Create once in a notebook namespace; redisplay without resetting drafts."""
    panel = namespace.get("lcprop_pr_workbench")
    if panel is not None:
        if not isinstance(panel, PRWorkbench):
            raise RuntimeError("An incompatible workbench exists. Save your work before restarting.")
    else:
        panel = PRWorkbench(**options)
        namespace["lcprop_pr_workbench"] = panel
    return panel.redisplay(hosted=hosted)
