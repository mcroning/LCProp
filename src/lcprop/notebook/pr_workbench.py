"""Optional reduced PR view. Reuses public sessions, limits, codecs and plots."""
from html import escape
from importlib import resources
from pathlib import Path
import math
import textwrap
from traitlets import Unicode

import ipywidgets as W
import matplotlib.pyplot as plt

from .batch_beam_plane import BatchBeamPlane
from .batch_files import MAX_UPLOAD_BYTES
from .interactive_batch import PRNotebook, disclosure
from .pr_workbench_preview import editing_preview


class _NumberString(Unicode):
    def validate(self, obj, value):
        if isinstance(value, (float, int)):
            value = format(value, ".6g")
        return super().validate(obj, value)


class _BeamNumber(W.Text):
    # Text view provides display precision; the request keeps its full float.
    value = _NumberString("").tag(sync=True)


class _EditingBeams(BatchBeamPlane):
    def _field_widget(self, label):
        return _BeamNumber(description=label, continuous_update=False,
                           style={"description_width": "initial"},
                           layout=W.Layout(width="100%"))

    def _numeric(self, name, value):
        def apply():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Enter a finite beam value; the previous value is unchanged.")
            self._replace(**{name: number})
        self._run(apply)

    def sync(self, *, redraw=True):
        super().sync(redraw=redraw)
        if not self.session.disposed:
            channel, _ = self._objects()[self._current(self._objects())]
            for name, widget in self.fields.items():
                widget.tooltip = f"Stored value: {getattr(channel, name)!r}. Display uses 6 significant digits."

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
        self.scan_button = W.Button(description="Refresh workbench files")
        self.open_button = W.Button(description="Open selected file")
        self.example_button = W.Button(description="Open synthetic example")
        for button in (self.scan_button, self.open_button, self.example_button,
                       self.save_button, self.export_button):
            button.layout = W.Layout(width="auto", min_width="180px", flex="1 1 180px")
        self.file_status = W.HTML()
        self.selected_field_help = W.HTML()
        self.scan_button.on_click(lambda _: self.scan_files())
        self.open_button.on_click(lambda _: self._guard(
            self.session.generation, lambda: self.open_path(self.saved_file.value)))
        self.example_button.on_click(lambda _: self._guard(
            self.session.generation, self.open_example))
        legacy_upload = W.VBox((self.upload, self.upload_receipt, self.load_button,
                                self.clear_upload_button), layout=W.Layout(display="none"))
        self.files.children = (
            W.HTML("<b>Saved experiments and results</b><br>1. Upload your JSON or result ZIP "
                   "using Colab’s left Files sidebar.<br>2. Click <b>Refresh workbench files</b> "
                   "here (the sidebar’s circular refresh does not update this list).<br>"
                   "3. Choose it under <b>Saved file</b>, then <b>Open selected file</b>. "
                   "Wait for sidebar upload progress to finish. No Drive mount is required."),
            self.example_button, self.saved_file, self.file_status,
            W.HBox((self.scan_button, self.open_button), layout=W.Layout(flex_flow="row wrap")),
            W.HBox((self.save_button, self.export_button), layout=W.Layout(flex_flow="row wrap")),
            self.export_notice, self.drive_panel, legacy_upload)
        beam = self.beam_plane
        beam.widget.children = (
            W.HTML("<b>BeamPlane · draft launch</b>"), beam.beam,
            W.HBox((beam.add, beam.duplicate, beam.delete)), beam.enabled, beam.coherence,
            beam.mode, beam.extent, beam.preview_status, beam.canvas_widget,
            disclosure("Beam values", W.VBox(tuple(beam.fields.values()))))
        for control in (self.source, self.field, self.cut, self.aspect, self.position):
            control.layout.width = "100%"
            control.style.description_width = "initial"
        results = W.VBox((self.scientific_status, self.source, self.field,
                         self.selected_field_help, self.cut, self.aspect,
                         self.position, self.plot_message, self.canvas_host, self.plot))
        self.sections = _Sections((beam.widget, results, self.files))
        self.sections.selector.observe(self._section_selected, names="value")
        self.widget.children = (
            W.HTML("<h3>PR workbench</h3>Reduced x-only Static and TD. CPU is the default. "
                   "Full-transverse material transport is unavailable in this workbench."),
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
        copy_row = parts["grid"].children[1].children[-1]
        copy_row.layout.flex_flow = "row wrap"
        copy_row.children[0].layout.width = "100%"
        copy_row.children[0].style.description_width = "initial"
        copy_row.children[0].description = "Grid source"
        copy_caption = W.HTML()
        def describe_copy_source(_=None):
            label = next((label for label, key in copy_row.children[0].options
                          if key == copy_row.children[0].value), "No other workflow")
            copy_caption.value = "<b>Grid source:</b> " + escape(label)
        copy_row.children[0].observe(describe_copy_source, names="value")
        describe_copy_source()
        copy_row.children = (*copy_row.children, copy_caption)
        release = parts["buttons"].children[0].children[-1]
        release.description = "Release workflow results"
        release.tooltip = "Discard retained results only in the selected workflow; saved files are unchanged."
        for button in parts["buttons"].children[0].children:
            button.layout = W.Layout(width="auto", min_width="140px", flex="1 1 140px")
        parts["buttons"].children[0].layout.flex_flow = "row wrap"
        device_help = W.HTML(
            "<b>CPU:</b> runs on this runtime’s processor; default for Static and TD.<br>"
            "<b>Auto:</b> checks the existing GPU rules on Run; if CPU fallback is needed, "
            "you review and confirm it before dispatch.<br>"
            "<b>GPU:</b> only the two exact qualified reduced TD requests (20 µm and 1 mm) "
            "on a separately verified T4 runtime are admitted. Static and edited/arbitrary "
            "requests are not GPU-qualified. This CPU notebook installs no CUDA software "
            "and does not enable GPU. Showing this help does not probe a GPU.")
        self.editor.children = (W.HBox((self.workflow_control, self.device_control),
                                     layout=W.Layout(flex_flow="row wrap")),
                                parts["buttons"], compact, device_help, self.advanced)

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
        if getattr(self, "_workbench_ready", False):
            fields, _, _ = self._products()
            field = fields.get(self.field.value)
            label = "" if field is None else field.display_name
            self.field.tooltip = label
            normalized = field is not None and field.quantity == "normalized_space_charge_field"
            self.selected_field_help.value = (
                "<b>Selected field:</b> " + escape(label) +
                ("<br>Normalized space-charge field (dimensionless); shown in the saved model’s "
                 "normalization, not V/m. Values are not rescaled for display." if normalized else ""))

    def _decorate_result_figure(self, fig, ax, field, plane):
        # Explicit short lines keep titles out of the vertical colorbar column.
        ax.set_title(textwrap.fill(plane.caption, width=28), fontsize=10, wrap=False)
        ax.tick_params(labelsize=9)
        for axis in fig.axes[1:]:
            axis.tick_params(labelsize=8)
        if field.quantity == "normalized_space_charge_field" and field.value_unit == "1":
            colorbar = ax.collections[0].colorbar
            if colorbar is not None:
                colorbar.set_label("Normalized field", fontsize=9)

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

    def _section_selected(self, change):
        if change["new"] == 2 and not self.session.busy:
            self.scan_files()

    def scan_files(self):
        previous = self.saved_file.value
        paths = sorted(p for p in self.files_root.iterdir()
                       if p.is_file() and p.suffix.lower() in {".json", ".zip"})
        self.saved_file.options = [("Choose a file", "")] + [(p.name, str(p)) for p in paths]
        if previous in {str(path) for path in paths}:
            self.saved_file.value = previous
        self.file_status.value = (
            f"{len(paths)} JSON/ZIP file(s) found. Missing upload? Wait for it to finish, "
            "then click <b>Refresh workbench files</b> here, not the sidebar refresh.")

    def open_path(self, path):
        if not path:
            raise ValueError("Upload in Colab’s Files sidebar, click Refresh workbench files here, then choose Saved file and Open selected file.")
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
