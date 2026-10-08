from __future__ import annotations

from datetime import datetime
from lcprop.gui.runtime_status import RuntimeStatusLabel
from lcprop.gui.convergence_summary import member_explanations
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lcprop.products.data_model import FieldCollection
from lcprop.gui.views import ImagePane, LongitudinalPane, CurvePane
from lcprop.gui.views.table_pane import TablePane
from lcprop.gui.views.display_scale import DisplayScales


class Workspace(QWidget):
    """Workflow-independent results workspace."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.operation_status = RuntimeStatusLabel("Idle")
        layout.addWidget(self.operation_status)
        self.result_ownership = RuntimeStatusLabel("No displayed result")
        layout.addWidget(self.result_ownership)
        self._attempt = 0
        self._displayed_attempt = None
        self._request_text = ""
        self._displayed_request = ""
        self._display_state = ""

        self.convergence_box = QWidget()
        convergence_layout = QHBoxLayout(self.convergence_box)
        convergence_layout.setContentsMargins(0, 0, 0, 0)
        self.convergence_selector = QComboBox()
        self.convergence_selector.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.convergence_selector.setMinimumContentsLength(24)
        self.convergence_details = QPushButton("Convergence details")
        convergence_layout.addWidget(self.convergence_selector, 1)
        convergence_layout.addWidget(self.convergence_details)
        self.convergence_dialog = QDialog(self)
        self.convergence_dialog.setWindowTitle("Member convergence details")
        self.convergence_dialog.resize(660, 340)
        details_layout = QVBoxLayout(self.convergence_dialog)
        self.convergence_ownership = RuntimeStatusLabel("No displayed result")
        details_layout.addWidget(self.convergence_ownership)
        self.convergence_text = QTextEdit()
        self.convergence_text.setReadOnly(True)
        details_layout.addWidget(self.convergence_text)
        self.convergence_details.clicked.connect(self.convergence_dialog.show)
        self._member_explanations = {}
        self.convergence_selector.currentIndexChanged.connect(self._show_member_explanation)
        layout.addWidget(self.convergence_box)
        self.convergence_box.hide()

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.display_scales = DisplayScales()
        self.image_pane = ImagePane(self.display_scales)
        self.longitudinal_display_scales = self.display_scales
        self.longitudinal_pane = LongitudinalPane(self.longitudinal_display_scales)

        self.spatial_selector = QComboBox()
        self.spatial_selector.setMinimumContentsLength(28)
        self.spatial_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.spatial_selector.setAccessibleName('Physical field or independent product')
        self._explicit_spatial_selection = None
        self.spatial_selector.activated.connect(self._remember_spatial_selection)
        self.spatial_selector.currentIndexChanged.connect(self._select_spatial_product)
        self.linked_log = QCheckBox('Log color scale — linked nonnegative field')
        self.linked_log.setToolTip('Display only. Zeros are masked; the automatic log range spans up to eight decades below the global maximum.')
        self.linked_log.toggled.connect(self._apply_linked_norm)
        self.image_pane.fieldChanged.connect(self._apply_linked_norm)
        self.longitudinal_pane.cutChanged.connect(self._apply_linked_norm)
        self.longitudinal_pane.zPlaneChanged.connect(self._apply_linked_norm)
        self.display_scales.changed.connect(self._apply_linked_norm)
        self.image_pane.field_selector.hide()
        self.longitudinal_pane.field_selector.hide()
        self.longitudinal_pane.field_selector_label.hide()
        self.fields_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.fields_splitter.addWidget(self.image_pane)
        self.fields_splitter.addWidget(self.longitudinal_pane)
        self.fields_splitter.setChildrenCollapsible(False)
        self.fields_splitter.setStretchFactor(0, 3)
        self.fields_splitter.setStretchFactor(1, 4)
        self.fields_splitter.setSizes([420, 560])
        self.fields_page = QWidget()
        fields_layout = QVBoxLayout(self.fields_page)
        fields_layout.addWidget(self.spatial_selector)
        fields_layout.addWidget(self.linked_log)
        fields_layout.addWidget(self.fields_splitter, 1)
        self.fields_scroll = QScrollArea()
        self.fields_scroll.setWidgetResizable(True)
        self.fields_scroll.setWidget(self.fields_page)
        self.tabs.addTab(self.fields_scroll, "Fields")

        self.image_pane.physicalPositionSelected.connect(
            self._image_position_selected
        )
        self.longitudinal_pane.cutChanged.connect(
            self._longitudinal_cut_changed
        )
        self.longitudinal_pane.guidesVisibilityChanged.connect(
            self._guides_visibility_changed
        )
        self.longitudinal_pane.zPlaneChanged.connect(
            self.image_pane.set_z_index
        )
        self.longitudinal_pane.guidesChanged.connect(self._sync_physical_guides)
        self.image_pane.fieldChanged.connect(self._sync_physical_guides)
        self.image_pane.sourceVolumeSelected.connect(
            self.longitudinal_pane.select_volume
        )
        self.longitudinal_pane.volumeSelectionChanged.connect(
            self.image_pane.select_source_volume
        )

        from lcprop.gui.trajectory_player import TrajectoryPlayer
        self.trajectory_player = TrajectoryPlayer(self)
        layout.addWidget(self.trajectory_player)

        self.curve_pane = CurvePane()
        self.tabs.addTab(self.curve_pane, "Curves")

        self.diagnostics_view = QTextEdit()
        self.diagnostics_view.setReadOnly(True)
        self.tabs.addTab(self.diagnostics_view, "Diagnostics")

        self.request_summary = QTextEdit()
        self.request_summary.setReadOnly(True)
        self.tabs.addTab(self.request_summary, "Request")

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.tabs.addTab(self.console, "Console")

        self.table_pane = TablePane()
        self.tabs.addTab(self.table_pane, "Samples / Tables")

        self.open_td_preview = QPushButton("Open downsampled TD preview")
        self.open_td_preview.setToolTip(
            "Open the visualization-only material-time MP4 preview"
        )
        self.open_td_preview.clicked.connect(self._open_td_preview)
        self.open_td_preview.hide()
        layout.addWidget(self.open_td_preview)
        self._td_preview_artifact = None
        self._artifact_directory = None

    def begin_request(self, operation: str) -> None:
        """Identify a new attempt before construction can fail."""
        self._attempt += 1
        self._request_text = "Request construction pending"
        self.mark_previous()
        self.operation_boundary(operation)
        self.set_operation_status("Validating / Preparing")
        self._refresh_request_text()

    def _set_ownership_text(self, text):
        self.result_ownership.setText(text)
        self.convergence_ownership.setText(text)

    def mark_previous(self) -> None:
        if self._displayed_attempt is not None:
            self._set_ownership_text(
                f"Previous result ({self._display_state})"
            )

    def operation_boundary(self, operation: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self._append_console_text(f"[{stamp}] {operation}", boundary=True)

    def set_operation_status(self, text: str) -> None:
        self.operation_status.setText(f"Execution status: {text}")
        # Paint acknowledgement before synchronous GUI request preparation;
        # do not process arbitrary events/re-enter a Run slot.
        self.operation_status.repaint()

    def finish_attempt(self, state: str) -> None:
        self.set_operation_status(state)
        if (self._displayed_attempt == self._attempt
                and (state != "State at failure" or self._display_state.startswith("Current"))):
            self._display_state = state
            self._set_ownership_text(state)

    def _refresh_request_text(self) -> None:
        text = (f"Request {self._attempt}\n" if self._attempt else "") + self._request_text
        if self._displayed_request and self._displayed_request != self._request_text:
            text = (
                "Requested / preview configuration:\n" + text
                + "\n\nDisplayed result request "
                + str(self._displayed_attempt) + ":\n" + self._displayed_request
            )
        self.request_summary.setPlainText(text)

    def set_request_summary(self, text: str) -> None:
        self._request_text = text
        self._refresh_request_text()

    def append_console(self, text: str) -> None:
        self._append_console_text(text)

    def _append_console_text(self, text, *, boundary=False):
        # QTextEdit.append can inherit an HTML rule/block format. Construct each
        # block explicitly so ordinary logs cannot inherit boundary decoration.
        scroll = self.console.verticalScrollBar()
        follow_tail = scroll.value() >= scroll.maximum()
        cursor = QTextCursor(self.console.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        block = QTextBlockFormat()
        if boundary:
            block.setTopMargin(6)
            block.setBottomMargin(4)
        char = QTextCharFormat()
        char.setFontWeight(QFont.Weight.Bold if boundary else QFont.Weight.Normal)
        if not self.console.document().isEmpty():
            cursor.insertBlock(block, char)
        else:
            cursor.setBlockFormat(block)
            cursor.setCharFormat(char)
        cursor.insertText(text, char)
        if follow_tail:
            scroll.setValue(scroll.maximum())

    def set_td_time_indicator(self, text: str | None) -> None:
        # New-run progress must not relabel a retained previous field's time.
        if self._displayed_attempt is not None and self._displayed_attempt != self._attempt:
            return
        self.image_pane.set_td_time_indicator(text)

    def reset_field_color_scales(self) -> None:
        self.image_pane.reset_color_scales()
        self.longitudinal_display_scales.reset_locks()

    def _image_position_selected(self, x: float, y: float) -> None:
        self.longitudinal_pane.set_cut_coordinates(x, y)

    def _longitudinal_cut_changed(self, ix: int, iy: int) -> None:
        self._sync_physical_guides()

    def _guides_visibility_changed(self, visible: bool) -> None:
        self._sync_physical_guides()

    def _sync_physical_guides(self):
        view = self.image_pane.image_view
        coords = self.longitudinal_pane.guide_coordinates()
        if (coords is not None and not getattr(self.longitudinal_pane, "_independent", False)
                and self.longitudinal_pane.show_guides.isChecked()
                and view._field is not None and view._field.axes == ("x", "y")):
            pane = self.longitudinal_pane
            if (not pane._is_fixed_cut_selection()
                    and view._field.source_volume_key == pane.field_selector.currentData()):
                # A genuinely linked volume retains the established index API.
                view.set_crosshair(pane._ix, pane._iy)
            else:
                view.set_crosshair_coordinates(*coords[:2])
        else:
            view.clear_crosshair()

    def invalidate_products(self):
        """Drop rendered arrays at execution start, retaining saved results elsewhere."""
        from lcprop.products.data_model import RunData, Geometry
        self.set_run_data(RunData(workflow="pending", geometry=Geometry()),
                          state="Waiting for current result")
        self._displayed_attempt = None
        self._displayed_request = ""
        self._refresh_request_text()

    def set_run_data(self, run_data, *, state: str | None = None) -> None:
        # Publish ownership only after every pane has accepted the product.
        # Suppress painting while panes may contain a mixture of old/new data.
        self.setUpdatesEnabled(False)
        self.convergence_dialog.setUpdatesEnabled(False)
        try:
            summary = run_data.diagnostics.get("summary")
            values = {} if summary is None else summary.values
            if state is None:
                state = values.get("presentation_state") or (
                    "State at stop/cancellation"
                    if values.get("status") in {"stopped", "cancelled"}
                    else "Completed result"
                )
            self._render_run_data(run_data)
            self.longitudinal_pane.set_progress_state(state.startswith("Current") or state == "Waiting for current result")
        except Exception:
            # A partially updated product has no coherent displayed ownership.
            # Keep the selected tab, but make all result content unavailable.
            self.trajectory_player.pause()
            self.trajectory_player.artifact = None
            self.image_pane.hide()
            self.longitudinal_pane.hide()
            self.curve_pane.curve_view.hide()
            self.curve_pane.curve_selector.hide()
            self.diagnostics_view.clear()
            self.table_pane.clear()
            self._member_explanations = {}
            self.convergence_selector.clear()
            self.convergence_text.clear()
            self.convergence_box.hide()
            self.convergence_dialog.hide()
            self.open_td_preview.hide()
            self._td_preview_artifact = None
            self._displayed_attempt = None
            self._displayed_request = ""
            self._display_state = ""
            self._set_ownership_text("No displayed result — result update failed")
            self.result_ownership.setToolTip("")
            self._refresh_request_text()
            raise
        else:
            self._displayed_attempt = self._attempt
            self._displayed_request = self._request_text
            self._display_state = state
            self._set_ownership_text(state)
            self.result_ownership.setToolTip(self._displayed_request)
            self._refresh_request_text()
            self.image_pane.show()
            selection = self.spatial_selector.currentData()
            self.longitudinal_pane.setVisible(selection is None or selection[0] != 'plane')
            self.curve_pane.curve_selector.show()
        finally:
            self.convergence_dialog.setUpdatesEnabled(True)
            self.setUpdatesEnabled(True)

    def _apply_linked_norm(self, *_args):
        if not self.spatial_selector.isEnabled():
            return
        from matplotlib.colors import Normalize, LogNorm
        selection = self.spatial_selector.currentData()
        linked = selection is not None and selection[0] == 'volume'
        field = getattr(self, '_endpoint_view_data', None)
        field = None if field is None or not linked else field.fields.get(selection[1])
        eligible = False
        if field is not None:
            low, high = self.display_scales.global_limits(field)
            from lcprop.gui.views.display_scale import scale_key
            global_range = self.display_scales.limits(scale_key(field), (low, high))
            eligible = (field.kind in ('intensity', 'intensity_preview') or
                        field.quantity in ('carrier', 'carrier_density', 'n')) and low >= 0 and high > 0 and global_range[0] >= 0 and global_range[1] > 0
        self.linked_log.setEnabled(eligible)
        if not eligible:
            self.linked_log.blockSignals(True)
            self.linked_log.setChecked(False)
            self.linked_log.blockSignals(False)
        for view in (self.image_pane.image_view, self.longitudinal_pane.xz_view,
                     self.longitudinal_pane.yz_view):
            limits = global_range if field is not None else view.image.get_clim()
            if linked and eligible and self.linked_log.isChecked():
                from lcprop.gui.views.display_scale import scale_key
                limits = self.display_scales.limits(scale_key(field), (low, high))
                # Display range only; never floor or rewrite the scientific array.
                lo, hi = limits
                norm = LogNorm(max(lo, hi * 1e-8), hi) if hi > 0 else Normalize(*limits)
            else:
                norm = Normalize(*limits)
            view.image.set_norm(norm)
            view.draw_idle()

    def _select_spatial_product(self, _index=None):
        selection = self.spatial_selector.currentData()
        if selection is None:
            return
        mode, key = selection
        # Reset presentation transforms before ImageView installs a signed plane.
        from lcprop.gui.views.display_scale import reset_display_norm
        for view in (self.image_pane.image_view, self.longitudinal_pane.xz_view,
                     self.longitudinal_pane.yz_view):
            reset_display_norm(view)
        image, longitudinal = self.image_pane, self.longitudinal_pane
        if mode == 'volume':
            longitudinal.show()
            longitudinal.set_independent(False)
            longitudinal.select_volume(key)
            longitudinal._update_views()
            image.select_source_volume(key)
            image.set_z_index(longitudinal._iz)
            image.image_view.show()
            image.selection_message.setText('Linked spatial field: xy / xz / yz — same retained quantity.')
        elif mode == 'plane':
            image._z_index = None
            image.field_selector.setCurrentIndex(image.field_selector.findData(key))
            image._field_changed(image.field_selector.currentIndex())
            longitudinal.set_independent(True)
            longitudinal.hide()
            image.image_view.show()
            image.selection_message.setText('Independent plane product; no linked longitudinal slices.')
        else:
            longitudinal.show()
            longitudinal.set_independent(False)
            longitudinal.select_volume(key)
            longitudinal._update_views()
            image.image_view.clear_field()
            image.image_view.hide()
            image.scale_controls.setEnabled(False)
            image.selection_message.setText('Retained fixed cuts only. Matching arbitrary xy slice unavailable.')
        self._sync_physical_guides()
        self._apply_linked_norm()

    def _remember_spatial_selection(self, _index):
        self._explicit_spatial_selection = self.spatial_selector.currentData()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Stack readable canvases on narrow windows; scroll rather than clip.
        if hasattr(self, 'fields_splitter'):
            orientation = (Qt.Orientation.Vertical if self.width() < 1000
                           else Qt.Orientation.Horizontal)
            self.fields_splitter.setOrientation(orientation)

    def _populate_spatial_selector(self, view_data):
        from lcprop.gui.field_selection import spatial_registry
        previous = self.spatial_selector.currentData()
        available, unavailable = spatial_registry(view_data)
        self.spatial_selector.blockSignals(True)
        self.spatial_selector.clear()
        for key in available:
            f = view_data.fields[key]
            self.spatial_selector.addItem(f'Linked: {f.display_name} [{key}; {f.value_unit}]', ['volume', key])
        for key, reason in unavailable.items():
            self.spatial_selector.addItem(f'Unavailable: {key} — {reason}', ['unavailable', key])
            self.spatial_selector.model().item(self.spatial_selector.count()-1).setEnabled(False)
        for key, f in view_data.fields.items():
            if (getattr(f.data, 'ndim', None) == 2 and f.source_volume_key not in available
                    and not f.coordinates.get('paired_cut_key')):
                self.spatial_selector.addItem('Independent: ' + f.display_name, ['plane', key])
        for key in self.longitudinal_pane._fixed_cut_fields:
            self.spatial_selector.addItem('Fixed cuts: ' + self.longitudinal_pane.field_selector.itemText(
                self.longitudinal_pane.field_selector.findData(key)), ['cuts', key])
        preferred = self._explicit_spatial_selection
        if preferred is None and 'timedependent' in view_data.workflow:
            if 'optical_intensity_stack' in available:
                preferred = ['volume', 'optical_intensity_stack']
        index = self.spatial_selector.findData(preferred or previous)
        if index < 0:
            selected = self.image_pane.field_selector.currentData()
            index = self.spatial_selector.findData(['plane', selected])
        if index < 0:
            index = next((i for i in range(self.spatial_selector.count())
                          if self.spatial_selector.model().item(i).isEnabled()), -1)
        self.spatial_selector.setCurrentIndex(index)
        self.spatial_selector.blockSignals(False)
        self._select_spatial_product()

    def _render_run_data(self, run_data) -> None:
        self.trajectory_player.set_run_data(run_data)
        if self._artifact_directory is not None:
            self._artifact_directory.cleanup()
            self._artifact_directory = None
        self._td_preview_artifact = getattr(run_data, "artifacts", {}).get(
            "td_preview_movie"
        )
        self.open_td_preview.setVisible(self._td_preview_artifact is not None)
        if run_data.workflow == "timedependent":
            summary = run_data.diagnostics.get("summary")
            values = {} if summary is None else summary.values
            cumulative_time = values.get("cumulative_time")
            if cumulative_time is not None:
                prefix = (
                    "TD time at stop: "
                    if values.get("status") == "cancelled"
                    else "Final TD time: "
                )
                self.image_pane.set_td_time_indicator(
                    prefix + f"{float(cumulative_time):.3f}"
                )
            else:
                self.image_pane.set_td_time_indicator(None)
        elif run_data.workflow == "static":
            summary = run_data.diagnostics.get("summary")
            values = {} if summary is None else summary.values
            coordinate = values.get("z_reached_um")
            if coordinate is None:
                self.image_pane.set_td_time_indicator(None)
            else:
                prefix = (
                    "z at stop: "
                    if values.get("status") == "stopped"
                    else "Final z: "
                )
                completed = values.get("completed_slices")
                total = values.get("total_slices")
                self.image_pane.set_td_time_indicator(
                    prefix
                    + f"{float(coordinate):.3f} um; slices: {completed}/{total}"
                )
        else:
            self.image_pane.set_td_time_indicator(None)

        # Rename only view records; preserve original names in Diagnostics and
        # all scientific arrays, product keys, and persisted/transport records.
        view_data = replace(run_data, fields=FieldCollection([
            (key, replace(field, display_name=field.display_name.replace(
                "Final Authoritative Optical", "Optical"
            ).replace("Authoritative Optical", "Optical")))
            for key, field in run_data.fields.items()
        ]))
        from lcprop.gui.field_selection import with_transverse_aliases
        view_data = with_transverse_aliases(view_data)
        self._endpoint_view_data = view_data
        self.image_pane.set_run_data(view_data)
        selected_image = self.image_pane.field_selector.currentData()
        self.longitudinal_pane.set_run_data(view_data)
        # A volume's automatic default must not replace an independent selected
        # output/angular product during live -> completed publication.
        if selected_image is not None:
            self.image_pane.field_selector.setCurrentIndex(
                self.image_pane.field_selector.findData(selected_image))
        self._populate_spatial_selector(view_data)
        self._sync_physical_guides()
        self.curve_pane.set_run_data(run_data)
        self.table_pane.set_run_data(run_data)
        selected = self.convergence_selector.currentData()
        explanations = member_explanations(run_data)
        self._member_explanations = {identity: text for identity, _label, text in explanations}
        self.convergence_selector.blockSignals(True)
        self.convergence_selector.clear()
        for identity, label, _text in explanations:
            self.convergence_selector.addItem(label, identity)
        self.convergence_selector.setCurrentIndex(max(0, self.convergence_selector.findData(selected)))
        self.convergence_selector.blockSignals(False)
        self._show_member_explanation()
        self.convergence_box.setVisible(bool(explanations))
        if not explanations:
            self.convergence_dialog.hide()

        self.diagnostics_view.setPlainText(self._format_diagnostics(run_data))

    def _show_member_explanation(self, _index=None):
        label = self.convergence_selector.currentText()
        self.convergence_selector.setToolTip(label)
        text = self._member_explanations.get(self.convergence_selector.currentData(), "")
        self.convergence_text.setPlainText(label + "\n\n" + text if text else "")

    def _open_td_preview(self) -> None:
        artifact = self._td_preview_artifact
        if artifact is None:
            return
        self._artifact_directory = TemporaryDirectory(
            prefix="lcprop-pr-td-preview-"
        )
        path = Path(self._artifact_directory.name) / artifact.filename
        path.write_bytes(np.asarray(artifact.data, dtype=np.uint8).tobytes())
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _format_diagnostics(self, run_data) -> str:
        lines: list[str] = [f"Workflow: {run_data.workflow}"]

        for name, diagnostic in run_data.diagnostics.items():
            lines.append("")
            lines.append(f"[{name}] {diagnostic.display_name}")
            for key, value in diagnostic.values.items():
                if key == "rows" and name == "carrier_power" and value:
                    lines.append(
                        "Carrier | Input Power (normalized) | "
                        "Output Power (normalized) | Gain | "
                        "Delta Power (normalized)"
                    )
                    for row in value:
                        gain = row["gain"]
                        gain_text = "n/a" if gain is None else f"{gain:.6g}"
                        lines.append(
                            f"{row['carrier']} | "
                            f"{row['input_power_normalized']:.6g} | "
                            f"{row['output_power_normalized']:.6g} | "
                            f"{gain_text} | "
                            f"{row['delta_power_normalized']:.6g}"
                        )
                elif key == "rows":
                    lines.append(f"{key}: {len(value)} rows")
                else:
                    lines.append(f"{key}: {value}")

        if run_data.curves:
            lines.append("")
            lines.append("Curves:")
            for key, curve in run_data.curves.items():
                lines.append(f"  {key}: {curve.display_name}")

        if run_data.fields:
            lines.append("")
            lines.append("Fields:")
            for key, field in run_data.fields.items():
                shape = getattr(field.data, "shape", None)
                lines.append(f"  {key}: {field.display_name}, shape={shape}")

        return "\n".join(lines)
