"""Session-only display limits. Never writes to FieldData or scientific arrays."""
import math
from lcprop.gui.layout import FlowLayout
from lcprop.gui.number_format import format_number

import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


def scale_key(field):
    return (field.source_volume_key or field.key, field.kind, field.quantity, field.value_unit)


def volume_limits(*arrays):
    """Existing full-range Auto policy, with finite/constant-field handling."""
    bounds = []
    for array in arrays:
        values = np.asarray(array)
        finite = np.isfinite(values)
        # Avoid copying an entire finite scientific volume merely for limits.
        if not finite.all():
            values = values[finite]
        if values.size:
            bounds.append((float(values.min()), float(values.max())))
    if not bounds:
        return 0., 1.
    lo, hi = min(x[0] for x in bounds), max(x[1] for x in bounds)
    if lo == hi:
        padding = max(abs(lo) * 1e-12, 1e-15)
        lo, hi = lo - padding, hi + padding
    return lo, hi


class DisplayScales(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.settings = {}
        self.last = {}

    def limits(self, key, automatic):
        mode, fixed = self.settings.get(key, ("auto", None))
        limits = automatic if mode == "auto" else fixed
        if limits is None:
            limits = automatic
            self.settings[key] = (mode, limits)
        self.last[key] = limits
        return limits

    def configure(self, key, mode, limits=None):
        if mode not in {"auto", "fixed", "locked"}:
            raise ValueError("Unknown display scale mode")
        if mode != "auto":
            limits = self.last.get(key) if limits is None else limits
            if limits is not None:
                lo, hi = map(float, limits)
                if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    raise ValueError("Display limits must be finite with lower < upper")
                limits = lo, hi
        self.settings[key] = (mode, limits)
        self.changed.emit()

    def reset_locks(self):
        self.last.clear()
        self.settings = {key: (mode, None if mode == "locked" else limits)
                         for key, (mode, limits) in self.settings.items()}


class DisplayScaleControls(QWidget):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales
        self.key = None
        self._displayed_limits = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = FlowLayout()
        # Establish Qt ownership before adding temporary label/input groups.
        layout.addLayout(row)
        self.mode = QComboBox()
        for label, value in [("Auto each frame", "auto"), ("Manual limits", "fixed"), ("Lock scale across frames", "locked")]:
            self.mode.addItem(label, value)
        self.lower = QLineEdit()
        self.upper = QLineEdit()
        self.lower.setPlaceholderText("Minimum")
        self.upper.setPlaceholderText("Maximum")
        for entry in (self.lower, self.upper):
            entry.setMinimumWidth(100)
            entry.setFixedWidth(110)
        self._entry_values = None
        self._entry_text = None
        self.apply_button = QPushButton("Apply")
        self.mode.setAccessibleName("Display scale")
        row.addWidget(self.mode)
        for label, entry in (("Min", self.lower), ("Max", self.upper)):
            group = QWidget()
            pair = QHBoxLayout(group)
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(2)
            pair.addWidget(QLabel(label))
            pair.addWidget(entry)
            entry.setAccessibleName("Minimum" if label == "Min" else "Maximum")
            row.addWidget(group)
        row.addWidget(self.apply_button)
        self.message = QLabel("Display only; fixed/locked limits are shared by linked slices.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.apply_button.clicked.connect(self.apply_limits)
        self.setEnabled(False)

    def show_scale(self, key, limits):
        # Progress refreshes must not overwrite a manual range being edited.
        editing = key == self.key and (self.lower.isModified() or self.upper.isModified())
        self.key = key
        self._displayed_limits = limits
        self.setEnabled(True)
        mode = self.scales.settings.get(key, ("auto", None))[0]
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(mode))
        self.mode.blockSignals(False)
        if not editing:
            self._entry_values = tuple(limits)
            self._entry_text = tuple(format_number(v) for v in limits)
            for entry, text, value in zip((self.lower, self.upper), self._entry_text, limits):
                entry.setText(text)
                entry.setToolTip("Exact limit: " + format_number(value, exact=True))

    def _mode_changed(self, _index):
        if self.key is None:
            return
        # Lock the limits currently displayed here, not another linked pane's Auto.
        try:
            mode = self.mode.currentData()
            limits = None if mode == "auto" else self._displayed_limits
            self.lower.setModified(False)
            self.upper.setModified(False)
            self.scales.configure(self.key, mode, limits)
        except ValueError as exc:
            self.message.setText(str(exc))
            actual = self.scales.settings.get(self.key, ("auto", None))[0]
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(self.mode.findData(actual))
            self.mode.blockSignals(False)
        else:
            self.message.setText("Display only; fixed/locked limits are shared by linked slices.")

    def apply_limits(self):
        if self.key is None:
            return
        try:
            limits = tuple(
                self._entry_values[i]
                if self._entry_values is not None and not entry.isModified()
                and entry.text() == self._entry_text[i]
                else float(entry.text())
                for i, entry in enumerate((self.lower, self.upper))
            )
            if not (all(math.isfinite(v) for v in limits) and limits[0] < limits[1]):
                raise ValueError("Invalid display limits")
            self.lower.setModified(False)
            self.upper.setModified(False)
            self.scales.configure(self.key, "fixed", limits)
        except ValueError:
            self.message.setText("Invalid limits: enter finite numbers with lower < upper.")
        else:
            self.message.setText("Display only; fixed/locked limits are shared by linked slices.")
