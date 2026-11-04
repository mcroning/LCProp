"""Compact spin-box display with the existing scientific editing precision."""
from PySide6.QtGui import QValidator
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

from lcprop.gui.number_format import format_number


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Show concise values at rest; use the original decimal editor on focus.

    Reinterpreting an unchanged compact string must return the original value,
    not round it to the number of digits used for presentation.
    """

    def textFromValue(self, value):
        if self.hasFocus():
            return super().textFromValue(value)
        return format_number(value)

    def valueFromText(self, text):
        if not self.hasFocus() and text == self.prefix() + format_number(self.value()) + self.suffix():
            return self.value()
        return super().valueFromText(text)

    def validate(self, text, pos):
        if not self.hasFocus() and text == self.prefix() + format_number(self.value()) + self.suffix():
            return QValidator.State.Acceptable, text, pos
        return super().validate(text, pos)

    def focusInEvent(self, event):
        self.lineEdit().setText(self.prefix() + super().textFromValue(self.value()) + self.suffix())
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.lineEdit().setText(self.prefix() + format_number(self.value()) + self.suffix())

    def sizeHint(self):
        # Qt's compact endpoint hints omit digits exposed when editing.
        hint = super().sizeHint()
        compact = max(self.fontMetrics().horizontalAdvance(format_number(v))
                      for v in (self.minimum(), self.maximum()))
        full = max(self.fontMetrics().horizontalAdvance(
            QDoubleSpinBox.textFromValue(self, v))
            for v in (self.minimum(), self.maximum(), self.value()))
        return QSize(hint.width() + max(0, full - compact) + 12, hint.height())

    def minimumSizeHint(self):
        return self.sizeHint()


class GridSpinBox(QSpinBox):
    """Reserve five visible digits without changing the legal grid range."""

    def sizeHint(self):
        hint = super().sizeHint()
        metrics = self.fontMetrics()
        current = max(metrics.horizontalAdvance(self.textFromValue(v))
                      for v in (self.minimum(), self.maximum()))
        return QSize(hint.width() + max(0, metrics.horizontalAdvance('00000') - current) + 12,
                     hint.height())

    def minimumSizeHint(self):
        return self.sizeHint()
