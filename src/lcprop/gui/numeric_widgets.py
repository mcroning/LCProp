"""Compact spin-box display with the existing scientific editing precision."""
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox

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
