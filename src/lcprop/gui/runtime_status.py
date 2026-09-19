"""Stable presentation geometry for frequently changing runtime messages."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy


class RuntimeStatusLabel(QLabel):
    """Keep full text accessible while painting one elided, fixed-height line."""

    def __init__(self, text="", parent=None, *, preferred_width=220):
        super().__init__(text, parent)
        self._preferred_width = preferred_width
        self._detail_hint = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._update_details()

    def sizeHint(self):
        return QSize(self._preferred_width, self.fontMetrics().height())

    def minimumSizeHint(self):
        return QSize(40, self.sizeHint().height())

    def setText(self, text):
        super().setText(text)
        self._update_details()

    def setToolTip(self, text):
        self._detail_hint = text
        self._update_details()

    def _update_details(self):
        detail = getattr(self, "_detail_hint", "")
        super().setToolTip(self.text() + ("\n" + detail if detail else ""))
        self.setAccessibleName(self.text())

    def paintEvent(self, event):
        painter = QPainter(self)
        text = self.fontMetrics().elidedText(
            " ".join(self.text().splitlines()), Qt.TextElideMode.ElideRight,
            self.contentsRect().width(),
        )
        self.style().drawItemText(
            painter, self.contentsRect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.palette(), self.isEnabled(), text, self.foregroundRole(),
        )


def reserve_button_text(button, texts, *, retain_hidden=False):
    """Reserve the largest lifecycle caption without changing button semantics."""
    original = button.text()
    widths = []
    for text in (original, *texts):
        button.setText(text)
        widths.append(button.sizeHint().width())
    button.setText(original)
    button.setFixedWidth(max(widths))
    if retain_hidden:
        policy = button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        button.setSizePolicy(policy)
