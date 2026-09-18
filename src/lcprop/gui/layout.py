"""Shared presentation policies for readable forms and wrapping action rows."""
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QFormLayout, QLayout, QVBoxLayout


def application_layout(parent):
    """Use compact shared chrome so wrapping controls leave room for results."""
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)
    return layout


def readable_form(form):
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    return form


class FlowLayout(QLayout):
    """Lay out visible controls at their natural size, wrapping to available width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)

    def addItem(self, item):
        self._items.append(item)

    def addStretch(self, _stretch=0):
        # Wrapping rows use available width instead of a fixed spacer.
        pass

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation.Horizontal)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), False)

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def _arrange(self, rect, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, height = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint().expandedTo(item.minimumSize())
            width = min(size.width(), max(1, area.width()))
            if height and x + width > area.right() + 1:
                x, y, height = area.x(), y + height + self.spacing(), 0
            h = item.heightForWidth(width) if item.hasHeightForWidth() else size.height()
            if apply:
                item.setGeometry(QRect(x, y, width, h))
            x += width + self.spacing()
            height = max(height, h)
        return y + height - rect.y() + margins.bottom()
