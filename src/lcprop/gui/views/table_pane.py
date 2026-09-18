"""Small read-only browser for existing DiagnosticData row records."""
from collections.abc import Mapping

from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class TablePane(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.selector = QComboBox()
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self._tables = {}
        self.selector.currentIndexChanged.connect(self._show_table)
        layout.addWidget(self.selector)
        layout.addWidget(self.table)

    def clear(self):
        self._tables = {}
        self.selector.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

    def set_run_data(self, run_data):
        previous = self.selector.currentData()
        self.clear()
        for key, diagnostic in run_data.diagnostics.items():
            rows = diagnostic.values.get("rows")
            if isinstance(rows, (list, tuple)) and rows and all(isinstance(r, Mapping) for r in rows):
                self._tables[key] = rows
                self.selector.addItem(diagnostic.display_name, key)
        index = self.selector.findData(previous)
        self.selector.setCurrentIndex(max(0, index))
        self._show_table(self.selector.currentIndex())

    def _show_table(self, index):
        rows = self._tables.get(self.selector.itemData(index), ())
        columns = list(dict.fromkeys(key for row in rows for key in row))
        self.table.clear()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(columns))
        labels = {"converged": "Solver converged within configured limits",
                  "field_rel": "Field relative change", "dtheta_rms": "Theta update RMS",
                  "overlap_abs": "Mode overlap", "requested_power_mW": "Requested power (mW)"}
        self.table.setHorizontalHeaderLabels([labels.get(key, key.replace('_', ' ')) for key in columns])
        for i, row in enumerate(rows):
            for j, key in enumerate(columns):
                value = row.get(key)
                text = "Unavailable" if value is None else repr(value) if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
