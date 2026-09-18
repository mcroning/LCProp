"""Small read-only browser for existing DiagnosticData row records."""
from collections.abc import Mapping
from lcprop.gui.number_format import format_number

from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QLabel, QTableWidget, QTableWidgetItem,
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
        self.empty_label = QLabel("No retained tables in this result.")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        self.selector.currentIndexChanged.connect(self._show_table)
        layout.addWidget(self.selector)
        layout.addWidget(self.table)

    def clear(self):
        self._tables = {}
        self.selector.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.empty_label.show()
        self.selector.hide()
        self.table.hide()

    def set_run_data(self, run_data):
        previous = self.selector.currentData()
        self.clear()
        for key, diagnostic in run_data.diagnostics.items():
            rows = diagnostic.values.get("rows")
            if isinstance(rows, (list, tuple)) and rows and all(isinstance(r, Mapping) for r in rows):
                self._tables[key] = rows
                self.selector.addItem(diagnostic.display_name, key)
        self.empty_label.setVisible(not self._tables)
        self.selector.setVisible(bool(self._tables))
        self.table.setVisible(bool(self._tables))
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
                text = format_number(value, quantity="tolerance" if key == "tolerance" or key.endswith("_tolerance") else key)
                item = QTableWidgetItem(text)
                item.setToolTip(format_number(value, exact=True))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
