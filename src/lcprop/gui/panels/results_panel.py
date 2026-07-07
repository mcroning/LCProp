from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.request_summary = QTextEdit()
        self.request_summary.setReadOnly(True)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout.addWidget(QLabel("Request summary"))
        layout.addWidget(self.request_summary)
        layout.addWidget(QLabel("Run console"))
        layout.addWidget(self.output)

    def set_request_summary(self, text: str) -> None:
        self.request_summary.setPlainText(text)

    def append_console(self, text: str) -> None:
        self.output.append(text)
