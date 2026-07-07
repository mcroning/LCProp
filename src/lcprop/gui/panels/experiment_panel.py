from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget


class ExperimentPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.experiment = QComboBox()
        self.experiment.addItems([
            "Static propagation",
            "Time-dependent propagation",
            "Soliton",
            "Soliton existence curve",
        ])

        form = QFormLayout()
        form.addRow("Experiment", self.experiment)

        layout.addLayout(form)
        layout.addWidget(QLabel("Current milestone: Static propagation is wired to LocalRunner."))
        layout.addStretch(1)

    def current_experiment(self) -> str:
        return self.experiment.currentText()
