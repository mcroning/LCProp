from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.gui.panels.helpers import double_spin_box


class BeamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.wavelength_um = double_spin_box(0.1, 10.0, 0.633)
        self.power_mW = double_spin_box(0.0, 1000.0, 1.0)
        self.waist_um = double_spin_box(0.1, 1000.0, 3.0)

        form.addRow("Wavelength (µm)", self.wavelength_um)
        form.addRow("Power (mW)", self.power_mW)
        form.addRow("Waist (µm)", self.waist_um)

        layout.addLayout(form)
        layout.addWidget(QLabel("Multichannel beam editor will be added here."))
        layout.addStretch(1)

    def beams(self) -> BeamStack:
        return BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=self.wavelength_um.value(),
                    power_mW=self.power_mW.value(),
                    waist_x_um=self.waist_um.value(),
                    waist_y_um=self.waist_um.value(),
                ),
            )
        )
