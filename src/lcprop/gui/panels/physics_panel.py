from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget

from lcprop.core.context import LCMaterial, BiasSpec
from lcprop.gui.panels.helpers import double_spin_box


class PhysicsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.ne = double_spin_box(0.1, 10.0, 1.7)
        self.no = double_spin_box(0.1, 10.0, 1.5)
        self.K = double_spin_box(1e-15, 1e-9, 7e-12, decimals=15)
        self.delta_epsilon = double_spin_box(0.0, 100.0, 13.0)
        self.V_bias = double_spin_box(0.0, 100.0, 0.9144)
        self.theta_bc = double_spin_box(0.0, 1.5708, 0.0)

        form.addRow("ne", self.ne)
        form.addRow("no", self.no)
        form.addRow("K (N)", self.K)
        form.addRow("Δε", self.delta_epsilon)
        form.addRow("V bias (V)", self.V_bias)
        form.addRow("theta_bc (rad)", self.theta_bc)

        layout.addLayout(form)
        layout.addStretch(1)

    def material(self) -> LCMaterial:
        return LCMaterial(
            ne=self.ne.value(),
            no=self.no.value(),
            K=self.K.value(),
            delta_epsilon=self.delta_epsilon.value(),
        )

    def bias(self) -> BiasSpec:
        return BiasSpec(
            V_bias=self.V_bias.value(),
            theta_bc=self.theta_bc.value(),
        )
