"""Liquid-crystal material and bias controls."""

from PySide6.QtWidgets import QFormLayout, QLabel, QGridLayout, QVBoxLayout, QWidget

from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.gui.panels.helpers import double_spin_box
from lcprop.lc.bias import (
    compute_b_from_voltage,
    compute_freedericksz_voltage,
)
from lcprop.lc.bias import theta0_from_b_dirichlet_bc


class PhysicsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.ne = double_spin_box(0.1, 10.0, 1.7)
        self.no = double_spin_box(0.1, 10.0, 1.5)
        self.K_pN = double_spin_box(0.001, 1000.0, 7.0, decimals=3)
        self.K_pN.setSingleStep(0.1)
        self.delta_epsilon = double_spin_box(0.0, 100.0, 13.0)
        self.V_bias = double_spin_box(0.0, 100.0, 0.9144)
        self.V_bias.setSingleStep(0.1)
        self.theta_bc = double_spin_box(0.0, 1.5708, 0.0, decimals=6)
        self.theta_bc.setSingleStep(3.141592653589793 / 8.0)

        form.addRow("ne", self.ne)
        form.addRow("no", self.no)
        form.addRow("K (pN)", self.K_pN)
        form.addRow("Δε", self.delta_epsilon)
        form.addRow("V bias (V)", self.V_bias)
        form.addRow("x-boundary director angle θ_bc (rad)", self.theta_bc)

        layout.addLayout(form)

        self.derived_widget = QWidget()
        self.derived_widget.setStyleSheet(
            """
            QWidget {
                border: 1px solid palette(mid);
                border-radius: 4px;
                background: palette(base);
            }
            QLabel {
                border: none;
                background: transparent;
                padding: 2px;
            }
            """
        )
        derived_grid = QGridLayout(self.derived_widget)
        derived_grid.setContentsMargins(6, 6, 6, 6)
        derived_grid.setHorizontalSpacing(12)
        derived_grid.setVerticalSpacing(2)

        self.derived_values = {}
        derived_rows = [
            ("Resolved dimensionless bias b", "b"),
            ("Freedericksz transition voltage", "vf"),
            ("Voltage / Freedericksz voltage", "ratio"),
            ("Dark-bias center director angle θ(0)", "theta0"),
        ]
        for row, (label_text, key) in enumerate(derived_rows):
            label = QLabel(label_text)
            value = QLabel("—")
            derived_grid.addWidget(label, row, 0)
            derived_grid.addWidget(value, row, 1)
            self.derived_values[key] = value

        layout.addWidget(self.derived_widget)

        for widget in [
            self.ne,
            self.no,
            self.K_pN,
            self.delta_epsilon,
            self.V_bias,
            self.theta_bc,
        ]:
            widget.valueChanged.connect(self.update_derived_readout)

        self.update_derived_readout()
        layout.addStretch(1)

    def material(self) -> LCMaterial:
        return LCMaterial(
            ne=self.ne.value(),
            no=self.no.value(),
            K=self.K_pN.value() * 1e-12,
            delta_epsilon=self.delta_epsilon.value(),
        )

    def bias(self) -> BiasSpec:
        return BiasSpec(
            V_bias=self.V_bias.value(),
            theta_bc=self.theta_bc.value(),
        )

    def set_material(self, material: LCMaterial) -> None:
        """Populate the material controls without discarding hidden fields."""

        material.validate()
        if material.name != LCMaterial().name:
            raise ValueError("LC GUI cannot represent a non-default material name")
        self.ne.setValue(material.ne)
        self.no.setValue(material.no)
        self.K_pN.setValue(material.K * 1e12)
        self.delta_epsilon.setValue(material.delta_epsilon)

    def set_bias(self, bias: BiasSpec) -> None:
        """Populate the bias controls after hidden-option validation."""

        bias.validate()
        default = BiasSpec()
        if (
            bias.theta_min != default.theta_min
            or bias.theta_max != default.theta_max
            or bias.theta_center is not None
            or bias.b_override is not None
        ):
            raise ValueError("LC GUI cannot represent non-default hidden bias options")
        self.V_bias.setValue(bias.V_bias)
        self.theta_bc.setValue(bias.theta_bc)


    def update_derived_readout(self) -> None:
        material = self.material()
        bias = self.bias()

        b = bias.b_override
        if b is None:
            b = compute_b_from_voltage(
                bias.V_bias,
                K=material.K,
                delta_epsilon=material.delta_epsilon,
            )

        vf = compute_freedericksz_voltage(
            K=material.K,
            delta_epsilon=material.delta_epsilon,
        )

        ratio = bias.V_bias / vf if vf > 0 else float("nan")

        theta0 = theta0_from_b_dirichlet_bc(b, bias.theta_bc)

        self.derived_values["b"].setText(f"{b:.6g}")
        self.derived_values["vf"].setText(f"{vf:.6g} V")
        self.derived_values["ratio"].setText(f"{ratio:.6g}")
        self.derived_values["theta0"].setText(f"{theta0:.6g} rad")
