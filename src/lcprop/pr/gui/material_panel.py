from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from lcprop.pr.specs import PRMaterialSpec


def _double_spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 9,
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


class PRMaterialPanel(QWidget):
    """Controls that map directly to ``PRMaterialSpec``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        defaults = PRMaterialSpec()
        layout = QVBoxLayout(self)
        primary = QFormLayout()

        self.dark_intensity = _double_spin(0.0, 1e9, defaults.dark_intensity)
        self.uniform_background_intensity = _double_spin(
            0.0,
            1e9,
            defaults.uniform_background_intensity,
        )
        self.applied_field = _double_spin(-1e9, 1e9, defaults.applied_field)
        self.gain_length_product = _double_spin(
            -1e9,
            1e9,
            defaults.gain_length_product,
        )
        self.refractive_index = _double_spin(
            1e-9,
            100.0,
            defaults.refractive_index,
        )

        primary.addRow("Dark intensity (normalized)", self.dark_intensity)
        primary.addRow(
            "Uniform background (normalized)",
            self.uniform_background_intensity,
        )
        primary.addRow("Applied field (normalized)", self.applied_field)
        primary.addRow("Gain-length product", self.gain_length_product)
        primary.addRow("Refractive index", self.refractive_index)
        layout.addLayout(primary)

        advanced_box = QGroupBox("Advanced material normalization")
        advanced = QFormLayout(advanced_box)
        self.relative_permittivity = _double_spin(
            1e-9,
            1e12,
            defaults.relative_permittivity,
            decimals=6,
        )
        self.mobile_charge_density_m3 = _double_spin(
            1.0,
            1e30,
            defaults.mobile_charge_density_m3,
            decimals=3,
        )
        self.temperature_K = _double_spin(
            1e-9,
            1e6,
            defaults.temperature_K,
            decimals=6,
        )
        self.use_characteristic_wavenumber_override = QCheckBox(
            "Use characteristic-wavenumber override"
        )
        self.characteristic_wavenumber_per_um_override = _double_spin(
            1e-12,
            1e6,
            0.1,
            decimals=12,
        )
        self.characteristic_wavenumber_per_um_override.setEnabled(False)
        self.use_characteristic_wavenumber_override.toggled.connect(
            self.characteristic_wavenumber_per_um_override.setEnabled
        )

        advanced.addRow("Relative permittivity", self.relative_permittivity)
        advanced.addRow(
            "Mobile charge density (m⁻³)",
            self.mobile_charge_density_m3,
        )
        advanced.addRow("Temperature (K)", self.temperature_K)
        advanced.addRow(self.use_characteristic_wavenumber_override)
        advanced.addRow(
            "Characteristic wavenumber (µm⁻¹)",
            self.characteristic_wavenumber_per_um_override,
        )
        layout.addWidget(advanced_box)
        layout.addStretch(1)

    def material(self) -> PRMaterialSpec:
        override = (
            self.characteristic_wavenumber_per_um_override.value()
            if self.use_characteristic_wavenumber_override.isChecked()
            else None
        )
        return PRMaterialSpec(
            dark_intensity=self.dark_intensity.value(),
            uniform_background_intensity=(
                self.uniform_background_intensity.value()
            ),
            applied_field=self.applied_field.value(),
            gain_length_product=self.gain_length_product.value(),
            refractive_index=self.refractive_index.value(),
            relative_permittivity=self.relative_permittivity.value(),
            mobile_charge_density_m3=self.mobile_charge_density_m3.value(),
            temperature_K=self.temperature_K.value(),
            characteristic_wavenumber_per_um_override=override,
        )

    def set_material(self, material: PRMaterialSpec) -> None:
        material.validate()
        self.dark_intensity.setValue(material.dark_intensity)
        self.uniform_background_intensity.setValue(
            material.uniform_background_intensity
        )
        self.applied_field.setValue(material.applied_field)
        self.gain_length_product.setValue(material.gain_length_product)
        self.refractive_index.setValue(material.refractive_index)
        self.relative_permittivity.setValue(material.relative_permittivity)
        self.mobile_charge_density_m3.setValue(
            material.mobile_charge_density_m3
        )
        self.temperature_K.setValue(material.temperature_K)
        override = material.characteristic_wavenumber_per_um_override
        self.use_characteristic_wavenumber_override.setChecked(
            override is not None
        )
        if override is not None:
            self.characteristic_wavenumber_per_um_override.setValue(override)


__all__ = ["PRMaterialPanel"]
