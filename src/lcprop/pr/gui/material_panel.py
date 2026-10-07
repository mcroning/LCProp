from __future__ import annotations

from lcprop.gui.layout import readable_form
from lcprop.gui.numeric_widgets import CompactDoubleSpinBox

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLineEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.illumination import INTEGRAL_NORMALIZATION, LEGACY_NORMALIZATION


def _double_spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 9,
) -> QDoubleSpinBox:
    widget = CompactDoubleSpinBox()
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
        primary = readable_form(QFormLayout())

        self.normalization_mode = QComboBox()
        self.normalization_mode.addItem("Physical total illumination", INTEGRAL_NORMALIZATION)
        self.normalization_mode.addItem("Legacy channel-peak reference", LEGACY_NORMALIZATION)
        self.dark_irradiance = QLineEdit("10")
        self.uniform_irradiance = QLineEdit("0")
        for edit in (self.dark_irradiance, self.uniform_irradiance):
            edit.setPlaceholderText("Required: enter a value or explicit 0")
        primary.addRow("Intensity normalization", self.normalization_mode)
        primary.addRow("Dark-equivalent irradiance (mW/cm²)", self.dark_irradiance)
        primary.addRow("Uniform background irradiance (mW/cm²)", self.uniform_irradiance)
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
        primary.addRow(
            "Reduced x-only applied field (normalized)", self.applied_field
        )
        primary.addRow("Gain-length product", self.gain_length_product)
        primary.addRow("Refractive index", self.refractive_index)
        def update_normalization():
            physical = self.normalization_mode.currentData() == INTEGRAL_NORMALIZATION
            for control in (self.dark_irradiance, self.uniform_irradiance):
                control.setEnabled(physical)
            for control in (self.dark_intensity, self.uniform_background_intensity):
                control.setEnabled(not physical)
        self.normalization_mode.currentIndexChanged.connect(update_normalization)
        update_normalization()
        layout.addLayout(primary)

        advanced_box = QGroupBox("Advanced material normalization")
        advanced = readable_form(QFormLayout(advanced_box))
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

    def closure_background(self):
        """A7 physical requests defer the derived fraction until launch acceptance."""
        if self.normalization_mode.currentData() == INTEGRAL_NORMALIZATION:
            return 0.
        return self.dark_intensity.value() + self.uniform_background_intensity.value()

    def material(self, *, allow_incomplete=False) -> PRMaterialSpec:
        override = (
            self.characteristic_wavenumber_per_um_override.value()
            if self.use_characteristic_wavenumber_override.isChecked()
            else None
        )
        physical = self.normalization_mode.currentData() == INTEGRAL_NORMALIZATION
        def irradiance(edit):
            from decimal import Decimal, InvalidOperation
            text = edit.text().strip()
            if not text: return None
            try: return float(Decimal(text) / Decimal(1000))
            except InvalidOperation as exc:
                if allow_incomplete: return None  # GUI rollback snapshot, never execution.
                raise ValueError("physical irradiance must be numeric") from exc
        return PRMaterialSpec(
            normalization_identity=self.normalization_mode.currentData(),
            dark_irradiance_W_cm2=irradiance(self.dark_irradiance) if physical else None,
            uniform_irradiance_W_cm2=irradiance(self.uniform_irradiance) if physical else None,
            dark_intensity=0. if physical else self.dark_intensity.value(),
            uniform_background_intensity=(
                0. if physical else self.uniform_background_intensity.value()
            ),
            applied_field=self.applied_field.value(),
            gain_length_product=self.gain_length_product.value(),
            refractive_index=self.refractive_index.value(),
            relative_permittivity=self.relative_permittivity.value(),
            mobile_charge_density_m3=self.mobile_charge_density_m3.value(),
            temperature_K=self.temperature_K.value(),
            characteristic_wavenumber_per_um_override=override,
        )

    def set_material(self, material: PRMaterialSpec, *, validate=True) -> None:
        if validate: material.validate()
        self.normalization_mode.setCurrentIndex(self.normalization_mode.findData(material.normalization_identity))
        from decimal import Decimal
        self.dark_irradiance.setText("" if material.dark_irradiance_W_cm2 is None else
            str(Decimal(str(material.dark_irradiance_W_cm2))*1000))
        self.uniform_irradiance.setText("" if material.uniform_irradiance_W_cm2 is None else
            str(Decimal(str(material.uniform_irradiance_W_cm2))*1000))
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
