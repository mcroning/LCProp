from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget

from lcprop.core.context import GridSpec
from lcprop.gui.panels.helpers import double_spin_box, spin_box


PR_DEFAULT_GRID = GridSpec(
    Nx=128,
    Ny=128,
    dz_um=10.0,
    x_aperture_um=200.0,
    y_aperture_um=200.0,
    z_length_um=1000.0,
)


class PRGridPanel(QWidget):
    """PR-owned grid controls with PR-safe defaults."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.Nx = spin_box(2, 4096, PR_DEFAULT_GRID.Nx)
        self.Ny = spin_box(2, 4096, PR_DEFAULT_GRID.Ny)
        self.dz_um = double_spin_box(0.001, 1000.0, PR_DEFAULT_GRID.dz_um)
        self.z_length_um = double_spin_box(
            0.001,
            1e7,
            PR_DEFAULT_GRID.z_length_um,
        )
        self.x_aperture_um = double_spin_box(
            0.001,
            1e6,
            PR_DEFAULT_GRID.x_aperture_um,
        )
        self.y_aperture_um = double_spin_box(
            0.001,
            1e6,
            PR_DEFAULT_GRID.y_aperture_um,
        )

        form.addRow("Nx", self.Nx)
        form.addRow("Ny", self.Ny)
        form.addRow("Optical dz (µm)", self.dz_um)
        form.addRow("Interaction length (µm)", self.z_length_um)
        form.addRow("Periodic x aperture (µm)", self.x_aperture_um)
        form.addRow("Periodic y aperture (µm)", self.y_aperture_um)
        layout.addLayout(form)
        layout.addStretch(1)

    def grid(self) -> GridSpec:
        return GridSpec(
            Nx=self.Nx.value(),
            Ny=self.Ny.value(),
            dz_um=self.dz_um.value(),
            x_aperture_um=self.x_aperture_um.value(),
            y_aperture_um=self.y_aperture_um.value(),
            z_length_um=self.z_length_um.value(),
        )

    def set_grid(self, grid: GridSpec) -> None:
        grid.validate()
        self.Nx.setValue(grid.Nx)
        self.Ny.setValue(grid.Ny)
        self.dz_um.setValue(grid.dz_um)
        self.z_length_um.setValue(grid.z_length_um)
        self.x_aperture_um.setValue(grid.x_aperture_um)
        self.y_aperture_um.setValue(grid.y_aperture_um)


__all__ = ["PR_DEFAULT_GRID", "PRGridPanel"]
