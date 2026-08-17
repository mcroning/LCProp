from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget

from lcprop.core.context import GridSpec
from lcprop.gui.panels.helpers import double_spin_box, spin_box


class GridPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.Nx = spin_box(16, 4096, 64)
        self.Ny = spin_box(16, 4096, 64)
        self.dz_um = double_spin_box(0.1, 1000.0, 20.0)
        self.z_length_um = double_spin_box(1.0, 100000.0, 3000.0)
        self.x_aperture_um = double_spin_box(1.0, 10000.0, 75.0)
        self.y_aperture_um = double_spin_box(1.0, 10000.0, 100.0)

        form.addRow("Nx", self.Nx)
        form.addRow("Ny", self.Ny)
        form.addRow("dz (µm)", self.dz_um)
        form.addRow("z length (µm)", self.z_length_um)
        form.addRow("x aperture (µm)", self.x_aperture_um)
        form.addRow("y aperture (µm)", self.y_aperture_um)

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
        """Populate the controls from one exactly representable grid."""

        grid.validate()
        self.Nx.setValue(grid.Nx)
        self.Ny.setValue(grid.Ny)
        self.dz_um.setValue(grid.dz_um)
        self.z_length_um.setValue(grid.z_length_um)
        self.x_aperture_um.setValue(grid.x_aperture_um)
        self.y_aperture_um.setValue(grid.y_aperture_um)
