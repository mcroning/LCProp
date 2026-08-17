"""Material-neutral Save/Open Experiment GUI controls and file helpers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

from lcprop.persistence import (
    EXPERIMENT_FILE_EXTENSION,
    ExperimentMaterialError,
)


EXPERIMENT_FILE_FILTER = "LCProp experiment (*.lcprop.json);;JSON files (*.json)"
_MATERIAL_NAMES = {
    "lc": ("Liquid crystal (LC)", "LC"),
    "pr": ("Photorefractive (PR)", "PR"),
}


class ExperimentFileButtons(QWidget):
    """Shared compact Save/Open surface used by each material application."""

    saveRequested = Signal()
    openRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.save_button = QPushButton("Save Experiment...")
        self.open_button = QPushButton("Open Experiment...")
        self.save_button.clicked.connect(self.saveRequested.emit)
        self.open_button.clicked.connect(self.openRequested.emit)
        layout.addWidget(self.save_button)
        layout.addWidget(self.open_button)


def normalized_experiment_path(path: str | Path) -> Path:
    """Append the canonical compound suffix when a save path omits it."""

    result = Path(path)
    if not str(result).endswith(EXPERIMENT_FILE_EXTENSION):
        result = Path(str(result) + EXPERIMENT_FILE_EXTENSION)
    return result


def choose_experiment_save_path(parent: QWidget) -> Path | None:
    filename, _ = QFileDialog.getSaveFileName(
        parent,
        "Save LCProp Experiment",
        "",
        EXPERIMENT_FILE_FILTER,
    )
    return None if not filename else normalized_experiment_path(filename)


def choose_experiment_open_path(parent: QWidget) -> Path | None:
    filename, _ = QFileDialog.getOpenFileName(
        parent,
        "Open LCProp Experiment",
        "",
        EXPERIMENT_FILE_FILTER,
    )
    return None if not filename else Path(filename)


def launchplane_presentation_payload(beam_panel) -> dict:
    """Encode full editor provenance without making it canonical physics."""

    from launchplane.serialization import beam_stack_to_dict

    return {
        "beam_editor": {
            "provider": "launchplane",
            "schema_version": 1,
            "beam_stack": beam_stack_to_dict(beam_panel.beam_stack_definition),
        }
    }


def launchplane_stack_from_presentation(loaded_experiment):
    """Return saved editor state, or ``None`` for canonical q-mode fallback."""

    editor = loaded_experiment.presentation_payload.get("beam_editor")
    if editor is None:
        return None
    from launchplane.serialization import beam_stack_from_dict

    return beam_stack_from_dict(editor["beam_stack"])


def show_experiment_open_warning(parent: QWidget, error: Exception) -> bool:
    """Show a clear warning for a structured wrong-material load failure."""

    if not isinstance(error, ExperimentMaterialError):
        return False
    actual = error.actual_material_id
    material_name, application_name = _MATERIAL_NAMES.get(
        actual,
        (f"material {actual!r}", "matching material"),
    )
    QMessageBox.warning(
        parent,
        "Experiment Material Mismatch",
        f"This experiment is for {material_name}. "
        f"Open it with the {application_name} application.",
    )
    return True


__all__ = [
    "EXPERIMENT_FILE_FILTER",
    "ExperimentFileButtons",
    "choose_experiment_open_path",
    "choose_experiment_save_path",
    "launchplane_presentation_payload",
    "launchplane_stack_from_presentation",
    "normalized_experiment_path",
    "show_experiment_open_warning",
]
