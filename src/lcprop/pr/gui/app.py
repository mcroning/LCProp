"""Application entry point for the standalone LCProp PR GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase

from lcprop.pr.gui.main_window import PRMainWindow


def configure_application_font(app) -> None:
    """Resolve an unavailable Qt family before widgets trigger alias fallback."""
    font = app.font()
    families = QFontDatabase.families()
    if font.family() in families:
        return
    system = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    family = next((name for name in (system.family(), 'Arial', 'DejaVu Sans',
                                    'Liberation Sans') if name in families), None)
    if family is not None:
        font.setFamily(family)
        app.setFont(font)


def main() -> int:
    app = QApplication(sys.argv)
    configure_application_font(app)
    app.setApplicationName("LCProp PR")
    window = PRMainWindow()
    app.aboutToQuit.connect(window.shutdown_background_run)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
