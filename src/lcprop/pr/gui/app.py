"""Application entry point for the standalone LCProp PR GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from lcprop.pr.gui.main_window import PRMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LCProp PR")
    window = PRMainWindow()
    app.aboutToQuit.connect(window.shutdown_background_run)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
