"""Application entry point for the standalone LCProp PR GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.transport.defaults import default_slurm_runner_from_environment


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LCProp PR")
    window = PRMainWindow(
        slurm_runner=default_slurm_runner_from_environment()
    )
    app.aboutToQuit.connect(window.shutdown_background_run)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
