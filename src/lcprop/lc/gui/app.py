from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication
from lcprop.lc.gui.main_window import LCPropMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = LCPropMainWindow()
    app.aboutToQuit.connect(win.shutdown_background_run)
    win.resize(760, 760)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
