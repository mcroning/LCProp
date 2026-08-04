from __future__ import annotations

import os
from pathlib import Path
import tomllib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import lcprop.pr.gui.app as pr_app


class _Signal:
    def __init__(self) -> None:
        self.connected = []

    def connect(self, callback) -> None:
        self.connected.append(callback)


class _Application:
    instances = []

    def __init__(self, argv) -> None:
        self.argv = list(argv)
        self.application_name = None
        self.aboutToQuit = _Signal()
        self.instances.append(self)

    def setApplicationName(self, name: str) -> None:
        self.application_name = name

    def exec(self) -> int:
        return 23


class _Window:
    instances = []

    def __init__(self) -> None:
        self.shown = False
        self.shutdown_calls = 0
        self.instances.append(self)

    def show(self) -> None:
        self.shown = True

    def shutdown_background_run(self) -> bool:
        self.shutdown_calls += 1
        return True


def test_pr_application_main_constructs_shows_and_connects_shutdown(
    monkeypatch,
):
    _Application.instances.clear()
    _Window.instances.clear()
    monkeypatch.setattr(pr_app, "QApplication", _Application)
    monkeypatch.setattr(pr_app, "PRMainWindow", _Window)
    monkeypatch.setattr(pr_app.sys, "argv", ["lcprop-pr", "--example"])

    exit_code = pr_app.main()

    application = _Application.instances[0]
    window = _Window.instances[0]
    assert exit_code == 23
    assert application.argv == ["lcprop-pr", "--example"]
    assert application.application_name == "LCProp PR"
    assert application.aboutToQuit.connected == [
        window.shutdown_background_run
    ]
    assert window.shown


def test_packaging_exposes_only_the_new_pr_command_for_this_milestone():
    project_root = Path(__file__).resolve().parents[1]
    document = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert document["project"]["scripts"]["lcprop-pr"] == (
        "lcprop.pr.gui.app:main"
    )
