"""Opening flow only: mock installation and display; never run a solver."""
import hashlib
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def opening(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = Path(__file__).parents[1] / "notebooks/pr_workbench_open.py"
    scope = runpy.run_path(str(code), run_name="opening_tests")
    return scope["_open_pr_notebook"].__globals__


def kit(opening, tmp_path):
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lcprop/fake.py", "# fake only\n")
    data = wheel.getvalue()
    path = tmp_path / opening["SETUP_NAME"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(opening["WHEEL_NAME"], data)
    opening["SETUP_SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    opening["WHEEL_SHA256"] = hashlib.sha256(data).hexdigest()
    return path, data


def test_missing_kit_is_actionable_without_picker_or_install(opening, capsys):
    opening["_open_pr_notebook"]({})
    message = capsys.readouterr().out
    assert opening["SETUP_NAME"] in message and "Files sidebar" in message
    assert "No picker opens automatically" in message


def test_kit_and_inner_wheel_tampering_rejected(opening, tmp_path):
    path, _ = kit(opening, tmp_path)
    opening["WHEEL_SHA256"] = "0" * 64
    with pytest.raises(ValueError, match="Wheel checksum"):
        opening["_verified_wheel"](path)
    path.write_bytes(b"wrong kit")
    with pytest.raises(ValueError, match="Setup kit checksum"):
        opening["_verified_wheel"](path)


def test_fresh_install_and_same_cell_restart_instruction(opening, tmp_path, capsys):
    kit(opening, tmp_path)
    calls = []
    opening["sys"] = SimpleNamespace(modules={}, executable="fake-python")
    opening["_installed_matches"] = lambda _: bool(calls)
    opening["subprocess"] = SimpleNamespace(run=lambda args, **kw: calls.append(args))
    opening["_display_workbench"] = lambda _: pytest.fail("Must wait for explicit restart")
    opening["_open_pr_notebook"]({})
    assert len(calls) == 2
    assert "--force-reinstall" in calls[0] and "--no-deps" in calls[0]
    assert calls[1][-1].endswith("[colab]")
    receipt = json.loads((tmp_path / ".lcprop-pr-setup/installed.json").read_text())
    assert receipt["wheel_sha256"] == opening["WHEEL_SHA256"]
    opening["_open_pr_notebook"]({})
    assert len(calls) == 2  # Re-running before restart cannot reinstall in a loop.
    assert "Restart session once" in capsys.readouterr().out


def test_existing_imported_different_package_not_replaced(opening, tmp_path):
    kit(opening, tmp_path)
    opening["_installed_matches"] = lambda _: False
    opening["sys"] = SimpleNamespace(modules={"lcprop": object()})
    with pytest.raises(RuntimeError, match="Nothing was replaced"):
        opening["_open_pr_notebook"]({})


def test_live_controller_redisplayed_without_kit_or_replacement(opening):
    panel = object()
    namespace = {"lcprop_pr_workbench": panel}
    seen = []
    opening["_display_workbench"] = lambda ns: seen.append(ns["lcprop_pr_workbench"])
    opening["_open_pr_notebook"](namespace)
    assert seen == [panel] and namespace["lcprop_pr_workbench"] is panel


def test_after_restart_exact_install_opens_without_installing(opening, tmp_path, monkeypatch):
    kit(opening, tmp_path)
    opening["_installed_matches"] = lambda _: True
    marker = tmp_path / ".lcprop-pr-setup/installed.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"wheel_sha256": opening["WHEEL_SHA256"], "installed_pid": -1}))
    seen = []
    opening["_display_workbench"] = lambda ns: seen.append(ns)
    monkeypatch.setattr(opening["importlib"].metadata, "version", lambda name: "0.10.0" if name == "ipympl" else opening["RENDERER_VERSION"])
    namespace = {}
    opening["_open_pr_notebook"](namespace)
    assert seen == [namespace]


def test_interrupted_wheel_only_install_resumes_rendering_extra(opening, tmp_path, monkeypatch, capsys):
    kit(opening, tmp_path)
    opening["_installed_matches"] = lambda _: True
    opening["sys"] = SimpleNamespace(modules={}, executable="fake-python")
    def missing(name):
        raise opening["importlib"].metadata.PackageNotFoundError(name)
    monkeypatch.setattr(opening["importlib"].metadata, "version", missing)
    calls = []
    opening["subprocess"] = SimpleNamespace(run=lambda args, **kw: calls.append(args))
    opening["_display_workbench"] = lambda _: pytest.fail("Must restart after dependency install")
    opening["_open_pr_notebook"]({})
    assert len(calls) == 1 and calls[0][-1].endswith("[colab]")
    assert "--force-reinstall" not in calls[0]
    assert "Restart session once" in capsys.readouterr().out


def test_stock_renderer_is_upgraded_without_replacing_exact_engine(opening, tmp_path, monkeypatch):
    kit(opening, tmp_path)
    opening["_installed_matches"] = lambda _: True
    opening["sys"] = SimpleNamespace(modules={}, executable="fake-python")
    monkeypatch.setattr(opening["importlib"].metadata, "version",
                        lambda name: "3.10.0" if name == "matplotlib" else "0.10.0")
    calls = []
    opening["subprocess"] = SimpleNamespace(run=lambda args, **kw: calls.append(args))
    opening["_open_pr_notebook"]({})
    assert len(calls) == 1
    assert "matplotlib==3.11.2" in calls[0] and "--force-reinstall" not in calls[0]


def test_missing_published_setup_has_direct_verified_download(opening, capsys):
    opening['_open_pr_notebook']({})
    assert 'https://github.com/mcroning/LCProp/releases/download/v0.5.0rc4/LCProp-PR-workbench-setup.zip' in capsys.readouterr().out
