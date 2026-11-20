#@title Open PR workbench — run again after restart or reconnect
"""Single-cell opening flow for the unpublished source preview."""
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import zipfile

RENDERER_VERSION = "3.11.2"
SETUP_NAME = "LCProp-PR-workbench-setup.zip"
SETUP_SHA256 = "c7d93313ec5c9f1679ba45fd6bb9dff00854b578cffc8f7af5676ee8c7549811"
SETUP_DOWNLOAD_URL = ""
WHEEL_NAME = "lcprop-0.5.0rc5-py3-none-any.whl"
WHEEL_SHA256 = "5bdc6327a9418b12dab9f7f61f5e74bde108885b5d0c541df4547c980645a4e0"


def _installed_matches(wheel_bytes):
    try:
        dist = importlib.metadata.distribution("lcprop")
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
            members = [n for n in wheel.namelist() if n.startswith("lcprop/") and not n.endswith("/")]
            return bool(members) and all(
                Path(dist.locate_file(name)).read_bytes() == wheel.read(name)
                for name in members)
    except (importlib.metadata.PackageNotFoundError, OSError):
        return False


def _verified_wheel(kit):
    if kit.stat().st_size > 16 * 1024**2:
        raise ValueError("Setup kit is too large; choose the supplied workbench setup ZIP.")
    data = kit.read_bytes()
    if hashlib.sha256(data).hexdigest() != SETUP_SHA256:
        raise ValueError("Setup kit checksum does not match. Upload the supplied setup ZIP again; nothing was installed.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entry = archive.getinfo(WHEEL_NAME)
        if entry.file_size > 8 * 1024**2:
            raise ValueError("Setup wheel is too large")
        wheel = archive.read(entry)
    if hashlib.sha256(wheel).hexdigest() != WHEEL_SHA256:
        raise ValueError("Wheel checksum does not match. Nothing was installed.")
    return wheel


def _display_workbench(namespace):
    try:
        from google.colab import output
    except ImportError:
        hosted = False
    else:
        output.enable_custom_widget_manager()
        hosted = True
    get_ipython().run_line_magic("matplotlib", "widget")
    from lcprop.notebook import open_pr_workbench
    namespace["lcprop_pr_workbench"] = open_pr_workbench(namespace, hosted=hosted)


def _open_pr_notebook(namespace):
    # Redisplay the same controller before considering any installation. No hot
    # reload, replacement, reset or package change in a live editing session.
    if namespace.get("lcprop_pr_workbench") is not None:
        _display_workbench(namespace)
        return
    if list(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("This preview opens on CPU. Choose Runtime > Change runtime type > CPU, then reconnect.")
    kit = Path.cwd() / SETUP_NAME
    if not kit.is_file():
        if SETUP_DOWNLOAD_URL:
            print("Download the required setup ZIP: " + SETUP_DOWNLOAD_URL)
        print(f"One-time prerequisite: upload {SETUP_NAME} using Colab's Files sidebar (folder icon, Upload).")
        print("Choose that setup ZIP only — not a result ZIP or wheel. Then run this same Open cell again. No picker opens automatically.")
        return
    wheel_bytes = _verified_wheel(kit)
    setup_dir = Path.cwd() / ".lcprop-pr-setup"
    marker = setup_dir / "installed.json"
    try:
        receipt = json.loads(marker.read_text())
    except (OSError, ValueError):
        receipt = {}
    if receipt.get("wheel_sha256") == WHEEL_SHA256 and receipt.get("installed_pid") == os.getpid():
        print("Setup complete. Choose Runtime > Restart session once (not Disconnect and delete), then run this same Open cell again.")
        return
    installed = _installed_matches(wheel_bytes)
    try:
        importlib.metadata.version("ipympl")
        rendering_ready = importlib.metadata.version("matplotlib") == RENDERER_VERSION
    except importlib.metadata.PackageNotFoundError:
        rendering_ready = False
    if not installed or not rendering_ready:
        if any(name == "lcprop" or name.startswith("lcprop.") for name in sys.modules):
            raise RuntimeError("A different LCProp is already loaded. Save your work, restart this session, then rerun Open. Nothing was replaced.")
        setup_dir.mkdir(exist_ok=True)
        wheel_path = setup_dir / WHEEL_NAME
        wheel_path.write_bytes(wheel_bytes)
        print("Verified setup kit. Installing this preview and its notebook dependencies in this temporary CPU runtime…")
        # A previous development wheel can share the rc3 version string. Replace
        # only LCProp explicitly; resolve the extra separately without forcing
        # reinstalls of Colab's scientific dependencies.
        if not installed:
            subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel_path)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib==" + RENDERER_VERSION, str(wheel_path) + "[colab]"], check=True)
        if not _installed_matches(wheel_bytes):
            raise RuntimeError("Installation did not match the verified wheel. Stop here; no workbench or solver was started.")
        marker.write_text(json.dumps({"wheel_sha256": WHEEL_SHA256, "installed_pid": os.getpid()}))
        print("Setup complete. Choose Runtime > Restart session once (not Disconnect and delete), then run this same Open cell again.")
        return
    _display_workbench(namespace)


if __name__ == "__main__":
    try:
        _open_pr_notebook(globals())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"Workbench did not open: {error}")
        print("Your scientific request was not changed and no propagation was started.")
