"""Stage an unpublished wheel plus a one-cell notebook; never upload or install."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def stage(wheel, destination):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"lcprop-[A-Za-z0-9_.+-]+\.whl", wheel.name):
        raise ValueError("Choose an LCProp .whl file; no setup files were changed")
    wheel_bytes = wheel.read_bytes()
    destination.mkdir(parents=True, exist_ok=True)
    kit = destination / "LCProp-PR-workbench-setup.zip"
    with zipfile.ZipFile(kit, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in [(wheel.name, wheel_bytes), ("READ-ME.txt", (
            "Unpublished reduced PR workbench setup. Keep this ZIP intact.\n"
            "In Colab CPU, upload this ZIP through the Files sidebar, then run\n"
            "the supplied notebook's Open PR workbench cell. If asked, restart\n"
            "the session once and run that same cell again. No propagation\n"
            "starts until you explicitly choose Run in the workbench.\n"
        ).encode())]:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    hashes = {"setup": hashlib.sha256(kit.read_bytes()).hexdigest(),
              "wheel": hashlib.sha256(wheel_bytes).hexdigest()}
    opener = root / "notebooks/pr_workbench_open.py"
    code = opener.read_text()
    for key, value in [("SETUP_SHA256", hashes["setup"]),
                       ("WHEEL_NAME", wheel.name), ("WHEEL_SHA256", hashes["wheel"])]:
        code, count = re.subn(r'^' + key + r' = "[^"]+"',
                              lambda _: key + " = " + json.dumps(value), code, flags=re.M)
        assert count == 1
    opener.write_text(code)
    markdown = '''# PR workbench — reduced Static / time dependent

1. Connect to a **CPU** runtime.
2. One-time prerequisite for this unpublished preview: upload **LCProp-PR-workbench-setup.zip** with the **Files sidebar → Upload**. Choose that setup ZIP only, not a result ZIP or wheel. Keep it zipped.
3. Run **Open PR workbench** below. If it asks for **Runtime → Restart session**, do that once and run this same cell again. Then use **Beams / Results / Files**. **Files → Open synthetic example** displays the included example without propagation.

**Opening help:** No automatic file picker or Drive mount. After a browser disconnect, rerun Open to reconnect to the existing live workbench; this preserves committed drafts and results. A deleted/replaced runtime loses unsaved state. Save needed files before restarting an established session. Never rerun the setup by deleting your current workbench.

**Using the workbench:** Evolution selects reduced x-only Static or time dependent; CPU is default. Advanced settings hold grid, material and solver controls. Only Run starts propagation. Stop is cooperative. Results show saved fields and XY/XZ/YZ when available, with Physical or Square aspect. In Files, upload your experiment/result through the sidebar, refresh the list, and select Open. Drive is optional.

**Limits:** 256 MiB per result; 2.5 GiB working plus resident results; 512 MiB runtime reserve; 60-second cooperative PR limit; 128 MiB ZIP import/export. Existing GPU restrictions remain; this CPU setup installs no GPU software. Full x–y transport and private large-result backups are outside this preview. This is an unpublished source build, not the released rc3 wheel.
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "LCProp-PR-workbench.ipynb"}},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)},
                  {"cell_type": "code", "metadata": {"cellView": "form"}, "source": code.splitlines(True),
                   "execution_count": None, "outputs": []}]}
    path = root / "notebooks/pr_workbench_source.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    shutil.copy2(path, destination / "LCProp-PR-workbench.ipynb")
    (destination / "artifact-pins.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.wheel, args.destination)))
