"""Synthetic staging checks: no installation or scientific execution."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import zipfile

import pytest


@pytest.fixture
def staging(tmp_path):
    root = Path(__file__).parents[1]
    for folder in ('tools', 'notebooks'):
        (tmp_path / folder).mkdir()
    shutil.copy2(root / 'tools/stage_pr_workbench.py', tmp_path / 'tools/stage_pr_workbench.py')
    shutil.copy2(root / 'notebooks/pr_workbench_open.py', tmp_path / 'notebooks/pr_workbench_open.py')
    return tmp_path, runpy.run_path(str(tmp_path / 'tools/stage_pr_workbench.py'))['stage']


@pytest.mark.parametrize('version', ['0.5.0rc3', '0.5.0rc4'])
def test_generated_opener_verifies_supplied_wheel_name_and_bytes(staging, version):
    root, stage = staging
    wheel = root / f'lcprop-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('lcprop/example.py', '# synthetic only\n')
        archive.writestr(f'lcprop-{version}.dist-info/METADATA',
                         f'Metadata-Version: 2.1\nName: lcprop\nVersion: {version}\n')
    hashes = stage(wheel, root / 'delivery')
    opener_path = root / 'notebooks/pr_workbench_open.py'
    opening = runpy.run_path(str(opener_path), run_name='staging_test')
    assert opening['WHEEL_NAME'] == wheel.name
    data = opening['_verified_wheel'](root / 'delivery/LCProp-PR-workbench-setup.zip')
    assert data == wheel.read_bytes()
    assert hashlib.sha256(data).hexdigest() == hashes['wheel']
    notebook = json.loads((root / 'delivery/LCProp-PR-workbench.ipynb').read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert len(cells) == 1 and cells[0]['outputs'] == []
    assert ''.join(cells[0]['source']) == opener_path.read_text()


@pytest.mark.parametrize('name', ['other-1.0-py3-none-any.whl', 'lcprop-wheel.txt'])
def test_invalid_wheel_name_rejected_before_output_or_pin_changes(staging, name):
    root, stage = staging
    wheel = root / name
    wheel.write_bytes(b'not a wheel')
    opener = root / 'notebooks/pr_workbench_open.py'
    before = opener.read_bytes()
    destination = root / 'delivery'
    destination.mkdir()
    old_kit = destination / 'LCProp-PR-workbench-setup.zip'
    old_kit.write_bytes(b'previous kit retained')
    with pytest.raises(ValueError, match='LCProp .whl'):
        stage(wheel, destination)
    assert opener.read_bytes() == before
    assert old_kit.read_bytes() == b'previous kit retained'
    assert not (destination / 'LCProp-PR-workbench.ipynb').exists()
