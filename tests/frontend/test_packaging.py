"""Portable packaging needs explicit pinned inputs; no private checkout paths."""
import hashlib, importlib.util, json, os, sys, zipfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import stage_pr_capabilities as packaging
import prepare_pr_frontend_tests as preparation


def kit():
    path=Path(os.environ.get('PR_CAPABILITY_KIT',str(ROOT/'delivery-v4/LCProp-PR-capabilities-setup.zip')))
    if not path.is_file():pytest.skip('Explicit local kit required; see frontend/README.md')
    return path


def test_packaging_has_no_implicit_private_input_or_overwrite(tmp_path):
    with pytest.raises(ValueError,match='Supply exactly one'):packaging.engine_bytes()
    source=kit();digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='checksum'):packaging.engine_bytes(input_kit=source,input_kit_sha256='bad')
    with pytest.raises(ValueError,match='ce915366'):
        bad=tmp_path/'bad.whl';bad.write_bytes(b'wrong engine');packaging.engine_bytes(engine_wheel=bad)
    with pytest.raises(FileExistsError):packaging.stage(input_kit=source,input_kit_sha256=digest,output=tmp_path)


def test_deterministic_build_from_supplied_kit_and_exact_wheel(tmp_path):
    source=kit();digest=hashlib.sha256(source.read_bytes()).hexdigest()
    out=tmp_path/'from-kit';packaging.stage(input_kit=source,input_kit_sha256=digest,output=out)
    other=tmp_path/'from-wheel';packaging.stage(engine_wheel=out/packaging.ENGINE_NAME,output=other)
    for file in out.iterdir():assert file.read_bytes()==(other/file.name).read_bytes()
    assert (out/'LCProp-PR-capabilities-setup.zip').read_bytes()==source.read_bytes()


def test_extract_only_test_preparation_checks_every_member(tmp_path):
    source=kit();digest=hashlib.sha256(source.read_bytes()).hexdigest()
    out=tmp_path/'import-target';receipt=preparation.prepare(source,digest,out)
    assert receipt['engine_sha256']==packaging.ENGINE_SHA
    assert (out/'lcprop/__init__.py').is_file()
    with pytest.raises(FileExistsError):preparation.prepare(source,digest,out)
    with pytest.raises(ValueError,match='checksum'):preparation.prepare(source,'bad',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_headless_vendored_source_provenance():
    provenance=json.loads((ROOT/'frontend/launchplane/PROVENANCE.json').read_text())
    assert provenance['source_wheel_sha256']==packaging.LP_SHA
    for name,digest in provenance['files'].items():
        assert hashlib.sha256((ROOT/'frontend/launchplane'/name).read_bytes()).hexdigest()==digest
    assert set(provenance['files'])=={'__init__.py','model.py','serialization.py','LICENSE'}
