import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'frontend'))
@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Scientific execution prohibited')
    monkeypatch.setattr('lcprop.runners.local.LocalRunner.run_operation',forbidden)
    monkeypatch.setitem(sys.modules,'cupy',None)


def pytest_addoption(parser):
    parser.addoption('--pr-saved-fixtures',default=None,help='JSON map of optional saved ZIP paths; checked against saved-fixture-pins.json')


@pytest.fixture
def saved_archive(request):
    import os,json,hashlib
    def locate(name):
        manifest=request.config.getoption('--pr-saved-fixtures') or os.environ.get('PR_SAVED_FIXTURES')
        if not manifest:
            pytest.skip('Optional historical saved-file test: set --pr-saved-fixtures or PR_SAVED_FIXTURES; synthetic tests run separately')
        mapping=json.loads(Path(manifest).read_text())
        path=Path(mapping[name])
        pins=json.loads(Path(__file__).with_name('saved-fixture-pins.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pins[name]:
            pytest.fail('Saved fixture checksum mismatch: '+name)
        return path
    return locate
