"""Actual researcher-facing setup cells; no solver execution or test installations."""
import json
from pathlib import Path
import sys
import importlib.metadata as md
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def cells(monkeypatch):
    import IPython.display
    import lcprop.notebook.interactive_lc as ui
    from lcprop.runners.local import LocalRunner
    def forbidden(*a, **k):
        pytest.fail('Setup must not execute the solver')
    monkeypatch.setattr(ui, 'execute', forbidden)
    monkeypatch.setattr(LocalRunner, 'run_operation', forbidden)
    monkeypatch.setattr(IPython.display, 'display', lambda *a, **k: None)
    monkeypatch.setattr(IPython.display, 'clear_output', lambda **k: None)
    calls = []
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    import builtins
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)
    monkeypatch.setattr(shell, 'run_line_magic', lambda name, args: calls.append((name, args)))
    nb = json.loads((ROOT / 'notebooks/lc_static_pypi_cpu.ipynb').read_text())
    return [shell.input_transformer_manager.transform_cell(''.join(nb['cells'][i]['source'])) for i in (1, 2)], calls


def test_install_and_initialize_qualified_request(cells):
    code, calls = cells; ns = {}
    for cell in code: exec(cell, ns)
    from lcprop.notebook.lc_static_demo import build_request
    assert ns['app'].request() == build_request()
    assert ns['app'].completed is None
    assert calls == [('pip', 'install -q \"lcprop[colab]==0.3.0\"')]
    assert not any(n in sys.modules for n in ('PySide6', 'launchplane', 'cupy'))


def test_repeat_preserves_controls_and_completed_snapshot(cells):
    code, _ = cells; ns = {}
    for cell in code: exec(cell, ns)
    app = ns['app']; marker = object(); app.completed = marker
    app.time_budget.value = 180
    exec(code[1], ns)
    for cell in code: exec(cell, ns)
    assert ns['app'] is app and app.completed is marker
    assert app.time_budget.value == 180


def test_incompatible_loaded_version_blocks_install_and_init(cells, monkeypatch):
    code, calls = cells; original = md.version
    monkeypatch.setattr(md, 'version', lambda name: '0.2.0' if name == 'lcprop' else original(name))
    with pytest.raises(RuntimeError, match='version is loaded'): exec(code[0], {})
    with pytest.raises(RuntimeError, match='0.3.0 is required'): exec(code[1], {})
    assert calls == []


def test_prior_session_version_is_not_silently_replaced(cells):
    code, calls = cells; old = object(); ns = {'app': old, '_lcprop_session_version': '0.2.0'}
    for cell in code:
        with pytest.raises(RuntimeError): exec(cell, ns)
    assert ns['app'] is old and calls == []


def test_install_failure_keeps_completed_app(cells, monkeypatch):
    code, _ = cells; ns = {}; exec(code[1], ns)
    app = ns['app']; marker = object(); app.completed = marker
    from IPython.core.interactiveshell import InteractiveShell
    def fail(*args): raise RuntimeError('Installation failed')
    monkeypatch.setattr(InteractiveShell.instance(), 'run_line_magic', fail)
    with pytest.raises(RuntimeError, match='Installation failed'): exec(code[0], ns)
    assert ns['app'] is app and app.completed is marker


def test_incompatible_app_not_overwritten(cells):
    code, _ = cells; old = object(); ns = {'app': old}
    with pytest.raises(RuntimeError, match='incompatible interface'): exec(code[1], ns)
    assert ns['app'] is old


def test_readable_cells_without_integrity_framework(cells):
    code, _ = cells
    assert all(len(c.splitlines()) <= 15 for c in code)
    for forbidden in ('urlopen', 'hashlib', 'sha256', 'ZipFile', '267', 'prepare_pypi_interface'):
        assert forbidden not in '\n'.join(code)
    nb = json.loads((ROOT / 'notebooks/lc_static_pypi_cpu.ipynb').read_text())
    assert all(not c.get('outputs') for c in nb['cells'])


def test_initialization_replaces_output_without_replacing_app(cells, monkeypatch):
    import IPython.display
    code, _ = cells; ns = {}; events = []
    monkeypatch.setattr(IPython.display, 'clear_output', lambda **k: events.append(('clear', k)))
    monkeypatch.setattr(IPython.display, 'display', lambda obj: events.append(('display', obj)))
    exec(code[1], ns); app = ns['app']; events.clear()
    exec(code[1], ns)
    assert events == [('clear', {'wait': True}), ('display', app.widget)]
    assert ns['app'] is app
