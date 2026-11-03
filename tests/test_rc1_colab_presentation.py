"""Notebook-only presentation checks; no LCProp execution or installation."""
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch
import ipywidgets as W

NOTEBOOK = Path(os.environ.get('RC_NOTEBOOK', 'notebooks/lc_static_pypi_rc1_colab_qualification.ipynb'))

def cells():
    return json.loads(NOTEBOOK.read_text())['cells']

def test_reuse_widgets_and_record_without_callbacks():
    class Controller:
        disposed = False
        def __init__(self):
            self.completed = object()
            self.preview_host = W.VBox([W.HTML('canvas stand-in')])
            self.beam2_form = W.Box(layout=W.Layout(display='none'))
            self.button = W.Button()
            self.button.on_click(lambda _: None)
            self.configure = W.Accordion([W.VBox([self.beam2_form, self.preview_host])])
            self.widget = W.VBox([self.configure, W.VBox([self.button]), W.VBox([W.Output()])])
    app = Controller()
    record = app.completed
    callbacks = tuple(app.button._click_handlers.callbacks)
    module = types.ModuleType('lcprop.notebook')
    module.InteractiveLC = Controller
    shown = []
    ns = dict(_rc_ready=True, app_rc1=app, _rc_mode='interactive')
    code = ''.join(cells()[2]['source'])
    with patch.dict(sys.modules, {'lcprop.notebook': module}), patch('IPython.display.display', side_effect=shown.append), patch('IPython.display.clear_output'):
        exec(code, ns)
        view = ns['_rc_presentation']
        exec(code, ns)
    assert shown == [view, view]
    assert view.children == (app.configure.children[0], *app.widget.children[1:])
    assert app.completed is record
    assert tuple(app.button._click_handlers.callbacks) == callbacks
    assert app.beam2_form.layout.display == 'none'
    assert app.preview_host.layout.overflow is None
    assert view.layout.overflow == 'visible'
    assert view.children[0].layout.height == 'auto'

def test_static_backend_before_new_controller():
    events = []
    class Controller:
        disposed = False
        def __init__(self):
            events.append('controller')
            self.preview_host = W.VBox()
            self.configure = W.Accordion([W.VBox([self.preview_host])])
            self.widget = W.VBox([self.configure, W.VBox(), W.VBox()])
    module = types.ModuleType('lcprop.notebook'); module.InteractiveLC = Controller
    shell = types.SimpleNamespace(run_line_magic=lambda *args: events.append(args))
    code = ''.join(cells()[2]['source']).replace('MODE = "interactive"', 'MODE = "static"')
    with patch.dict(sys.modules, {'lcprop.notebook': module}), patch('IPython.display.display'), patch('IPython.display.clear_output'):
        exec(code, dict(_rc_ready=True, get_ipython=lambda: shell))
    assert events == [('matplotlib', 'inline'), 'controller']

def test_output_free_and_no_automatic_science():
    for cell in cells():
        assert not cell.get('outputs')
        assert cell.get('execution_count') is None
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(code, str(NOTEBOOK), 'exec')
            assert '._run(' not in code and '.click(' not in code
    text = ''.join(cells()[0]['source'])
    assert 'Restart session' in text
