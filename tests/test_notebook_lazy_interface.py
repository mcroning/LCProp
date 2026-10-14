"""The candidate package initializer must not pull in optional UI dependencies."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_lazy_package_and_both_paths_in_isolated_process():
    script = r'''
import sys, importlib.abc
sys.path.insert(0, sys.argv[1])
blocked = {'ipywidgets', 'matplotlib', 'IPython', 'PySide6', 'PyQt6', 'cupy', 'launchplane'}
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Optional import: ' + fullname)
sys.meta_path.insert(0, NoOptional())
import lcprop
import lcprop.notebook as notebook
assert 'lcprop.notebook.interactive_lc' not in sys.modules
assert 'InteractiveLC' in dir(notebook)
from lcprop.notebook import InteractiveLC
from lcprop.notebook.interactive_lc import InteractiveLC as Legacy
assert InteractiveLC is Legacy is notebook.InteractiveLC
assert not blocked.intersection(sys.modules)
try:
    notebook.not_a_public_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Missing AttributeError')
print('PASS lazy import, identical aliases, no optional dependencies')
'''
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT / 'src')],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
