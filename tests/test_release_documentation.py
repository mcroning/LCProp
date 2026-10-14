"""Release documentation contracts; no imports of the scientific engine."""
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the declared core dependency.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_long_description_source_and_release_version():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['readme'] == 'README.md'
    assert project['version'] == '0.3.0rc2'
    assert '0.3.0rc2' in (ROOT / 'docs/notebook_package.md').read_text()


def test_installation_separates_production_candidate_and_development():
    text = (ROOT / 'README.md').read_text()
    section = text.split('## Installation', 1)[1].split('## Graphical applications', 1)[0]
    assert 'after LCProp is published there' in section
    assert 'candidate is not yet uploaded' in section
    for command in ('python -m pip install lcprop', 'python -m pip install "lcprop[colab]"',
                    'python -m pip install "lcprop[gui]"'):
        assert command in section
        assert section.index(command) < section.index('For developers only')
    assert '--index-url https://test.pypi.org' not in section
    assert 'does **not** install' in section and 'LaunchPlane' in section


def test_immutable_colab_link_and_scientific_scope():
    text = (ROOT / 'README.md').read_text()
    assert 'https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb' in text
    assert 'does **not** install or qualify RC2' in text
    assert '**exploratory**' in text
    assert 'do not establish stationary soliton formation' in text


def test_dependency_and_python_declarations_unchanged():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['requires-python'] == '>=3.10'
    assert project['dependencies'] == ['numpy', 'scipy', "tomli; python_version < '3.11'"]
    assert project['optional-dependencies'] == {
        'colab': ['matplotlib', 'ipywidgets', 'IPython'], 'gpu': ['cupy-cuda12x'],
        'gui': ['PySide6', 'pyqtgraph', 'matplotlib'], 'test': ['pytest>=8', 'Pillow']}
