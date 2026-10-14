"""Release documentation contracts; no imports of the scientific engine."""
from pathlib import Path
from html.parser import HTMLParser
import re
from urllib.parse import urlsplit, urljoin
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the declared core dependency.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_long_description_source_and_release_version():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['readme'] == 'README.md'
    assert project['version'] == '0.3.0'
    assert '0.3.0' in (ROOT / 'docs/notebook_package.md').read_text()


def test_installation_separates_production_candidate_and_development():
    text = (ROOT / 'README.md').read_text()
    section = text.split('## Installation', 1)[1].split('## Graphical applications', 1)[0]
    assert '**production PyPI** and apply to versions' in section
    assert 'published there' in section
    assert 'candidate is not yet uploaded' not in section
    for command in ('python -m pip install lcprop', 'python -m pip install "lcprop[colab]"',
                    'python -m pip install "lcprop[gui]"'):
        assert command in section
        assert section.index(command) < section.index('For developers only')
    assert '--index-url https://test.pypi.org' not in section
    assert 'does **not** install' in section and 'LaunchPlane' in section


def test_immutable_colab_link_and_scientific_scope():
    text = (ROOT / 'README.md').read_text()
    assert 'https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb' in text
    assert 'does **not** install or qualify the latest package version' in text
    assert '**exploratory**' in text
    assert 'do not establish stationary soliton formation' in text


def test_dependency_and_python_declarations_unchanged():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['requires-python'] == '>=3.10'
    assert project['dependencies'] == ['numpy', 'scipy', "tomli; python_version < '3.11'"]
    assert project['optional-dependencies'] == {
        'colab': ['matplotlib', 'ipywidgets', 'IPython'], 'gpu': ['cupy-cuda12x'],
        'gui': ['PySide6', 'pyqtgraph', 'matplotlib'], 'test': ['pytest>=8', 'Pillow']}


def _readme_targets(text):
    """Collect inline/reference Markdown and HTML link/image targets outside code."""
    text = re.sub(r'(?ms)^```.*?^```\s*$', '', text)
    targets = re.findall(r'!?\[[^\]]*\]\(\s*<?([^\s)>]+)', text)
    targets += re.findall(r'(?m)^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)', text)

    class References(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src') and value:
                    targets.append(value)

    References().feed(text)
    return targets


def test_readme_links_and_images_are_absolute_on_both_hosts():
    targets = _readme_targets((ROOT / 'README.md').read_text())
    assert targets
    for target in targets:
        assert urlsplit(target).scheme == 'https' and urlsplit(target).netloc, target
        for base in ('https://pypi.org/project/lcprop/0.3.0/',
                     'https://github.com/mcroning/LCProp/blob/main/README.md'):
            assert urljoin(base, target) == target


def test_link_inventory_detects_relative_markdown_and_html_images():
    targets = _readme_targets('[Docs](docs/README.md)\n![Plot](images/plot.png)\n'
                              '[guide]: docs/user/guide.md\n'
                              '<a href="relative.html">Link</a><img src="plot.svg">')
    assert set(targets) == {'docs/README.md', 'images/plot.png',
                            'docs/user/guide.md', 'relative.html', 'plot.svg'}


def test_release_status_is_not_a_time_sensitive_publication_claim():
    text = (ROOT / 'README.md').read_text()
    for stale in ('not yet uploaded', 'not yet published', 'still require their own'):
        assert stale not in text
