"""New-user acceptance checks; no cluster or scientific execution."""
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QFormLayout

from lcprop.gui.help import HELP_TOPICS, ProductHelpButton
from lcprop.gui.panels.beam_panel import BeamPanel
from lcprop.gui.remote_execution import RemoteExecutionDialog, RemoteExecutionControls
from lcprop.runners.cluster_profiles import ClusterCatalog, load_cluster_profiles, write_cluster_profiles
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from tests.test_remote_cluster_setup import _cluster

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv('LCPROP_CLUSTER_CONFIG', str(tmp_path / 'isolated.toml'))


def test_installation_sequence_and_qualified_modes():
    text = (ROOT/'docs/user/quick_start.md').read_text()
    steps = ['mkdir lcprop-work', 'cd lcprop-work', 'python3.12 -m venv .venv',
             'git clone https://github.com/mcroning/LaunchPlane.git',
             'git clone --branch feature/pr-second-order-static https://github.com/mcroning/LCProp.git',
             'python -m pip install ./LaunchPlane', 'cd LCProp', "python -m pip install '.[gui]'"]
    offsets = [text.index(step) for step in steps]
    assert offsets == sorted(offsets)
    assert 'Python 3.10 or newer' in text and 'Python 3.11 or newer' in text
    assert 'developer editable' in text and 'non-editably' in text
    assert 'clean deployable Git' in text
    assert 'step-size, timestep, precision, and model convergence' in text


@pytest.mark.parametrize('name', ['README.md','docs/README.md','docs/STATUS.md','docs/user/quick_start.md','docs/user/user_guide.md'])
def test_current_touched_docs_use_product_name_and_defined_slice_terms(name):
    text = (ROOT/name).read_text()
    assert 'LaunchPane' not in text
    assert 'MPR' not in text


def test_empty_catalog_and_override_never_write_startup(app, tmp_path):
    real = tmp_path/'real.toml'
    original = write_cluster_profiles(ClusterCatalog((_cluster(),), 'alpha', real))
    before = real.read_bytes()
    controls = RemoteExecutionControls()
    assert not controls.slurm_available
    assert controls.cluster_selector.count() == 0
    assert 'first user-local profile' in controls.availability_label.text()
    assert not Path(os.environ['LCPROP_CLUSTER_CONFIG']).exists()
    assert real.read_bytes() == before and load_cluster_profiles(real) == original
    controls.close()


def test_explicit_defaults_roundtrip_and_edit_preserves_other_resource(app, tmp_path):
    path = tmp_path/'catalog.toml'
    catalog = write_cluster_profiles(ClusterCatalog((_cluster(),_cluster('beta')), 'alpha', path))
    before = path.read_bytes()
    dialog = RemoteExecutionDialog(catalog)
    assert path.read_bytes() == before
    assert str(path) in dialog.config_location.text()
    assert dialog.default_cluster_check.isChecked()
    dialog.saved_resource.setCurrentIndex(dialog.saved_resource.findData('GPU'))
    assert not dialog.default_resource_check.isChecked()
    dialog.time_limit.setText('00:25:00')
    dialog.save_profile()
    saved = load_cluster_profiles(path)
    assert saved['alpha'].default_resource_profile == 'CPU'
    assert saved['beta'] == catalog['beta']
    dialog.saved_resource.setCurrentIndex(dialog.saved_resource.findData('GPU'))
    dialog.default_resource_check.setChecked(True)
    dialog.default_cluster_check.setChecked(False)
    dialog.save_profile()
    saved = load_cluster_profiles(path)
    assert saved.default_cluster is None
    assert saved['alpha'].default_resource_profile == 'GPU'
    dialog.default_resource_check.setChecked(False)
    dialog.save_profile()
    assert load_cluster_profiles(path)['alpha'].default_resource_profile is None
    dialog.saved_cluster.setCurrentIndex(dialog.saved_cluster.findData('beta'))
    dialog.default_cluster_check.setChecked(True)
    dialog.save_profile()
    assert load_cluster_profiles(path).default_cluster == 'beta'
    dialog.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_beam_canvas_allocation_secondary_controls_and_boundary_identity(app, enabled):
    panel = BeamPanel(input_screens_enabled=enabled)
    before = panel.beam_stack_definition
    panel.resize(1200, 640);panel.show();app.processEvents()
    assert panel.beam_tabs.currentIndex() == 0
    assert panel.launch_plane_widget.isVisible()
    assert panel.launch_plane_widget.width() >= panel.width() * .9
    assert not panel.input_screen_editor.isVisible()
    assert panel.beam_tabs.count() == (3 if enabled else 2)
    panel.beam_tabs.setCurrentIndex(1);app.processEvents()
    assert panel.boundary_mode.isVisible()
    for label, value in [('None','periodic'),('Sponge','sponge'),('Tukey','tukey')]:
        panel.boundary_mode.setCurrentIndex(panel.boundary_mode.findText(label))
        assert panel.optical_boundary().mode == value
    if enabled:
        panel.beam_tabs.setCurrentIndex(2);app.processEvents()
        assert panel.input_screen_editor.isVisible()
    assert panel.beam_stack_definition == before
    panel.close()


@pytest.mark.parametrize('cls', [LCPropMainWindow, PRMainWindow])
@pytest.mark.parametrize('size', [(1200,760),(1450,900)])
def test_headers_fit_and_controls_keep_readable_sizes(app, cls, size):
    window = cls();window.resize(*size);window.show();app.processEvents()
    assert window.width() <= size[0]
    controls = [window.execution_target_selector,window.preview_button,window.run_button,
                window.help_button,window.experiment_file_buttons]
    for widget in controls:
        assert widget.isVisible()
        assert widget.width() >= widget.minimumSizeHint().width()
        point = widget.mapTo(window, QPoint(0,0))
        assert 0 <= point.x() and point.x()+widget.width() <= window.width()
    if isinstance(window, PRMainWindow):
        window.tabs.setCurrentWidget(window.evolution_panel);app.processEvents()
        form = window.evolution_panel.findChild(QFormLayout)
        assert form.fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow
        assert form.rowWrapPolicy() == QFormLayout.WrapLongRows
        assert window.evolution_panel.transport_model.width() >= window.evolution_panel.transport_model.sizeHint().width()
        for label in [window.evolution_panel.algorithm_status,window.evolution_panel.execution_guidance]:
            assert label.wordWrap()
            assert label.height() >= label.heightForWidth(label.width())
    window.close()


def test_help_scientific_semantics_and_installed_navigation(app):
    topics = {topic.key:topic.markdown for topic in HELP_TOPICS}
    assert all('MPR' not in text and 'LaunchPane' not in text for text in topics.values())
    assert 'periodic computational domain' in topics['boundary_conditions']
    assert all(word in topics['boundary_conditions'] for word in ['None','Sponge','Tukey','per-distance','soliton'])
    assert all(word in topics['beam_focusing'] for word in ['Every Gaussian','waist','curvature','upstream'])
    assert 'Full-transverse static' in topics['fanning_scattering'] and 'Reduced static currently does not' in topics['fanning_scattering']
    assert all(word in topics['fanning_scattering'] for word in ['Strength','correlation length','seed','canonical slab'])
    assert 'Local only' in topics['slurm'] and 'clean deployable' in topics['slurm']
    assert 'does not waive' in topics['results'] and 'nonexistence' in topics['results']
    assert 'Image Amplification' in topics['results'] and 'input and' in topics['results']
    for application in ['lc','pr']:
        button = ProductHelpButton(application=application)
        dialog = button.open_topic('slurm')
        text = dialog.browser.toPlainText()
        assert ('Current LC Slurm capability' in text) == (application == 'lc')
        assert 'https://github.com/mcroning/LCProp/blob/feature/pr-second-order-static/docs/user/user_guide.md' in dialog.browser.toHtml()
        dialog.close();button.close()


def test_profile_editor_fits_and_keeps_actions_accessible(app, tmp_path):
    dialog = RemoteExecutionDialog(ClusterCatalog(config_path=tmp_path/'new.toml'))
    dialog.resize(760,720);dialog.show();app.processEvents()
    assert dialog.height() <= 720
    assert dialog.form_scroll.widgetResizable()
    for button in [dialog.save_button,dialog.close_button]:
        point = button.mapTo(dialog, QPoint(0,0))
        assert point.y()+button.height() <= dialog.height()
    dialog.form_scroll.ensureWidgetVisible(dialog.setup_commands)
    assert dialog.setup_commands.isVisible()
    dialog.close()


def test_configuration_documentation_limits_isolation_and_preserves_neutral_dialogs():
    guide = (ROOT/'docs/user/user_guide.md').read_text()
    assert 'LCPROP_CLUSTER_CONFIG="$(mktemp -d)/clusters.toml"' in guide
    assert 'does not isolate native' in guide and 'empty initial directory' in guide
    assert 'outside the package/repository' in guide
