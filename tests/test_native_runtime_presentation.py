"""Native follow-up acceptance contracts with offscreen widgets and retained data."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QApplication, QLabel
from lcprop.gui.workspace import Workspace
from lcprop.gui.runtime_status import RuntimeStatusLabel
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from tests.test_stage1b_transparency import image_data


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_console_boundaries_do_not_leak_block_decoration(app):
    w = Workspace()
    w.tabs.setCurrentWidget(w.console)
    w.operation_boundary('Run <example>')
    w.append_console('step 1 < 5')
    w.append_console('step 2\nretained warning')
    w.operation_boundary('Continue')
    w.append_console('complete')
    blocks = []
    block = w.console.document().begin()
    while block.isValid():
        blocks.append(block)
        block = block.next()
    assert [b.text() for b in blocks][1:4] == ['step 1 < 5', 'step 2', 'retained warning']
    assert len(blocks) == 6
    for i, b in enumerate(blocks):
        boundary = i in (0, 4)
        assert b.begin().fragment().charFormat().fontWeight() == (QFont.Weight.Bold if boundary else QFont.Weight.Normal)
        assert b.blockFormat().topMargin() == (6 if boundary else 0)
        assert b.blockFormat().bottomMargin() == (4 if boundary else 0)
    assert '<hr' not in w.console.toHtml()
    assert blocks[0].text().startswith('[') and 'Run <example>' in blocks[0].text()
    assert w.tabs.currentWidget() is w.console
    cursor = w.console.textCursor()
    cursor.setPosition(0);cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
    w.console.setTextCursor(cursor)
    selected = cursor.selectedText()
    w.append_console('another progress message')
    assert w.console.textCursor().selectedText() == selected
    w.close()


@pytest.mark.parametrize('window_type,identity', [(LCPropMainWindow, 'LCProp LC'), (PRMainWindow, 'LCProp PR')])
def test_application_identity(app, window_type, identity):
    w = window_type()
    assert w.windowTitle() == identity
    assert identity in [label.text() for label in w.findChildren(QLabel)]
    w.close()


def test_scale_groups_independent_bidirectionally_and_locks_reset(app):
    w = Workspace();data = image_data()
    original = {k:f.data.copy() for k,f in data.fields.items()}
    w.set_run_data(data)
    xy, long = w.image_pane, w.longitudinal_pane
    x0 = xy.image_view.image.get_clim()
    l0 = long.xz_view.image.get_clim()
    assert xy.scales is not long.scales
    xy.scale_controls.lower.setText('2');xy.scale_controls.upper.setText('10')
    xy.scale_controls.apply_limits()
    assert xy.image_view.image.get_clim() == (2.,10.)
    assert long.xz_view.image.get_clim() == long.yz_view.image.get_clim() == l0
    long.scale_controls.lower.setText('3');long.scale_controls.upper.setText('18')
    long.scale_controls.apply_limits()
    assert long.xz_view.image.get_clim() == long.yz_view.image.get_clim() == (3.,18.)
    assert xy.image_view.image.get_clim() == (2.,10.)
    for pane in (xy,long):
        c = pane.scale_controls;c.mode.setCurrentIndex(c.mode.findData('locked'))
    w.set_run_data(image_data(100.))
    assert xy.image_view.image.get_clim() == (2.,10.)
    assert long.xz_view.image.get_clim() == long.yz_view.image.get_clim() == (3.,18.)
    w.reset_field_color_scales();w.set_run_data(image_data(100.))
    assert xy.image_view.image.get_clim() != (2.,10.)
    assert long.xz_view.image.get_clim() == long.yz_view.image.get_clim() == (0.,2300.)
    for k,f in data.fields.items():np.testing.assert_array_equal(f.data,original[k])
    w.close()


@pytest.mark.parametrize('width', [1200,1450])
@pytest.mark.parametrize('window_type', [LCPropMainWindow,PRMainWindow])
def test_runtime_messages_and_lifecycle_buttons_have_stable_geometry(app, width, window_type):
    w = window_type();w.resize(width,900);w.show();app.processEvents()
    workspace = w.results_panel.workspace
    labels = [w.runner_label, w.remote_execution_controls.availability_label,
              workspace.operation_status,workspace.result_ownership]
    if hasattr(w,'status_label'):labels.append(w.status_label)
    w.tabs.setCurrentWidget(w.results_panel);app.processEvents()
    workspace.tabs.setCurrentWidget(workspace.fields_splitter);app.processEvents()
    controls = [w.preview_button,w.run_button,w.continue_button,w.stop_button,w.help_button,
                w.remote_execution_controls.configure_button,w.tabs,workspace.tabs]
    def geometry():
        return [(c.mapTo(w,QPoint()).x(),c.mapTo(w,QPoint()).y(),c.width(),c.height()) for c in controls]
    before = geometry();hints = [label.sizeHint() for label in labels]
    messages = ['Preparing', 'Running', 'Runner: Local CPU Checkpoint ready: step 10, t=0.01',
                'Continue: additional material steps', 'Stopped / cancelled',
                'Completed', 'Failed: ' + 'detailed diagnostic ' * 20,
                'Slurm job 123456789 awaiting resource allocation on a configured cluster',
                'First line\nSecond line with detailed state']
    for text in messages:
        for label in labels:
            assert isinstance(label,RuntimeStatusLabel)
            label.setText(text)
            assert label.text() == text and text in label.toolTip()
            assert label.accessibleName() == text and not label.wordWrap()
        w.run_button.setText('Running…');w.stop_button.setText('Stopping…');w.stop_button.show()
        workspace.image_pane.set_td_time_indicator(text)
        app.processEvents();app.processEvents()
        assert [label.sizeHint() for label in labels] == hints
        assert geometry() == before
        w.stop_button.hide();workspace.image_pane.set_td_time_indicator(None)
        app.processEvents()
        assert geometry() == before
    w.close()
