"""Focused offscreen sizing checks; never instantiate PhysicsPanel or a solver.

Only its actual presentation-construction prefix is extracted into a QWidget
fixture. This validates the form repair, not native application acceptance.
"""
import ast
import os
from pathlib import Path
import unittest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFormLayout
except ImportError as exc:
    raise unittest.SkipTest("Optional PySide6 GUI dependency unavailable") from exc
from lcprop.gui.layout import bounded_form
from lcprop.gui.panels.helpers import double_spin_box

SOURCE = Path(__file__).resolve().parents[1] / 'src/lcprop/lc/gui/panels/physics_panel.py'
FIELDS = ('ne', 'no', 'K_pN', 'delta_epsilon', 'V_bias', 'theta_bc')


def form_fixture(native_policy_baseline=False):
    """Execute source presentation statements through layout.addLayout(form).

    Derived widgets/readout and their scientific connections are not in this
    fixture. No method of PhysicsPanel is imported, replaced or invoked.
    """
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PhysicsPanel')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    prefix = []
    found = False
    for node in init.body[1:]:  # QWidget fixture is initialized normally below.
        prefix.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'addLayout'):
            found = True
            break
    assert found
    body = ast.Module(body=prefix, type_ignores=[])
    widget = QWidget()
    env = dict(self=widget, QVBoxLayout=QVBoxLayout, QFormLayout=QFormLayout,
               double_spin_box=double_spin_box, bounded_form=bounded_form)
    exec(compile(body, str(SOURCE), 'exec'), env)
    form = env['form']
    if native_policy_baseline:
        # Restore the old bare-form outcome under documented QMacStyle policy.
        # This is policy emulation, not a native-platform reproduction.
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
    widget.resize(1200, 760)
    widget.show()
    QApplication.processEvents()
    return widget, form


def fits(editor):
    line = editor.lineEdit()
    margins = line.textMargins()
    available = line.contentsRect().width() - margins.left() - margins.right() - 4
    needed = line.fontMetrics().horizontalAdvance(line.text())
    return needed <= available, needed, available


import json
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit
from lcprop.gui.panels.grid_panel import GridPanel
from lcprop.pr.gui.grid_panel import PRGridPanel
from lcprop.pr.gui.material_panel import PRMaterialPanel
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.image_input_panel import PRImageInputPanel
from lcprop.pr.gui.resource_estimator_panel import PRResourceEstimatorPanel
from lcprop.gui.workspace import Workspace

MEASUREMENTS = {}

class CoordinatedLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('QT_QPA_PLATFORM') == 'offscreen'
        cls.app = QApplication.instance() or QApplication(['isolated-layout'])

    def show(self, widget, width=1400, height=760):
        widget.resize(width, height); widget.show(); self.app.processEvents()
        self.addCleanup(widget.close)
        return widget

    def test_physics_readable_bounded_and_unchanged(self):
        w, form = form_fixture()
        self.addCleanup(w.close)
        values = (9.876543, .123456, 999.999, 99.123456, .123456, 1.570799)
        rows = []
        for name, val in zip(FIELDS, values):
            e = getattr(w, name)
            original = (e.minimum(), e.maximum(), e.decimals(), e.singleStep())
            for value in (e.value(), val):
                e.setValue(value)
                before = e.value()
                for focus in (False, True, False):
                    e.setFocus() if focus else e.clearFocus(); self.app.processEvents()
                    self.assertTrue(fits(e)[0], (name, focus, fits(e)))
                    self.assertEqual(e.value(), before)
                    self.assertLess(e.width(), 320)
                self.assertEqual(original, (e.minimum(),e.maximum(),e.decimals(),e.singleStep()))
            rows.append((name,e.width(),fits(e)))
        self.assertEqual(form.fieldGrowthPolicy(),QFormLayout.FieldsStayAtSizeHint)
        MEASUREMENTS['physics'] = rows

    def test_grid_five_digit_display_without_new_range(self):
        w=self.show(GridPanel())
        for e in (w.Nx,w.Ny):
            self.assertEqual((e.minimum(),e.maximum(),e.value()),(16,4096,64))
            # Display fixture only: inject glyphs, never set an out-of-range value.
            e.lineEdit().setText('99999')
            self.assertTrue(fits(e)[0],fits(e))
            e.lineEdit().setText('64'); e.interpretText()
            self.assertEqual(e.value(),64)
        MEASUREMENTS['grid']=[w.Nx.width(),w.Ny.width(),w.Nx.maximum()]

    def test_pr_numeric_bounds_and_long_choices(self):
        result={}
        for cls in (PRMaterialPanel, PRGridPanel, PREvolutionPanel):
            w=self.show(cls())
            sizes=[]
            editors=w.findChildren(QAbstractSpinBox)
            for e in editors: e.show()
            self.app.processEvents()
            for e in editors:
                if not e.isVisible(): continue
                before=e.value(); e.setFocus();self.app.processEvents()
                self.assertTrue(fits(e)[0],(cls.__name__,e.text(),fits(e)))
                self.assertLess(e.width(),420)
                e.clearFocus(); self.app.processEvents(); self.assertEqual(e.value(),before)
                sizes.append(e.width())
            self.assertTrue(sizes)
            for combo in w.findChildren(QComboBox):
                if combo.isVisible():
                    needed=max((combo.fontMetrics().horizontalAdvance(combo.itemText(i)) for i in range(combo.count())),default=0)
                    self.assertGreaterEqual(combo.width(),needed+15,(cls.__name__,needed,combo.width()))
            result[cls.__name__]=sizes
        MEASUREMENTS['pr_numeric_widths']=result

    def test_input_choices_and_content_sized_actions(self):
        w=self.show(PRImageInputPanel())
        self.assertEqual(w.input_mode.currentData(),'gaussian_beams')
        for i in range(w.input_mode.count()):
            needed=w.input_mode.fontMetrics().horizontalAdvance(w.input_mode.itemText(i))
            self.assertGreaterEqual(w.input_mode.width(),needed+15)
        self.assertLess(w.input_mode.width(),650)
        w.input_mode.setCurrentIndex(1) # isolated presentation fixture, no Beam binding.
        self.app.processEvents()
        self.assertTrue(w.clear_roles.isVisible()); self.assertLess(w.clear_roles.width(),250)
        p=self.show(PRResourceEstimatorPanel())
        self.assertLess(p.estimate_button.width(),350)
        self.assertEqual(p.output.toPlainText(),'No estimate yet.')
        MEASUREMENTS['actions']=[w.input_mode.width(),w.clear_roles.width(),p.estimate_button.width()]

    def test_results_observed_sizes_and_smaller_fallback(self):
        w=self.show(Workspace(),1200,600)
        results=[]
        for width,height in ((1200,600),(1450,760),(800,400)):
            w.resize(width,height);self.app.processEvents()
            scrollbar=w.fields_scroll.verticalScrollBar()
            views=(w.image_pane.image_view,w.longitudinal_pane.xz_view,w.longitudinal_pane.yz_view)
            geometry=[(v.width(),v.height()) for v in views]
            if height>=600:
                self.assertEqual(scrollbar.maximum(),0,(width,height,scrollbar.maximum(),geometry))
                for v in views:
                    rect=v.mapTo(w.fields_page,v.rect().bottomRight())
                    self.assertLessEqual(rect.y(),w.fields_page.height())
                    self.assertGreaterEqual(v.height(),200)
            else:self.assertGreater(scrollbar.maximum(),0)
            results.append(dict(size=[width,height],scroll_max=scrollbar.maximum(),canvases=geometry))
        w.trajectory_disclosure.setChecked(True);self.app.processEvents()
        self.assertTrue(w.trajectory_player.isVisible())
        self.assertGreater(len(w.trajectory_player.findChildren(QWidget)),10)
        w.trajectory_disclosure.setChecked(False);self.app.processEvents()
        self.assertFalse(w.trajectory_player.isVisible())
        MEASUREMENTS['results']=results

if __name__=='__main__':
    result=unittest.main(exit=False)
    dest=os.environ.get('LAYOUT_MEASUREMENTS')
    if dest: Path(dest).write_text(json.dumps(MEASUREMENTS,indent=2)+'\n')
    raise SystemExit(not result.result.wasSuccessful())
