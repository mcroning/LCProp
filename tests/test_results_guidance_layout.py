"""Synthetic display/widget checks only; no application window or solver."""
import os,unittest,hashlib,io
from matplotlib.backend_bases import MouseEvent
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QRect
from PySide6.QtWidgets import QApplication,QAbstractSpinBox,QComboBox
from lcprop.gui.views.image_view import ImageView
from lcprop.gui.workspace import Workspace
from lcprop.products.data_model import FieldData,Geometry,RunData,FieldCollection
from lcprop.gui.views.longitudinal_pane import LongitudinalPane
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.presentation_guidance import unified_support_guidance,TD_ELECTRICAL_GUIDANCE

def synthetic(axes=('x','y'),units=None):
    return FieldData(key='synthetic',display_name='Synthetic display fixture',
        data=np.arange(32,dtype=float).reshape(8,4),axes=axes,kind='intensity',
        units=units or {axes[0]:'um',axes[1]:'um'},value_unit='a.u.')

def pump(app):
    for _ in range(4):app.processEvents()

class ResultsGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('QT_QPA_PLATFORM')=='offscreen'
        cls.app=QApplication.instance() or QApplication(['display-only'])
        font=cls.app.font();font.setPointSize(12);cls.app.setFont(font)

    def display(self,w,width=1200,height=600):
        w.resize(width,height);w.show();pump(self.app);self.addCleanup(w.close);return w

    def check_aspect(self,v,ratio):
        v.draw()
        box=v.ax.get_window_extent();bar=v.colorbar.ax.get_window_extent()
        self.assertAlmostEqual(box.width/box.height,ratio,places=5)
        self.assertAlmostEqual(box.y0,bar.y0,places=5)
        self.assertAlmostEqual(box.height,bar.height,places=5)
        self.assertGreater(box.width,0);self.assertGreater(box.height,0)

    def test_physical_aspect_data_and_coordinate_mapping_survive_resize_swap(self):
        for axes,units,extent in [(('x','y'),{'x':'um','y':'um'},(-4,4,-1,1)),
                (('z','x'),{'z':'um','x':'um'},(0,8,-1,1)),
                (('z','x'),{'z':'mm','x':'um'},(0,.008,-1,1))]:
            with self.subTest(axes=axes,units=units):
                f=synthetic(axes,units);digest=hashlib.sha256(f.data.tobytes()).hexdigest()
                v=self.display(ImageView())
                selected=[];v.positionSelected.connect(lambda x,y:selected.append((x,y)))
                for swap in (False,True):
                    v.set_field(f,extent=extent,vmin=0,vmax=31,swap_axes=swap)
                    original_limits=(v.ax.get_xlim(),v.ax.get_ylim())
                    ids=(id(v.image),id(v.colorbar))
                    for width,height in ((500,500),(900,400),(500,900)):
                        v.resize(width,height);pump(self.app)
                        self.check_aspect(v,.25 if swap else 4.)
                        self.assertEqual((v.ax.get_xlim(),v.ax.get_ylim()),original_limits)
                        self.assertEqual((id(v.image),id(v.colorbar)),ids)
                        self.assertEqual(v.image.get_clim(),(0.,31.))
                        self.assertTrue(np.array_equal(v.image.get_array(),f.data if swap else f.data.T))
                        point=v._index_to_display_coordinates(2,1)
                        self.assertEqual(v._display_coordinates_to_index(*point),(2,1))
                        pixel=v.ax.transData.transform(point)
                        v._on_mouse_press(MouseEvent('button_press_event',v,*pixel,button=1))
                        self.assertEqual(selected[-1],(2,1))
                        png=io.BytesIO();v.figure.savefig(png,format='png')
                        self.assertTrue(png.getvalue().startswith(b'\x89PNG'))
                        self.assertEqual((v.ax.get_xlim(),v.ax.get_ylim()),original_limits)
                        self.assertEqual(hashlib.sha256(f.data.tobytes()).hexdigest(),digest)

    def test_pending_cut_labels_and_aspect_selector(self):
        workspace = self.display(Workspace())
        views = (workspace.image_pane.image_view, workspace.longitudinal_pane.xz_view,
                 workspace.longitudinal_pane.yz_view)
        self.assertGreaterEqual(workspace.spatial_selector.count(), 2)
        workspace.spatial_selector.setCurrentIndex(1)
        for view, labels in zip(views, (("x", "y"), ("z", "x"), ("z", "y"))):
            self.assertEqual((view.ax.get_xlabel(), view.ax.get_ylabel()), labels)
            self.assertIn("Output intensity", view.ax.get_title())
            self.assertFalse(view.image.get_visible())
            self.assertFalse(view.colorbar.ax.get_visible())

        cut = views[1]
        field = synthetic(("z", "x"), {"z": "mm", "x": "um"})
        cut.set_field(field, extent=(0, .008, -1, 1), vmin=0, vmax=31)
        limits = cut.ax.get_xlim(), cut.ax.get_ylim()
        self.check_aspect(cut, 4.)
        workspace.aspect_selector.setCurrentIndex(1)
        self.check_aspect(cut, 1.)
        workspace.aspect_selector.setCurrentIndex(0)
        self.check_aspect(cut, 4.)
        self.assertEqual((cut.ax.get_xlim(), cut.ax.get_ylim()), limits)
        self.assertTrue(np.array_equal(cut.image.get_array(), field.data.T))

    def test_placeholder_is_stable_but_not_scientific_aspect_evidence(self):
        v=self.display(ImageView())
        for size in ((550,300),(400,700)):
            v.resize(*size);pump(self.app);self.check_aspect(v,1.)

    def test_display_options_retain_controls_states_and_scales(self):
        w=self.display(Workspace())
        self.assertFalse(w.image_pane.controls_widget.isVisible())
        self.assertFalse(w.longitudinal_pane.options_widget.isVisible())
        settings=dict(w.display_scales.settings)
        # A data-owned hidden child stays hidden when the parent is disclosed.
        w.longitudinal_pane.scale_controls.hide()
        w.display_options.setChecked(True);pump(self.app)
        self.assertTrue(w.image_pane.fit_button.isVisible())
        self.assertTrue(w.image_pane.scale_controls.isVisible())
        self.assertTrue(w.longitudinal_pane.orientation.isVisible())
        self.assertFalse(w.longitudinal_pane.scale_controls.isVisible())
        w.display_options.setChecked(False);pump(self.app)
        self.assertEqual(w.display_scales.settings,settings)
        w.trajectory_disclosure.setChecked(True);pump(self.app)
        self.assertTrue(w.trajectory_player.isVisible())

    def test_results_normal_tall_and_small_keep_axes_inside_canvas(self):
        w=self.display(Workspace())
        fields=[(w.image_pane.image_view,('x','y'),(-2,2,-2,2)),
                (w.longitudinal_pane.xz_view,('z','x'),(0,8,-1,1)),
                (w.longitudinal_pane.yz_view,('z','y'),(0,8,-1,1))]
        for v,axes,extent in fields:v.set_field(synthetic(axes),extent=extent,vmin=0,vmax=31)
        for size in ((1200,600),(1450,760),(1200,900),(760,460)):
            w.resize(*size);pump(self.app)
            if size[0]>=1200:
                self.assertEqual(w.fields_scroll.verticalScrollBar().maximum(),0,size)
            for i,(v,axes,extent) in enumerate(fields):
                if i: w.longitudinal_pane.cuts.setCurrentIndex(i-1);pump(self.app)
                self.check_aspect(v,1. if axes==('x','y') else 4.)
                box=v.ax.get_window_extent();canvas=v.figure.bbox
                self.assertGreaterEqual(box.x0,0);self.assertGreaterEqual(box.y0,0)
                self.assertLessEqual(box.x1,canvas.width+1)
                self.assertLessEqual(box.y1,canvas.height+1)
                for artist in (v.ax,v.colorbar.ax):
                    tight=artist.get_tightbbox(v.get_renderer())
                    self.assertGreaterEqual(tight.x0,-2)
                    self.assertGreaterEqual(tight.y0,-2)
                    self.assertLessEqual(tight.x1,canvas.width+2)
                    self.assertLessEqual(tight.y1,canvas.height+2)

    def test_cut_tab_visibility_tracks_synthetic_data_without_mutation(self):
        pane=self.display(LongitudinalPane(),700,500)
        data=np.arange(60,dtype=float).reshape(5,4,3)
        geometry=Geometry(x=np.linspace(-1,1,4),y=np.linspace(-1,1,3),z=np.linspace(0,8,5))
        f=FieldData('synthetic-volume','Synthetic volume',data,('z','x','y'),'intensity',
                    units={'x':'um','y':'um','z':'um'})
        run=RunData('static',geometry=geometry,fields=FieldCollection([('synthetic-volume',f)]))
        digest=hashlib.sha256(data.tobytes()).hexdigest()
        pane.set_run_data(run);pump(self.app)
        for index in (0,1,0):
            pane.cuts.setCurrentIndex(index);pump(self.app)
            self.assertTrue(pane.cuts.currentWidget().isVisible())
            self.assertFalse(pane.cuts.widget(1-index).isVisible())
        pane.set_independent(True);pump(self.app);self.assertFalse(pane.cuts.isVisible())
        pane.set_independent(False);pump(self.app);self.assertTrue(pane.cuts.isVisible())
        pane.orientation.setCurrentIndex(1);pump(self.app)
        self.assertEqual(hashlib.sha256(data.tobytes()).hexdigest(),digest)
        pane.set_run_data(RunData('static'));pump(self.app)
        self.assertFalse(pane.cuts.isVisible())

    def test_existing_guidance_wraps_without_clipping_or_numeric_changes(self):
        w=self.display(PREvolutionPanel(),1450,900)
        numeric=w.findChildren(QAbstractSpinBox)
        values=[(e.value(),e.minimum(),e.maximum(),e.singleStep()) for e in numeric]
        messages=[(w.algorithm_status.text(),w.execution_guidance.text()),
            ('Unified nonlinear Static: published optical-first; certified mixed precision. No coupled passes or replay.',
             unified_support_guidance()),
            ('Experimental: compatible architecture; specialized Image Amplification validation is pending.',
             'Slurm recommended except for small smoke cases.\n'+TD_ELECTRICAL_GUIDANCE)]
        for width in (1450,900,650):
            w.resize(width,650)
            for status,guidance in messages:
                w.algorithm_status.setText(status);w.execution_guidance.setText(guidance);pump(self.app)
                for label in (w.algorithm_status,w.execution_guidance):
                    self.assertTrue(label.hasHeightForWidth())
                    need=label.fontMetrics().boundingRect(QRect(0,0,label.width(),10000),Qt.TextWordWrap,label.text()).height()
                    self.assertGreaterEqual(label.height()+2,need,(width,label.text(),label.size(),need))
                    self.assertGreater(label.width(),250)
                self.assertEqual(values,[(e.value(),e.minimum(),e.maximum(),e.singleStep()) for e in numeric])
                for e in numeric:
                    if e.isVisible():self.assertLess(e.width(),420)
        w.resize(650,300);pump(self.app)
        self.assertGreater(w.form_scroll.verticalScrollBar().maximum(),0)

if __name__=='__main__':unittest.main()
