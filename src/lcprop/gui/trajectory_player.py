"""Playback of retained samples through the common image pane; no science."""
from dataclasses import replace
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget,QHBoxLayout,QVBoxLayout,QPushButton,QSlider,QLabel,QDoubleSpinBox,QCheckBox,QComboBox,QFileDialog,QMessageBox
from lcprop.products.data_model import FieldData,FieldCollection
from lcprop.gui.views.display_scale import scale_key


class TrajectoryPlayer(QWidget):
    def __init__(self, workspace):
        super().__init__(workspace);self.workspace=workspace;self.artifact=None;self.run_data=None
        layout=QVBoxLayout(self);row=QHBoxLayout();layout.addLayout(row)
        self.play=QPushButton('Play');self.back=QPushButton('‹');self.next=QPushButton('›')
        self.quantity=QComboBox();self.quantity.currentIndexChanged.connect(self.select_quantity)
        self.quantity.setMinimumContentsLength(32)
        self.quantity.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        selection_row=QHBoxLayout();layout.insertLayout(0,selection_row)
        selection_row.addWidget(QLabel('Time evolution'));selection_row.addWidget(self.quantity,1)
        self.orientation=QComboBox();self.orientation.addItems(['z horizontal','z vertical'])
        self.orientation.setToolTip('Longitudinal time-evolution display only; retained frames and movie export are unchanged.')
        selection_row.addWidget(self.orientation)
        self.orientation.setEnabled(False)
        self.orientation.currentIndexChanged.connect(lambda:self.show_frame(self.slider.value()))
        self.restore=QPushButton('Return to spatial field views');self.slider=QSlider(Qt.Horizontal)
        self.speed=QDoubleSpinBox();self.speed.setRange(.1,60);self.speed.setValue(6);self.speed.setSuffix(' frames/second')
        self.loop=QCheckBox('Loop');self.scale=QComboBox();self.scale.addItems(['Linear — global scale','Log10 — global scale'])
        self.scale.setToolTip('Fixed range across all retained times. Log intensity uses one display-only floor eight decades below the global peak.')
        self.save=QPushButton('Save Movie…');self.save.clicked.connect(self.save_movie)
        for w in (self.play,self.back,self.slider,self.next,self.speed,self.loop):row.addWidget(w)
        export_row=QHBoxLayout();layout.addLayout(export_row)
        for w in (self.scale,self.save,self.restore):export_row.addWidget(w)
        self.label=QLabel('Time evolution unavailable: no accepted samples retained.');self.label.setWordWrap(True);layout.addWidget(self.label)
        self.timer=QTimer(self);self.timer.timeout.connect(lambda:self.step(1,playback=True))
        self.play.clicked.connect(self.toggle);self.back.clicked.connect(lambda:self.step(-1));self.next.clicked.connect(lambda:self.step(1))
        self.slider.valueChanged.connect(self.show_frame);self.scale.currentIndexChanged.connect(lambda:self.show_frame(self.slider.value()))
        self.speed.valueChanged.connect(lambda:self.timer.setInterval(round(1000/self.speed.value())))
        self.restore.clicked.connect(self.restore_endpoint)

    def set_run_data(self, data):
        self.workspace.image_pane.image_view._time_evolution_layout = False
        self.workspace.spatial_selector.setEnabled(True)
        self.orientation.setEnabled(False)
        self.pause();self.run_data=data;self.artifact=data.artifacts.get('td_trajectory')
        self.quantity.blockSignals(True);self.quantity.clear()
        for key in ('td_trajectory','td_trajectory_xz','td_trajectory_yz','td_trajectory_far_field','td_trajectory_material'):
            if key in data.artifacts:self.quantity.addItem(self._display_name(data.artifacts[key]),key)
        self.quantity.blockSignals(False)
        self.scale.model().item(1).setEnabled(True)
        enabled=self.artifact is not None
        for w in (self.play,self.back,self.next,self.slider,self.speed,self.loop,self.scale,self.save,self.restore):w.setEnabled(enabled)
        self.slider.blockSignals(True);self.slider.setRange(0,0 if not enabled else len(self.artifact.data)-1);self.slider.setValue(0);self.slider.blockSignals(False)
        self.label.setText('Time evolution unavailable: no accepted samples retained.' if not enabled else
            'Retained optical xy time evolution ready. Frame-uniform playback; labels use accepted τ. '+self.artifact.metadata['availability'])
        self.workspace.longitudinal_pane.show()
        self.workspace.image_pane.scale_controls.mode.model().item(0).setEnabled(True)

    @staticmethod
    def _display_name(artifact):
        return artifact.display_name.replace('Trajectory', 'Time evolution').replace('trajectory', 'time evolution')

    def select_quantity(self):
        if self.run_data is None:return
        self.artifact=self.run_data.artifacts.get(self.quantity.currentData())
        signed = self.artifact is not None and self.artifact.metadata.get('quantity') in ('E','psi')
        self.scale.model().item(1).setEnabled(not signed)
        if signed:self.scale.setCurrentIndex(0)
        self.show_frame(self.slider.value())

    def pause(self):self.timer.stop();self.play.setText('Play')

    def toggle(self):
        if self.timer.isActive():self.pause();return
        if self.artifact is None:return
        self.show_frame(self.slider.value());self.timer.start(round(1000/self.speed.value()));self.play.setText('Pause')

    def step(self, delta, playback=False):
        if self.artifact is None:return
        i=self.slider.value()+delta;n=len(self.artifact.data)
        if i<0 or i>=n:
            if self.loop.isChecked():i%=n
            else:self.pause();i=max(0,min(n-1,i))
        self.slider.setValue(i)
        self.show_frame(i)

    def show_frame(self, i):
        if self.artifact is None:return
        self.workspace.spatial_selector.setEnabled(False)
        self.workspace.linked_log.setEnabled(False)
        from lcprop.gui.views.display_scale import reset_display_norm
        reset_display_norm(self.workspace.image_pane.image_view)
        self.workspace.image_pane.image_view._time_evolution_layout = True
        a=self.artifact;m=a.metadata
        axes=tuple(m.get('axes',('x','y')))
        self.orientation.setEnabled('z' in axes)
        from lcprop.gui.movie_export import movie_display
        logarithmic = bool(self.scale.currentIndex()) and m.get('quantity') not in ('E','psi')
        values, limits = movie_display(a.data, logarithmic=logarithmic)
        suffix = '_log10' if logarithmic else ''
        field=FieldData(a.key+suffix,self._display_name(a)+(' (log10)' if logarithmic else ''),values[i],tuple(m.get('axes',('x','y'))),'intensity',
            quantity=m.get('quantity','optical_intensity'),value_unit=('log10 ' if logarithmic else '')+m['value_unit'],
            coordinates={k:np.asarray(v) for k,v in m['coordinates'].items()},
            units={k:('1' if k.startswith('s_') else 'um') for k in m['coordinates']},content_revision=i)
        scales=self.workspace.display_scales;key=scale_key(field)
        if scales.settings.get(key,('auto',None))[0]!='fixed':
            scales.configure(key,'locked', limits)
        fields=FieldCollection([(field.key,field)])
        self.workspace.image_pane.set_run_data(replace(self.run_data,fields=fields))
        view=self.workspace.image_pane.image_view
        if 'z' in axes:
            vertical=bool(self.orientation.currentIndex())
            swap=(axes.index('z') == 0) == vertical
            view.set_field(field, extent=self.workspace.image_pane._field_extent(field),
                           vmin=view.image.get_clim()[0], vmax=view.image.get_clim()[1], swap_axes=swap)
        # Do not juxtapose a final z volume
        # as though it belongs to the selected material time.
        self.workspace.longitudinal_pane.hide()
        self.workspace.image_pane.scale_controls.mode.model().item(0).setEnabled(False)
        self.label.setText(f'Frame {i+1}/{len(a.data)}; accepted τ={m["times"][i]:.12g}; '
            f'segment boundary τ={m["segment_start"]:.12g}. {self._display_name(a)}. Frame-uniform playback. '+m['availability'])

    def save_movie(self):
        if self.artifact is None:return
        self.pause()
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Movie', 'pr_time_evolution.mp4', 'MP4 movie (*.mp4)')
        if not filename:return
        if not filename.lower().endswith('.mp4'):filename += '.mp4'
        from lcprop.gui.movie_export import save_movie
        try:
            save_movie(self.artifact, filename, fps=self.speed.value(),
                       logarithmic=bool(self.scale.currentIndex()) and self.artifact.metadata.get('quantity') not in ('E','psi'))
        except (RuntimeError, ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Save Movie unavailable', str(exc))

    def restore_endpoint(self):
        self.workspace.image_pane.image_view._time_evolution_layout = False
        self.pause()
        self.orientation.setEnabled(False)
        if self.run_data is not None:
            self.workspace.image_pane.set_run_data(self.workspace._endpoint_view_data);self.workspace.longitudinal_pane.show()
            self.workspace.spatial_selector.setEnabled(True)
            self.workspace._select_spatial_product()
            self.workspace.image_pane.scale_controls.mode.model().item(0).setEnabled(True)
