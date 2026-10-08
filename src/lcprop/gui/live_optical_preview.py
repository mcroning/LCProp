"""Active-attempt optical display, independent of completed result ownership."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from lcprop.gui.views.image_view import ImageView
from lcprop.gui.views.display_scale import DisplayScales, DisplayScaleControls, volume_limits
from lcprop.products.data_model import FieldData


class LiveOpticalPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout=QVBoxLayout(self)
        self.label=QLabel('Waiting for a complete optical pass through accepted material.')
        self.label.setWordWrap(True);layout.addWidget(self.label)
        self.latest=QLabel();self.latest.setWordWrap(True);layout.addWidget(self.latest)
        self.view=ImageView();layout.addWidget(self.view,1)
        self.scales=DisplayScales()
        self.controls=DisplayScaleControls(self.scales);layout.addWidget(self.controls)
        self.scales.changed.connect(self._render)
        self.frame=None;self.initial_limits=None;self.key=('live_optical_pass',)
        self.run_id=None;self.terminal=None
        self.controls.mode.model().item(0).setEnabled(False)

    def begin(self, run_id, *, segment_number=None, preserve=False):
        self.run_id=run_id;self.terminal=None;self.pending_segment=segment_number
        self.awaiting_previous=bool(preserve and self.frame is not None)
        self.latest.clear()
        if self.awaiting_previous:
            self._render()
            return
        self.frame=None;self.initial_limits=None;self.checkpoint_identity=None
        self.view.clear_field();self.scales.settings.clear();self.scales.last.clear()
        self.controls.setEnabled(False)
        self.label.setToolTip('')
        self.label.setText('Waiting for a complete optical pass through accepted material.')

    def bind_checkpoint(self, identity):
        self.checkpoint_identity=identity

    @staticmethod
    def segment_label(number):
        return 'Segment (lineage unavailable)' if number is None else f'Segment {number}'

    def show_frame(self, frame):
        if frame.run_id!=self.run_id or (self.frame is not None
                and self.frame.run_id==frame.run_id and frame.pass_id<=self.frame.pass_id):return
        self.frame=frame
        self.awaiting_previous=False
        if self.initial_limits is None:
            low,high=volume_limits(frame.intensity_xz)
            # An all-zero dark launch has no useful automatic contrast range.
            self.initial_limits=(0.,1.) if high<=1e-15 else (min(0.,low),high)
            self.scales.configure(self.key,'locked',self.initial_limits)
        self._render()

    def set_latest_progress(self, text):
        self.latest.setText("Latest scalar acceptance: " + text)

    def finish(self, status):
        self.terminal=status;self._render()
        if self.frame is None:self.label.setText(f'{status}: no complete live optical preview available.')

    def _render(self,*args):
        if self.frame is None:return
        f=self.frame
        limits=self.scales.limits(self.key,self.initial_limits)
        field=FieldData('live_accepted_xz','Live accepted optical pass',f.intensity_xz,
            ('z','x'),'intensity',value_unit='1/um^2',
            units={'z':'um','x':'um'},coordinates={'z':f.z_um,'x':f.x_um})
        self.view.set_field(field,extent=(f.z_um[0],f.z_um[-1],f.x_um[0],f.x_um[-1]),
                            vmin=limits[0],vmax=limits[1])
        self.controls.show_scale(self.key,limits)
        segment=self.segment_label(f.segment_number)
        step=str(f.segment_step) if f.total_steps is None else f'{f.segment_step}/{f.total_steps}'
        description=f'Live optical intensity — {segment}, step {step}, cumulative τ={f.cumulative_time:.12g}'
        if getattr(self,'awaiting_previous',False):
            state=f'Previous segment — awaiting first optical observation from {self.segment_label(self.pending_segment)}'
            if self.terminal is not None:
                state=f'{self.terminal}: previous segment retained — no optical observation from {self.segment_label(self.pending_segment)}'
            description=state+'\n'+description
        elif self.terminal is not None:
            description=f'{self.terminal}: last complete preview; '+description
        self.label.setToolTip(f'Run: {f.run_id}\nSegment identifier: {f.segment_id}\n'
                             f'Pass: {f.pass_id}; cumulative step: {f.observed_step}')
        self.label.setText(description+f'; complete z pass {f.pass_id}; '
            f'preview x×z={len(f.x_um)}×{len(f.z_um)} of {f.original_shape[1]}×{f.original_shape[0]}, '
            f'y={f.y_um:g} µm; point sampling; numerical optical-only intensity. '
            'Fixed color range; use manual limits to rescale. Latest scalar progress may be newer.')
