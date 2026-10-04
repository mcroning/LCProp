"""Electrical experiment controls for fresh unified Static requests only."""
from PySide6.QtWidgets import QWidget, QFormLayout, QComboBox, QLabel
from lcprop.pr.gui.material_panel import _double_spin
from lcprop.pr.unified.specs import (
    PRElectricalClosureSpec,UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,A7_CURRENT,OPEN_TRANSVERSE,
)

class UnifiedClosurePanel(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.dimension=1
        self.form=QFormLayout(self);self.condition=QComboBox()
        self.x=_double_spin(-1e9,1e9,0.,decimals=12);self.y=_double_spin(-1e9,1e9,0.,decimals=12)
        self.info=QLabel();self.info.setWordWrap(True)
        self.form.addRow('Electrical condition',self.condition)
        self.form.addRow('Normalized x target',self.x);self.form.addRow('Normalized y target',self.y)
        self.form.addRow(self.info);self.condition.currentIndexChanged.connect(self.refresh)
        self.set_dimension(1)

    def set_dimension(self,d):
        current=self.condition.currentData();self.dimension=d
        choices=[('Unbiased / zero flux',UNBIASED),('Fixed mean internal field',FIXED_FIELD),
                 ('Prescribed normalized current',PRESCRIBED_CURRENT)]
        choices += [('A7 reservoir / prescribed current',A7_CURRENT)] if d==1 else [('Fixed x field / open transverse current',OPEN_TRANSVERSE)]
        self.condition.blockSignals(True);self.condition.clear()
        for label,value in choices:self.condition.addItem(label,value)
        self.condition.setCurrentIndex(max(0,self.condition.findData(current)));self.condition.blockSignals(False);self.refresh()

    def refresh(self,*_):
        kind=self.condition.currentData()
        self.x.setVisible(kind!=UNBIASED);self.form.labelForField(self.x).setVisible(kind!=UNBIASED)
        show_y=self.dimension==2 and kind in (FIXED_FIELD,PRESCRIBED_CURRENT)
        self.y.setVisible(show_y);self.form.labelForField(self.y).setVisible(show_y)
        label=('Reservoir E_app (normalized)' if kind==A7_CURRENT else
               'Prescribed J_x (normalized current)' if kind==PRESCRIBED_CURRENT else 'Prescribed b_x (mean internal field)')
        self.form.labelForField(self.x).setText(label)
        self.form.labelForField(self.y).setText('Prescribed J_y' if kind==PRESCRIBED_CURRENT else 'Prescribed b_y')
        self.info.setText('A7: J_ext = E_app × (dark + uniform); mean internal field b is solved.' if kind==A7_CURRENT else
            'Mean J_y = 0; b_y is solved, not forced to zero.' if kind==OPEN_TRANSVERSE else
            'Periodic bulk electrical closure; potential gauge is separate from mean field.')

    def closure(self,background):
        kind=self.condition.currentData()
        if kind==A7_CURRENT:return PRElectricalClosureSpec.a7(self.x.value(),background)
        target=(0.,)*self.dimension if kind==UNBIASED else ((self.x.value(),0.) if kind==OPEN_TRANSVERSE else
            (self.x.value(),) if self.dimension==1 else (self.x.value(),self.y.value()))
        return PRElectricalClosureSpec(kind,self.dimension,target)

    def set_closure(self,c):
        self.set_dimension(c.dimension);self.condition.setCurrentIndex(self.condition.findData(c.identity))
        self.x.setValue(c.reservoir_field if c.identity==A7_CURRENT else c.target[0])
        if c.dimension==2:self.y.setValue(c.target[1])
        self.refresh()
