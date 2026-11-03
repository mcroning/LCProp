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
        for spin in (self.x, self.y):
            spin.setMinimumWidth(spin.fontMetrics().horizontalAdvance('-123456789.123456789012') + 48)
        self.info=QLabel();self.info.setWordWrap(True)
        self.form.addRow('Electrical condition',self.condition)
        self.form.addRow('Normalized x target',self.x);self.form.addRow('Normalized y target',self.y)
        self.form.addRow(self.info);self.condition.currentIndexChanged.connect(self.refresh)
        self.set_dimension(1)

    def set_dimension(self,d):
        current=self.condition.currentData();self.dimension=d
        choices=[('Unbiased / zero flux',UNBIASED),('Fixed mean internal field',FIXED_FIELD),
                 ('Prescribed normalized current',PRESCRIBED_CURRENT)]
        choices += [('Applied bias / prescribed current (A7)',A7_CURRENT)] if d==1 else [('Fixed x field / open transverse current',OPEN_TRANSVERSE)]
        self.condition.blockSignals(True);self.condition.clear()
        for label,value in choices:self.condition.addItem(label,value)
        self.condition.setCurrentIndex(max(0,self.condition.findData(current)));self.condition.blockSignals(False);self.refresh()

    def refresh(self,*_):
        kind=self.condition.currentData()
        self.x.setVisible(kind!=UNBIASED);self.form.labelForField(self.x).setVisible(kind!=UNBIASED)
        show_y=self.dimension==2 and kind in (FIXED_FIELD,PRESCRIBED_CURRENT)
        self.y.setVisible(show_y);self.form.labelForField(self.y).setVisible(show_y)
        label=('Applied bias E_app (normalized)' if kind==A7_CURRENT else
               'Prescribed J_x (normalized current)' if kind==PRESCRIBED_CURRENT else 'Mean internal field b_x (dimensionless)')
        self.form.labelForField(self.x).setText(label)
        self.form.labelForField(self.y).setText('Prescribed J_y (normalized current)' if kind==PRESCRIBED_CURRENT else 'Mean internal field b_y (dimensionless)')
        field_help = ('Mean internal field: E_mean = E_s b, with E_s = k_B T k_D/e.\n'
                      'Mathematically this is the harmonic field.\n'
                      'k_D is the material screening wavenumber, not the optical wavenumber. '
                      'The full local field also contains −∇ψ in normalized coordinates.')
        target_help = ('Prescribes the spatial mean of dimensionless transport current.\n'
                       'J = n I E − ∇(n I), with E = b − ∇ψ and mean(n) = 1.\n'
                       'I is total transport intensity (including dark/background),\n'
                       'using the selected illumination normalization; gradients use material coordinates.\n'
                       'The mean internal field adjusts. This is not an SI current input.'
                       if kind==PRESCRIBED_CURRENT else
                       'A7 reservoir E_app sets J_ext via the normalized background; solved b need not equal E_app.'
                       if kind==A7_CURRENT else field_help)
        self.x.setToolTip(target_help);self.y.setToolTip(target_help)
        self.form.labelForField(self.x).setToolTip(target_help)
        self.form.labelForField(self.y).setToolTip(target_help)
        self.info.setText('Applied bias sets the current: J_ext = E_app × (dark + uniform). The mean internal field adjusts to carry it; it is not fixed to the applied parameter.' if kind==A7_CURRENT else
            'No net transverse current (mean J_y = 0). The mean y field adjusts to enforce this; it need not be zero.' if kind==OPEN_TRANSVERSE else
            'Periodic material response. The potential reference sets its zero level, independently of the mean internal field.')

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
