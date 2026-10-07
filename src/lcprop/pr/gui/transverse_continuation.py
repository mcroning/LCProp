"""Bounded local TD segment editor; state-defining inputs are inherited/locked."""
from dataclasses import replace
from decimal import Decimal
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QSpinBox, QTableWidget, QTableWidgetItem, QPlainTextEdit, QPushButton, QDialogButtonBox)
from lcprop.pr.transverse.continuation import describe_continuation


class TransverseContinuationDialog(QDialog):
    beam_fields=('power_mW','x0_um','y0_um','theta_ext_rad','phi_rad','phase_rad')

    def __init__(self, checkpoint, parent=None, *, describe=describe_continuation, reduced=False):
        super().__init__(parent);self.checkpoint=checkpoint;self.request=None
        self.describe=describe
        self.setWindowTitle('Continue accepted reduced TD state' if reduced else 'Continue accepted transverse TD state');self.resize(900,650)
        layout=QVBoxLayout(self);r=checkpoint.request
        owner = 'Reduced TD' if reduced else 'Full-transverse TD'
        header=QLabel(
            f'{owner} checkpoint · {checkpoint.completed_steps} cumulative steps · τ = {checkpoint.time_normalized:.12g}\n'
            'Continue uses this checkpoint, not the current Run PR controls.\n'
            'Additional time is characteristic τ, not seconds. Set beam power to 0 to disable it.')
        header.setWordWrap(True)
        header.setToolTip(
            f'Normalization: {r.material.normalization_identity}\n'
            'Locked: grid/z layout, material scales, wavelength/index, model, integrator, precision, scattering.')
        layout.addWidget(header)
        self.beams=QTableWidget(len(r.beams.channels),len(self.beam_fields))
        self.beams.setHorizontalHeaderLabels(self.beam_fields);layout.addWidget(self.beams)
        for i,c in enumerate(r.beams.channels):
            for j,name in enumerate(self.beam_fields):self.beams.setItem(i,j,QTableWidgetItem(str(getattr(c,name))))
        form=QFormLayout();layout.addLayout(form)
        self.dark=QLineEdit(str(Decimal(str(r.material.dark_irradiance_W_cm2))*1000))
        self.uniform=QLineEdit(str(Decimal(str(r.material.uniform_irradiance_W_cm2))*1000))
        self.steps=QSpinBox();self.steps.setRange(0,1000000000);self.steps.setValue(r.solver.Nt)
        self.dt=QLineEdit(str(r.solver.dt_normalized))
        self.dt.setEnabled(not reduced)
        if reduced: self.dt.setToolTip('Reduced checkpoint cadence is locked; cumulative τ = accepted steps × Δτ.')
        form.addRow('Dark-equivalent irradiance (mW/cm²)',self.dark)
        form.addRow('Uniform background irradiance (mW/cm²)',self.uniform)
        form.addRow('Additional material steps',self.steps);form.addRow('Step Δτ',self.dt)
        self.inspection_status=QLabel();self.inspection_status.setWordWrap(True)
        layout.addWidget(self.inspection_status)
        self.details_button=QPushButton('Technical details (hashes / JSON)')
        self.details_button.setCheckable(True);layout.addWidget(self.details_button)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);layout.addWidget(self.preview)
        self.preview.hide();self.details_button.toggled.connect(self.preview.setVisible)
        inspect=QPushButton('Inspect continuation request');inspect.clicked.connect(self.inspect_request);layout.addWidget(inspect)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('Continue locally')
        buttons.accepted.connect(self.accept_request);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.inspect_request()

    def build_request(self):
        r=self.checkpoint.request
        channels=tuple(replace(c,**{name:float(self.beams.item(i,j).text())
            for j,name in enumerate(self.beam_fields)}) for i,c in enumerate(r.beams.channels))
        request=replace(r,beams=replace(r.beams,channels=channels),
            material=replace(r.material,dark_irradiance_W_cm2=float(Decimal(self.dark.text())/1000),
                uniform_irradiance_W_cm2=float(Decimal(self.uniform.text())/1000)),
            solver=replace(r.solver,Nt=self.steps.value(),dt_normalized=float(self.dt.text())))
        self.describe(request,self.checkpoint)
        return request

    def inspect_request(self):
        try:
            request=self.build_request()
            self.inspection_status.setText('Valid local continuation from the retained checkpoint; Run PR controls are not used.')
            self.preview.setPlainText(json.dumps(self.describe(request,self.checkpoint),indent=2,allow_nan=False))
        except Exception as exc:
            self.inspection_status.setText('Invalid continuation: '+str(exc))
            self.preview.setPlainText('Invalid continuation: '+str(exc))

    def accept_request(self):
        try:self.request=self.build_request()
        except Exception as exc:
            self.inspection_status.setText('Invalid continuation: '+str(exc))
            self.preview.setPlainText('Invalid continuation: '+str(exc));return
        self.accept()
