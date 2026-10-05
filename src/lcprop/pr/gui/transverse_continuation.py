"""Bounded local TD segment editor; state-defining inputs are inherited/locked."""
from dataclasses import replace
from decimal import Decimal
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QSpinBox, QTableWidget, QTableWidgetItem, QPlainTextEdit, QPushButton, QDialogButtonBox)
from lcprop.pr.transverse.continuation import describe_continuation


class TransverseContinuationDialog(QDialog):
    beam_fields=('power_mW','x0_um','y0_um','theta_ext_rad','phi_rad','phase_rad')

    def __init__(self, checkpoint, parent=None):
        super().__init__(parent);self.checkpoint=checkpoint;self.request=None
        self.setWindowTitle('Continue accepted transverse TD state');self.resize(900,650)
        layout=QVBoxLayout(self);r=checkpoint.request
        layout.addWidget(QLabel(
            f'Source segment {len(checkpoint.record["lineage"])}; cumulative τ={checkpoint.time_normalized:.12g}\n'
            f'{checkpoint.record["normalization_identity"]}\n'
            'Locked: grid/z layout, material scales, wavelength/index, model, integrator, precision, scattering.\n'
            'Additional time is characteristic τ, not laboratory seconds. Set beam power to 0 to disable it.'))
        self.beams=QTableWidget(len(r.beams.channels),len(self.beam_fields))
        self.beams.setHorizontalHeaderLabels(self.beam_fields);layout.addWidget(self.beams)
        for i,c in enumerate(r.beams.channels):
            for j,name in enumerate(self.beam_fields):self.beams.setItem(i,j,QTableWidgetItem(str(getattr(c,name))))
        form=QFormLayout();layout.addLayout(form)
        self.dark=QLineEdit(str(Decimal(str(r.material.dark_irradiance_W_cm2))*1000))
        self.uniform=QLineEdit(str(Decimal(str(r.material.uniform_irradiance_W_cm2))*1000))
        self.steps=QSpinBox();self.steps.setRange(0,1000000000);self.steps.setValue(r.solver.Nt)
        self.dt=QLineEdit(str(r.solver.dt_normalized))
        form.addRow('Dark-equivalent irradiance (mW/cm²)',self.dark)
        form.addRow('Uniform background irradiance (mW/cm²)',self.uniform)
        form.addRow('Additional material steps',self.steps);form.addRow('Step Δτ',self.dt)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);layout.addWidget(self.preview)
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
        describe_continuation(request,self.checkpoint)
        return request

    def inspect_request(self):
        try:
            request=self.build_request()
            self.preview.setPlainText(json.dumps(describe_continuation(request,self.checkpoint),indent=2,allow_nan=False))
        except Exception as exc:self.preview.setPlainText('Invalid continuation: '+str(exc))

    def accept_request(self):
        try:self.request=self.build_request()
        except Exception as exc:
            self.preview.setPlainText('Invalid continuation: '+str(exc));return
        self.accept()
