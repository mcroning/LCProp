"""First-show geometry in the real application; no user resize workaround."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import QObject,QEvent
from PySide6.QtWidgets import QApplication,QStyle,QStyleOptionSpinBox,QStyleFactory
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.app import configure_application_font


class ResizeCounter(QObject):
    def __init__(self):super().__init__();self.count=0
    def eventFilter(self,watched,event):
        if event.type()==QEvent.Resize:self.count+=1
        return False


@pytest.mark.parametrize('style',[s for s in ('macOS','Fusion','Windows') if s in QStyleFactory.keys()])
def test_default_startup_tabs_and_resize(style):
    app=QApplication.instance() or QApplication([]);configure_application_font(app)
    previous=app.style().objectName();app.setStyle(style)
    w=PRMainWindow();w.show();app.processEvents();initial=w.size()
    counter=ResizeCounter();w.installEventFilter(counter)
    e=w.beam_panel.input_screen_editor
    controls=[e.width_um,e.height_um,e.center_x_um,e.center_y_um]
    signature=lambda:[(c.value(),c.minimum(),c.maximum(),c.decimals(),c.singleStep()) for c in controls]
    before=signature()
    def check():
        for c in controls:
            assert c.isVisible()
            option=QStyleOptionSpinBox();c.initStyleOption(option)
            edit=c.style().subControlRect(QStyle.CC_SpinBox,option,QStyle.SC_SpinBoxEditField,c)
            assert c.height()>=c.minimumSizeHint().height()
            assert edit.height()>=c.lineEdit().sizeHint().height()
            assert c.lineEdit().height()>=c.lineEdit().sizeHint().height()
            assert c.lineEdit().contentsRect().height()>=c.fontMetrics().height()
            target=QStyleOptionSpinBox(option);target.rect.setSize(c.sizeHint())
            for part in (QStyle.SC_SpinBoxUp,QStyle.SC_SpinBoxDown):
                arrow=c.style().subControlRect(QStyle.CC_SpinBox,option,part,c)
                expected=c.style().subControlRect(QStyle.CC_SpinBox,target,part,c)
                assert arrow.height()>=expected.height()>0
                assert c.rect().contains(arrow)
        assert signature()==before
    try:
        w.tabs.setCurrentWidget(w.beam_panel)
        w.beam_panel.beam_tabs.setCurrentWidget(w.beam_panel.input_screen_scroll)
        e.screen_type.setCurrentIndex(1)
        check()  # Immediate activation: no event processing or resize workaround.
        app.processEvents();check()
        assert w.size()==initial and counter.count==0
        w.tabs.setCurrentWidget(w.grid_panel);app.processEvents()
        w.tabs.setCurrentWidget(w.beam_panel);app.processEvents();check()
        assert counter.count==0
        w.resize(initial.width()+100,initial.height()+100);app.processEvents();check()
        w.resize(initial);app.processEvents();check()
    finally:
        w.close();app.setStyle(previous)
