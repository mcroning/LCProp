"""No solver: test existing installed controller's event path with synthetic mouse events."""
import json,dataclasses,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('module://ipympl.backend_nbagg')
from matplotlib.backend_bases import MouseEvent,LocationEvent
import matplotlib.pyplot as plt
from lcprop.notebook import InteractiveLC
from lcprop.notebook import interactive_lc as ui
ui.execute=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Science forbidden'))
results=[]
with plt.ioff():app=InteractiveLC()
app.run_button.disabled=True;app.live_preview.value=False
ed=app._position_editor;canvas=ed.canvas;canvas.draw()
def request():return dataclasses.asdict(app.request())
def mouse(name,xy):
 x,y=ed.ax.transData.transform(xy)
 MouseEvent(name,canvas,x,y,button=1)._process()
def gesture(start,end,cancel=False):
 mouse('button_press_event',start);mouse('motion_notify_event',end)
 if cancel:LocationEvent('figure_leave_event',canvas,0,0)._process()
 else:mouse('button_release_event',end)
for second in [False,True]:
 app.second_enabled.value=second;app.active_beam.value=int(second);canvas.draw()
 for action in ['Move','Tilt']:
  app.action.value=action;canvas.draw()
  marker=ed.markers[int(second)] if action=='Move' else ed.tips[int(second)]
  start=(float(marker.get_xdata()[0]),float(marker.get_ydata()[0]));end=(start[0]+2,start[1]+1)
  before=request();gesture(start,end,cancel=True);assert request()==before,(second,action,'cancel')
  canvas.draw();gesture(start,end);assert request()!=before,(second,action,'release')
  results.append({'beam':int(second)+1,'action':action,'cancel_no_commit':True,'release_updates_controls':True})
# A busy controller rejects a new gesture; no solver is invoked.
app.action.value='Move';canvas.draw();m=ed.markers[1];start=(float(m.get_xdata()[0]),float(m.get_ydata()[0]))
before=request();app.busy=True;gesture(start,(start[0]+1,start[1]+1));assert request()==before
app.busy=False;assert app.completed is None;app.close()
print(json.dumps({'pass':True,'results':results,'busy_guard':True,'completed_remains_none':True,'scientific_execution':False,'limitations':'Synthetic Python mouse events only; no browser adapter integration or real completed snapshot tested'},indent=2))
