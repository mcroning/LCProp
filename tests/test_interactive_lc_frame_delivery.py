"""Real ipympl PNG dependency tests; synthetic delivery faults, no propagation."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def compose(frames, order=None, initial=None):
    view = Image.new('RGBA', frames[0]['image'].size, 'white') if initial is None else initial.copy()
    for i in (range(len(frames)) if order is None else order):
        frame = frames[i]
        view = frame['image'].copy() if frame['mode'] == 'full' else Image.alpha_composite(view, frame['image'])
    return view


def mismatch(view, reference):
    return int(np.any(np.asarray(view) != reference, axis=2).sum())


class Recorder:
    def __init__(self, monkeypatch):
        pytest.importorskip('ipympl')
        from ipympl.backend_nbagg import Canvas
        from lcprop.notebook import interactive_lc as ui
        self.ui = ui
        self.frames = []
        self.events = []
        self.phase = 'constructor'
        self.mode = 'full'
        send, send_json = Canvas.send, Canvas.send_json
        def capture_json(canvas, content):
            self.events.append({'phase': self.phase, 'direction': 'out', 'type': content['type']})
            if content['type'] == 'image_mode': self.mode = content['mode']
            return send_json(canvas, content)
        def capture(canvas, content, buffers=None):
            if buffers:
                raw = memoryview(buffers[0]).tobytes()
                image = Image.open(io.BytesIO(raw)).convert('RGBA'); image.load()
                self.frames.append(dict(mode=self.mode, png=raw, image=image,
                    rgba=np.asarray(canvas.buffer_rgba()).copy(), phase=self.phase))
                self.events.append({'phase': self.phase, 'direction': 'out', 'type': 'png',
                                    'index':len(self.frames)-1, 'mode':self.mode})
            return send(canvas, content, buffers=buffers)
        monkeypatch.setattr(Canvas,'send_json',capture_json)
        monkeypatch.setattr(Canvas,'send',capture)
        monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('Propagation forbidden'))
        self.app = ui.InteractiveLC()
        self.editor = self.app._position_editor
        assert self.editor is not None, self.app.editor_status.value
        self.canvas = self.editor.canvas
        assert self.frames == []
        self.phase = 'initialization'
        for kind in ('refresh','send_image_mode','initialized','draw'): self.dispatch(kind)
        assert self.frames[0]['mode'] == 'full'
        self.initial = self.frames[-1]['rgba'].copy()
        self.states = {}

    def dispatch(self, kind):
        self.events.append({'phase':self.phase,'direction':'in','type':kind})
        self.canvas._handle_custom_msg({'type':kind},[])

    def step(self,label,callback):
        self.phase=label
        before=len(self.frames)
        callback()
        assert len(self.frames)==before  # edits request a draw, not image transport
        self.dispatch('draw')  # frontend response to draw_idle
        assert len(self.frames)>before
        self.dispatch('draw')  # an additional unchanged pending draw reply
        request=self.app.request()
        assert len(self.editor.markers)==len(request.beams.channels)
        canonical=self.ui.launch_preview(request)[2]
        mesh=np.asarray(self.editor.mesh.get_array()).reshape(canonical.T.shape).T.copy()
        np.testing.assert_array_equal(canonical,mesh)
        assert self.editor.mesh.get_clim()==(float(canonical.min()),float(canonical.max()))
        state=dict(request=self.ui.encode_lc_static_request(request),canonical=canonical,
                   mesh=mesh,clim=self.editor.mesh.get_clim(),index=len(self.frames)-1,
                   rgba=self.frames[-1]['rgba'].copy())
        self.states[label]=state
        return state

    def save(self,directory):
        directory.mkdir()
        for name,s in self.states.items():
            (directory/(name+'-request.json')).write_text(json.dumps(s['request'],sort_keys=True,indent=2)+'\n')
            np.savez(directory/(name+'-arrays.npz'),canonical=s['canonical'],mesh=s['mesh'],rgba=s['rgba'],clim=s['clim'])
        rows=[]
        for i,f in enumerate(self.frames):
            (directory/f'frame-{i:02}.png').write_bytes(f['png'])
            rows.append({'index':i,'phase':f['phase'],'mode':f['mode'],'bytes':len(f['png']),
                         'sha256':hashlib.sha256(f['png']).hexdigest()})
        (directory/'events.json').write_text(json.dumps(self.events,indent=2)+'\n')
        (directory/'frames.json').write_text(json.dumps(rows,indent=2)+'\n')


@pytest.fixture
def recorded(monkeypatch):
    r=Recorder(monkeypatch)
    yield r
    r.app.close()


def test_two_beam_callbacks_delivery_and_recovery(recorded,tmp_path):
    r=recorded;a=r.app
    first=r.step('beam2-on',lambda:setattr(a.second_enabled,'value',True))
    assert len(first['request']['beams']['channels'])==2
    r.step('radius-12',lambda:setattr(a.beam2_controls['w1_um'],'value',12.))
    r.step('beam2-off',lambda:setattr(a.second_enabled,'value',False))
    r.step('beam2-on-again',lambda:setattr(a.second_enabled,'value',True))
    r.step('radius-7',lambda:setattr(a.beam2_controls['w1_um'],'value',7.))
    r.step('beam2-off-again',lambda:setattr(a.second_enabled,'value',False))
    final=r.step('final-two-beam',lambda:setattr(a.second_enabled,'value',True))
    reference=final['rgba']; n=len(r.frames)
    # Every ordered prefix matches the exact corresponding rendered image.
    for i in range(n):
        assert mismatch(compose(r.frames,list(range(i+1))),r.frames[i]['rgba'])==0
    ordered=compose(r.frames)
    # Drop the last changed image but deliver its subsequent unchanged draw.
    lost=n-2
    missing=compose(r.frames,[i for i in range(n) if i!=lost])
    lost_count=mismatch(missing,reference); assert lost_count>0
    # Deliver a stale pre-final transition after the final frame.
    order=list(range(n)); stale=next(i for i,f in enumerate(r.frames) if f['phase']=='beam2-off-again')
    order.remove(stale);order.append(stale)
    reordered=compose(r.frames,order)
    reordered_count=mismatch(reordered,reference);assert reordered_count>0
    # A delayed frame is harmless when order is retained (above); an unpainted
    # dependency is not repaired by an unchanged differential draw.
    r.phase='unchanged-redraw';r.dispatch('draw')
    still_missing=compose(r.frames,[len(r.frames)-1],initial=missing)
    assert mismatch(still_missing,reference)==lost_count
    # Bounded supported recovery; no private state modification.
    r.phase='recovery';r.dispatch('refresh');r.canvas.draw()
    assert r.frames[-1]['mode']=='full'
    repaired=compose(r.frames,[len(r.frames)-1],initial=still_missing)
    assert mismatch(repaired,reference)==0
    # Recovery is not immune to delivery faults: discarding it leaves corruption.
    r.phase='post-recovery-idle';r.dispatch('draw')
    unrepaired=compose(r.frames,[len(r.frames)-1],initial=still_missing)
    assert mismatch(unrepaired,reference)==lost_count
    r.save(tmp_path/'capture')
    for name,im in [('ordered',ordered),('dropped',missing),('reordered',reordered),('repaired',repaired)]:im.save(tmp_path/(name+'.png'))
    (tmp_path/'metrics.json').write_text(json.dumps({'frames_before_recovery':n,
        'lost_frame':lost,'dropped_differing_pixels':lost_count,'stale_frame':stale,
        'reordered_differing_pixels':reordered_count,'repaired_differing_pixels':0,
        'request_sha256':hashlib.sha256(json.dumps(final['request'],sort_keys=True).encode()).hexdigest()},indent=2)+'\n')


@pytest.mark.parametrize('component',['markers','limits','image'])
def test_component_draws_are_correct_in_order(recorded,tmp_path,component):
    r=recorded
    r.step('two-beam',lambda:setattr(r.app.second_enabled,'value',True))
    original=r.editor.committed_intensity.copy()
    expected=original
    r.phase='isolate-'+component
    if component=='markers':
        r.app.active_beam.value=1  # marker/arrow highlighting only; no launch update
        r.editor.synchronize(immediate=True)
        r.dispatch('draw')
    elif component=='limits':
        low,high=r.editor.mesh.get_clim();r.editor.mesh.set_clim(low,high*.8);r.canvas.draw()
    else:
        # Canonical changed-radius image, deliberately isolating the mesh update
        # from normalization and marker changes in this rendering-only test.
        from dataclasses import replace
        request=r.app.request()
        beams=request.beams.channels
        request=replace(request,beams=replace(request.beams,channels=(beams[0],replace(beams[1],w1_um=14.))))
        expected=r.ui.launch_preview(request)[2]
        r.editor.mesh.set_array(expected.T);r.canvas.draw()
    assert mismatch(compose(r.frames),r.frames[-1]['rgba'])==0
    np.testing.assert_array_equal(r.editor.mesh.get_array(),expected.T)
    r.save(tmp_path/'capture')
