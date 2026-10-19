/* Disposable diagnostic. No ipympl model/view-manager access, package writes or Run calls. */
function installTouchAdapter(scope, doc = document) {
  if (!/^lcprop-touch-5b-[a-f0-9]+$/.test(scope)) throw Error('Invalid diagnostic scope');
  const roots = doc.querySelectorAll('.' + scope);
  if (roots.length !== 1) return {ok:false, reason:'Scoped widget not in this output document. STOP; no frame traversal.'};
  const root = roots[0];
  const canvases = [...root.querySelectorAll('canvas')];
  if (canvases.length !== 2) return {ok:false, reason:'Expected two ipympl canvases. STOP; frontend structure unqualified.'};
  // Version-specific DOM contract: ipympl 0.12 top interaction canvas follows image canvas.
  const target = canvases[1], win = doc.defaultView;
  if (!win.PointerEvent) return {ok:false, reason:'Pointer Events unavailable'};
  if (root.__lcpropTouch5b) return {ok:false, reason:'Adapter already installed; remove it before reinstalling'};
  const trace = [], listeners = [], synthetic = new WeakSet();
  let active = null, blockedUntil = 0, nextStart = 0, dropped = 0;
  const now = () => win.performance.now();
  function log(kind, e, extra = {}) {
    if (trace.length >= 2000) { dropped++; return; }
    trace.push({seq:trace.length, ms:now(), kind, pointerId:e?.pointerId,
      pointerType:e?.pointerType, clientX:e?.clientX, clientY:e?.clientY, ...extra});
  }
  function inside(e) {
    const r = target.getBoundingClientRect();
    return Number.isFinite(e.clientX) && Number.isFinite(e.clientY) &&
      e.clientX >= r.left && e.clientX < r.right && e.clientY >= r.top && e.clientY < r.bottom;
  }
  function send(type, e, buttons) {
    // Preserve CSS client coordinates: ipympl applies its own bounding-box/pixel ratio conversion.
    const m = new win.MouseEvent(type, {bubbles:true, cancelable:true, view:win,
      clientX:e.clientX, clientY:e.clientY, button:0, buttons,
      ctrlKey:false, shiftKey:false, altKey:false, metaKey:false});
    synthetic.add(m); log('emit:' + type, e, {buttons}); target.dispatchEvent(m);
  }
  function dropCapture(id) {
    if (target.hasPointerCapture(id)) target.releasePointerCapture(id);
  }
  function cancel(reason) {
    if (!active) return;
    const old = active; active = null; blockedUntil = now()+800; nextStart = now()+100;
    log('cancel', old.last, {reason});
    // Existing BeamPlane figure_leave handler discards provisional geometry; never synthesize mouseup.
    send('mouseleave', old.last, 0); dropCapture(old.id);
  }
  function stop(e) { e.preventDefault(); e.stopImmediatePropagation(); }
  function down(e) {
    log('pointerdown',e);
    if (e.pointerType === 'mouse') {
      cancel('mouse-interruption'); blockedUntil = 0; return; // Leave genuine desktop path untouched.
    }
    if (!['touch','pen'].includes(e.pointerType)) return;
    stop(e);
    if (active) { cancel('second-contact'); return; }
    if (!e.isPrimary || e.button !== 0 || now() < nextStart || !inside(e)) return;
    try { target.setPointerCapture(e.pointerId); }
    catch (_) { log('capture-failed', e); return; }
    active = {id:e.pointerId, last:e}; blockedUntil=now()+800;
    send('mousedown', e, 1);
  }
  function move(e) {
    if (!active || active.id !== e.pointerId) return;
    stop(e); active.last=e;
    if (!inside(e) || !(e.buttons & 1)) { cancel('outside-or-contact-ended'); return; }
    send('mousemove', e, 1);
  }
  function up(e) {
    if (!active || active.id !== e.pointerId) return;
    stop(e); active.last=e;
    if (!inside(e)) { cancel('release-outside'); return; }
    const id=active.id; active=null; blockedUntil=now()+800; nextStart=now()+100;
    send('mouseup', e, 0); dropCapture(id);
  }
  function interrupted(e) { if (active && active.id===e.pointerId) cancel(e.type); }
  function compat(e) {
    if (synthetic.has(e)) return;
    if (active || now()<blockedUntil || e.sourceCapabilities?.firesTouchEvents) {
      stop(e); log('suppressed:'+e.type,e);
    }
  }
  function on(obj,type,fn,options) { obj.addEventListener(type,fn,options); listeners.push([obj,type,fn,options]); }
  const oldTouch=target.style.touchAction;
  target.style.touchAction='none';
  on(target,'pointerdown',down,{capture:true,passive:false});
  on(target,'pointermove',move,{capture:true,passive:false});
  on(target,'pointerup',up,{capture:true,passive:false});
  on(target,'pointercancel',interrupted,true);
  on(target,'lostpointercapture',interrupted,true);
  on(target,'pointerleave',e=>{if(active && e.pointerId===active.id) cancel('pointerleave');},true);
  // Do not suppress real figure-leave: it remains an independent cancellation route.
  for(const type of ['mousedown','mousemove','mouseup','click','dblclick']) on(target,type,compat,true);
  on(doc,'pointerdown',e=>{if(active && e.pointerId!==active.id) cancel('other-contact');},true);
  on(win,'blur',()=>cancel('window-blur'));
  on(doc,'visibilitychange',()=>{if(doc.hidden) cancel('document-hidden');});
  const api={trace, snapshot:()=>({trace:[...trace],dropped,active:!!active}),
    cancel:()=>cancel('manual'),
    remove:()=>{cancel('remove'); for(const x of listeners)x[0].removeEventListener(x[1],x[2],x[3]);
      target.style.touchAction=oldTouch; delete root.__lcpropTouch5b;}};
  root.__lcpropTouch5b=api;
  log('installed',null,{canvasCount:canvases.length,touchAction:win.getComputedStyle(target).touchAction});
  return {ok:true,scope,canvasCount:canvases.length,touchAction:win.getComputedStyle(target).touchAction,
    note:'Diagnostic DOM contract only. Browser qualification required.'};
}
