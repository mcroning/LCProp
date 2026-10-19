function runAdapterTests() {
  function assert(v,msg) {if(!v)throw Error(msg);}
  let clock=1000;
  class E {constructor(type,o={}){Object.assign(this,o);this.type=type;} preventDefault(){this.prevented=true;} stopImmediatePropagation(){this.stopped=true;}}
  class Target {
    constructor(){this.l={};this.style={touchAction:'auto'};this.capture=null;}
    addEventListener(t,f,o){(this.l[t]??=[]).push({f,cap:!!(o===true||o?.capture)});}
    removeEventListener(t,f){this.l[t]=(this.l[t]||[]).filter(x=>x.f!==f);}
    dispatchEvent(e){for(const x of [...(this.l[e.type]||[])].sort((a,b)=>Number(b.cap)-Number(a.cap))){x.f(e);if(e.stopped)break;}}
    getBoundingClientRect(){return {left:20,top:30,right:220,bottom:230};}
    setPointerCapture(id){this.capture=id;}
    hasPointerCapture(id){return this.capture===id;}
    releasePointerCapture(id){this.capture=null;this.dispatchEvent(new E('lostpointercapture',{pointerId:id}));}
  }
  const tests=[];
  function fixture(){
    const target=new Target(),win=new Target(),doc=new Target(),root={};
    root.querySelectorAll=()=>[new Target(),target];doc.querySelectorAll=()=>[root];doc.defaultView=win;
    win.PointerEvent=E;win.MouseEvent=E;win.performance={now:()=>clock};win.getComputedStyle=t=>({touchAction:t.style.touchAction});
    const received=[];for(const n of ['mousedown','mousemove','mouseup','mouseleave'])target.addEventListener(n,e=>received.push({type:n,x:e.clientX,y:e.clientY,button:e.button,buttons:e.buttons}));
    assert(installTouchAdapter('lcprop-touch-5b-abc',doc).ok,'install');
    const pointer=(type,opts={})=>target.dispatchEvent(new E(type,{pointerId:1,pointerType:'touch',isPrimary:true,button:0,buttons:1,clientX:70,clientY:90,...opts}));
    return {target,win,doc,root,received,pointer};
  }
  for(const type of ['touch','pen']){
    const f=fixture();f.pointer('pointerdown',{pointerType:type});f.pointer('pointermove',{pointerType:type,clientX:90});f.pointer('pointerup',{pointerType:type,buttons:0});
    assert(f.received.map(x=>x.type).join(',')==='mousedown,mousemove,mouseup',type+' ordered sequence');
    assert(f.received[1].x===90&&f.received[1].y===90&&f.received[0].button===0&&f.received[2].buttons===0,'coordinates/button');
    for(const n of ['mousedown','mousemove','mouseup'])f.target.dispatchEvent(new E(n,{clientX:90,clientY:90}));
    assert(f.received.length===3,'compat duplicates');tests.push(type+' sequence/coordinates/dedup');clock+=1000;
  }
  for(const reason of ['pointercancel','lostpointercapture','pointerleave','outside','blur','hidden','second','remove']){
    const f=fixture();f.pointer('pointerdown');f.pointer('pointermove');
    if(reason==='outside')f.pointer('pointermove',{clientX:300});
    else if(reason==='blur')f.win.dispatchEvent(new E('blur'));
    else if(reason==='hidden'){f.doc.hidden=true;f.doc.dispatchEvent(new E('visibilitychange'));}
    else if(reason==='second')f.pointer('pointerdown',{pointerId:2,isPrimary:false});
    else if(reason==='remove')f.root.__lcpropTouch5b.remove();
    else f.pointer(reason);
    f.pointer('pointerup',{buttons:0});
    assert(!f.received.some(x=>x.type==='mouseup')&&f.received.at(-1).type==='mouseleave',reason+' no commit');
    tests.push(reason+' cancellation');clock+=1000;
  }
  const f=fixture();f.pointer('pointerdown',{pointerType:'mouse'});
  for(const n of ['mousedown','mousemove','mouseup'])f.target.dispatchEvent(new E(n,{clientX:70,clientY:90,button:0,buttons:n==='mouseup'?0:1}));
  assert(f.received.length===3,'native mouse unaltered');
  assert(!installTouchAdapter('lcprop-touch-5b-abc',f.doc).ok,'double install');
  f.root.__lcpropTouch5b.remove();assert(f.target.style.touchAction==='auto','restore style');
  assert(!installTouchAdapter('lcprop-touch-5b-abc',{querySelectorAll:()=>[]}).ok,'framing stop');
  tests.push('native mouse','double-install refusal','style restore','framing stop');
  return {pass:true,tests,limitations:'Mock DOM event ordering only; not a real browser or Colab test'};
}
