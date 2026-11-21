"""Route-aware typed controls generated from the native experiment payload."""
from dataclasses import replace
from html import escape
import json
import math
import ipywidgets as W
from .batch_requests import ROUTES, preset, validate
from .batch_resources import estimate, MIB

ROUTE_CHOICES = {('reduced','static'):'pr_static', ('reduced','td'):'pr',
                 ('full','static'):'pr_xy_static', ('full','td'):'pr_xy_td'}

def disclosure(title, content):
    from .interactive_batch import disclosure as make
    return make(title, content)


def controls_for(value, path=(), *, changed=lambda: None, widgets=None):
    """Scalar form with JSON for variable arrays; codec owns types and legality."""
    widgets = {} if widgets is None else widgets
    if isinstance(value, dict):
        children=[]
        for key, item in value.items():
            if key == 'schema_version':
                continue
            child=controls_for(item, (*path,key), changed=changed, widgets=widgets)
            children.append(disclosure(key.replace('_',' '), child) if isinstance(item,dict) else child)
        return W.VBox(children)
    label=path[-1].replace('_',' ')
    label=label.replace(' W cm2',' (W/cm²)').replace(' um',' (µm)').replace(' rad',' (rad)').replace(' mW',' (mW)')
    kwargs=dict(description=label, style={'description_width':'initial'}, layout=W.Layout(width='98%'))
    if isinstance(value,bool):
        widget=W.Checkbox(value=value,**kwargs); getter=lambda: widget.value
    elif isinstance(value,int):
        widget=W.IntText(value=value,**kwargs); getter=lambda: widget.value
    elif isinstance(value,float):
        widget=W.Text(value=repr(value),continuous_update=False,**kwargs)
        def getter():
            v=float(widget.value)
            if not math.isfinite(v): raise ValueError(label+' must be finite')
            return v
    elif isinstance(value,str):
        options=None
        if path[-1]=='precision':options=['float64','float32']
        if path==('optical_boundary','mode'):options=['periodic','sponge','tukey']
        if path==('material_response','model'):options=['nonlinear','linearized','field_linear_local_intensity']
        if path[-1]=='integrator':
            from lcprop.pr.specs import PR_INTEGRATORS
            from lcprop.pr.transverse.specs import PR_TRANSVERSE_INTEGRATORS
            options=list(dict.fromkeys((*PR_INTEGRATORS,*PR_TRANSVERSE_INTEGRATORS)))
        if path[-1]=='backend':options=['numpy','auto','cupy']
        if options and value in options:
            widget=W.Dropdown(options=options,value=value,**kwargs)
        else:widget=W.Text(value=value,continuous_update=False,**kwargs)
        getter=lambda: widget.value
    else:
        widget=W.Textarea(value=json.dumps(value,indent=2),**kwargs)
        getter=lambda: json.loads(widget.value)
        widget.tooltip='JSON: null for absent optional input; arrays/objects retain native codec schema.'
    widget.tooltip=getattr(widget,"tooltip","") or ('Native request field: '+'.'.join(path)+'. Validated by the pinned engine.')
    widget.observe(lambda _: changed(), names='value')
    widgets[path]=(widget,getter)
    return widget


def payload_values(template, controls):
    result=json.loads(json.dumps(template))
    for path,(_,getter) in controls.items():
        target=result
        for key in path[:-1]:target=target[key]
        target[path[-1]]=getter()
    return result


def build_editor(app):
    s=app.session; route=s.route; generation=s.generation
    operation,codec,_=ROUTES[route]
    payload=codec.encode_request(s.draft)
    if not s.busy:
        s.runtime_policy.override=False
        s.device_policy.memory_override=False
        s.device_policy.allow_unqualified=False
    app.workflow_control=W.Dropdown(options=[k for k in ROUTES if k.startswith("pr")],value=route)
    pair=next((pair for pair,key in ROUTE_CHOICES.items() if key==route),('reduced','static'))
    if route=='pr_unified':pair=('full' if s.draft.closure.dimension==2 else 'reduced','static')
    transport=W.Dropdown(description='Transport',options=[('Reduced x-only','reduced'),('Full x–y','full')],value=pair[0])
    evolution=W.Dropdown(description='Evolution',options=[('Static','static'),('Time dependent','td')],value=pair[1])
    engine_workflow=W.Dropdown(description='Engine workflow',options=[
        ('Reduced Static — coupled legacy arithmetic','pr_static'),('Reduced TD','pr'),
        ('Full x–y Static — zero flux','pr_xy_static'),('Full x–y TD','pr_xy_td'),
        ('Unified Static — electrical closure / solver in Advanced','pr_unified'),
        ('Reduced Static — published optical-first Local-I/nonlinear','pr_published')],value=route,layout=W.Layout(width='98%'))
    device=W.Dropdown(description='Device',options=[('CPU','numpy'),('Auto','auto'),('GPU','cupy')],value=s.device_intent)
    precision=W.Dropdown(description='Precision',options=['float64','float32'],value=s.draft.backend.precision)
    app.precision_control=precision
    app.device_control=device
    app.run_button=W.Button(description='Run',button_style='primary',disabled=s.busy)
    app.run_reason=W.HTML()
    summary=W.HTML(); hardware=W.HTML(value='Detected GPU: not inspected (CPU does not probe CUDA).')
    detect=W.Button(description='Check GPU resources (tiny FFT)')
    risk=W.Checkbox(description='I accept demanding/uncertain memory estimates',value=False,indent=False)
    unqualified=W.Checkbox(description='Allow compatible, numerically unqualified GPU execution',value=False,indent=False)
    timeout=W.IntText(description='Cooperative time budget (s)',value=s.time_limit,style={'description_width':'initial'})
    preview_samples=W.IntText(description='Preview samples (all channels)',value=app.preview_sample_budget,style={'description_width':'initial'})
    def preview_changed(_):
        if preview_samples.value<1:
            app.notice.value='Preview samples must be positive';return
        app.preview_sample_budget=preview_samples.value
        app._refresh_preview();app.beam_plane.draw();app._update_selection()
    preview_samples.observe(preview_changed,names='value')
    products=W.Dropdown(description='Result products',options=['full','analysis','fast'],value=s.result_policy)
    if not operation.supports_result_policy:
        products.options=['full'];products.value='full';products.description='Native products';s.result_policy='full'
        products.tooltip='This engine operation retains its native result products; selectable projection is not supported.'
    products.observe(lambda change:setattr(s,'result_policy',change['new']),names='value')
    reset=W.Button(description='Restore selected preset'); release=W.Button(description='Release workflow results')
    app._form_invalid=''; app._reviewed_auto=None
    form_error = ''; time_error = ''
    def update_errors():
        app._form_invalid='; '.join(error for error in (form_error,time_error) if error)
    controls={}; app.request_controls=controls
    app._loading_form=False

    def memory_summary():
        charge=estimate(route,s.draft); available=s.memory_available()
        text=(f'Estimated working {charge.working/MIB:.0f} MiB; retained {charge.retained/MIB:.0f} MiB; '
              f'estimated already retained {s.resident_bytes()/MIB:.0f} MiB. Host available: '
              + ('unknown' if available is None else f'{available/MIB:.0f} MiB')
              + '. Conservative array/history estimate, not measured peak; FFT/kernel/allocator overhead uncertain. '
                'Available memory already excludes resident allocations. No fixed grid/result ceiling.')
        summary.value=escape(text)
        if available is not None:
            plane=s.draft.grid.Nx*s.draft.grid.Ny*(16 if s.draft.backend.precision=="float64" else 8)
            if plane>available:raise ValueError('One required optical plane exceeds available host memory')
        s.require_admission(charge)

    def refresh(*_, draft_changed=False):
        if draft_changed:
            invalidate()
        reason=''
        try:
            if app._form_invalid:raise ValueError(app._form_invalid)
            s.validate_intent(route,s.draft)
            memory_summary()
        except Exception as error: reason=str(error)
        if not reason and device.value == 'cupy' and not unqualified.value:
            reason='GPU execution requires explicit unqualified-execution opt-in in Advanced; compatibility and device admission remain pending'
        readiness = ('CPU inputs ready · no run started' if device.value == 'numpy' else
                     'Auto backend review pending · Run reviews the decision before dispatch' if device.value == 'auto' else
                     'Inputs valid · GPU compatibility and resource checks pending · no run started')
        app.run_reason.value=escape('Run unavailable: '+reason if reason else readiness)
        app.run_button.disabled=s.busy or bool(reason)
        return not reason

    def invalidate():
        app._reviewed_auto=None; s.device_decision=None
        risk.value=False
        unqualified.value=False
        s.device_policy.allow_unqualified=False
        s.runtime_policy.override=False
        s.device_policy.memory_override=False

    def changed():
        nonlocal form_error
        if app._loading_form or s.busy:return
        invalidate()
        try:
            new=payload_values(codec.encode_request(s.draft),controls)
            new['backend']['backend']='numpy' if device.value=='auto' else device.value
            new['backend']['precision']=precision.value
            request=codec.decode_request(new)
            # Codec and GUI validation can reject a scientifically invalid draft;
            # text remains visible and Run stays blocked until corrected.
            s.validate_intent(route,request)
            s.set_draft(route,request,generation)
            form_error=''
            if route=='pr_unified':
                app._loading_form=True
                transport.value='full' if request.closure.dimension==2 else 'reduced'
                app._loading_form=False
        except Exception as error:form_error=str(error)
        update_errors()
        refresh(); app._refresh_preview(); app.beam_plane.draw(); app._update_selection()

    # Beam values remain in the visual editor; a complete codec JSON editor below
    # supports wavelength/profile/more channels, image launch elements and scattering.
    regular={key:value for key,value in payload.items() if key not in {'schema_version','backend','beams'}}
    form=controls_for(regular,changed=changed,widgets=controls)
    app.grid_inputs={path[-1]:widget for path,(widget,_) in controls.items() if path[0]=='grid'}
    app.solver_inputs={path[-1]:widget for path,(widget,_) in controls.items() if path[0]=='solver'}
    full_json=W.Textarea(value=json.dumps(payload,indent=2),layout=W.Layout(width='100%',height='300px'))
    apply_json=W.Button(description='Apply complete request JSON')
    def apply_complete(_):
        try:
            request=codec.decode_request(json.loads(full_json.value))
            validate(route,request)
            s.set_draft(route,request,generation,reset_presentation=True)
            s.generation+=1; app._refresh_preview(); app._sync()
        except Exception as error:app.notice.value=escape(str(error))
    apply_json.on_click(apply_complete)
    def switch(_):
        if s.busy or app._loading_form:return
        if app._form_invalid:
            app.notice.value='Correct invalid text or restore the preset before switching. Your text has not been discarded.'
            return
        target=ROUTE_CHOICES[(transport.value,evolution.value)]
        if target!=s.route:
            s.switch(target);app._refresh_preview();app._sync()
    transport.observe(switch,names='value'); evolution.observe(switch,names='value')
    def select_engine(change):
        if s.busy:return
        if app._form_invalid:
            app.notice.value='Correct invalid text before changing workflow';return
        if change['new']!=s.route:
            s.switch(change['new']);app._refresh_preview();app._sync()
    engine_workflow.observe(select_engine,names='value')
    def choose_device(_):
        if s.busy:return
        invalidate(); s.device_intent=device.value
        s.draft=replace(s.draft,backend=replace(s.draft.backend,backend='numpy' if device.value=='auto' else device.value))
        refresh()
    device.observe(choose_device,names='value')
    precision.observe(lambda _:changed(),names='value')
    def risk_changed(_):
        s.runtime_policy.override=risk.value; s.device_policy.memory_override=risk.value
        app._reviewed_auto=None; refresh()
    risk.observe(risk_changed,names='value')
    def qualification_changed(_):
        s.device_policy.allow_unqualified=unqualified.value; app._reviewed_auto=None; refresh()
    unqualified.observe(qualification_changed,names='value')
    def time_changed(_):
        nonlocal time_error
        time_error='Time budget must be positive' if timeout.value<=0 else ''
        if not time_error:s.time_limit=timeout.value
        update_errors();refresh()
    timeout.observe(time_changed,names='value')
    def review_auto(digest,decision):
        current=(digest,decision.metadata())
        if app._reviewed_auto==current:return True
        app._reviewed_auto=current
        app.notice.value=escape('Auto proposes '+decision.label+'. '+(decision.fallback_reason or '')+
                               ' Review this decision, then click Run again to confirm. No dispatch yet.')
        return False
    def run(_):
        if refresh():
            app._guard(generation,lambda:s.start(generation,review_auto=review_auto))
    app.run_button.on_click(run)
    def restore(_):
        if s.busy:return
        s.set_draft(route,preset(route),generation,reset_presentation=True);s.generation+=1
        app._refresh_preview();app._sync()
    reset.on_click(restore)
    release.on_click(lambda _:app._guard(generation,lambda:(s.release_result(),app._sync())))
    def inspect(_):
        try:
            hw,gpu=s.device_policy.probe();s.device_policy.detected=hw
            with gpu.bind():
                sample=gpu.readiness((s.draft.grid.Nx,s.draft.grid.Ny))
            if sample.free_bytes < 0 or sample.free_bytes > sample.total_bytes:
                raise ValueError('Invalid GPU memory telemetry')
            # Discovery alone does not execute the compatibility FFT or qualify a request.
            hardware.value=escape(f'Detected {hw.model}, identity {hw.identity}; CUDA {hw.runtime}; '
                                  f'GPU free {sample.free_bytes/MIB:.0f} MiB / total {sample.total_bytes/MIB:.0f} MiB. '
                                  'Compatibility rechecked at dispatch. Numerical qualification separate.')
        except Exception as error:hardware.value=escape('GPU unavailable: '+str(error))
    detect.on_click(inspect)
    help=W.HTML('<b>CPU</b>: editable valid requests; no CUDA probe. <b>Auto</b>: review selected backend '
        'before dispatch and confirm again if it changes. <b>GPU</b>: explicit, never silently falls back. '
        'Compatible but unqualified requests require opt-in; frozen-case evidence is not general validation. '
        'CUDA dependencies must already be compatible; this CPU opening cell does not install CuPy.')
    from .raster import install as raster_controls
    raster_panel=disclosure("Image / intensity screen on active beam",raster_controls(app))
    advanced=disclosure('Advanced settings',W.VBox((precision,form,raster_panel,products,preview_samples,timeout,summary,risk,unqualified,detect,hardware,
        disclosure('Complete native request JSON — advanced inputs / launch elements / scattering',
                   W.VBox((W.HTML('Use the engine codec schema. Applying replaces only this draft; no dispatch.'),full_json,apply_json))))))
    from . import continuation
    cp_info=W.HTML()
    capture=W.Button(description='Use completed TD checkpoint')
    cp_folder=W.Text(description='Checkpoint folder',placeholder='Existing folder in Colab Files',layout=W.Layout(width='98%'))
    save_cp=W.Button(description='Save checkpoint to new folder');load_cp=W.Button(description='Load checkpoint folder')
    cp_json=W.Textarea(description='Continuation request',layout=W.Layout(width='98%',height='220px'))
    if s.continuation_checkpoint is not None and route in {'pr','pr_xy_td'}:
        try:cp_json.value=json.dumps(codec.encode_request(s.continuation_checkpoint.request),indent=2)
        except (TypeError,ValueError):pass
    cp_steps=W.IntText(description='Additional TD steps',value=3)
    cp_dt=W.Text(description='Normalized dt',value=str(getattr(s.draft.solver,'dt_normalized',.001)) if hasattr(s.draft,'solver') else '.001')
    continue_button=W.Button(description='Continue accepted state',disabled=route not in {'pr','pr_xy_td'})
    def cp_action(action):
        if s.busy:return
        try:
            action();cp_info.value='Accepted checkpoint retained separately; Continue uses its inputs, not current Run controls.'
        except Exception as error:cp_info.value=escape(str(error))
    def capture_cp():
        s.continuation_checkpoint=continuation.checkpoint_for(s.completed)
        cp_json.value=json.dumps(ROUTES[s.completed.route][1].encode_request(s.continuation_checkpoint.request),indent=2)
    def perform_load_cp():
        s.continuation_checkpoint=continuation.load_checkpoint(route,cp_folder.value,admission=lambda size:s.runtime_policy.check(size,s.memory_available(),label="checkpoint loading"))
        cp_json.value=json.dumps(codec.encode_request(s.continuation_checkpoint.request),indent=2)
    def save_cp_action():
        continuation.save_checkpoint(route,s.continuation_checkpoint,cp_folder.value)
    def continue_cp():
        cp=getattr(s,'continuation_checkpoint',None)
        if cp is None:raise ValueError('Select or load an accepted TD checkpoint first')
        base=codec.decode_request(json.loads(cp_json.value)) if cp_json.value.strip() else cp.request
        request=replace(base,backend=replace(base.backend,backend='numpy'),
                        solver=replace(base.solver,Nt=cp_steps.value,dt_normalized=float(cp_dt.value)))
        continuation.describe(route,request,cp)
        s.start_continuation(route,request,cp)
    capture.on_click(lambda _:cp_action(capture_cp));load_cp.on_click(lambda _:cp_action(perform_load_cp))
    save_cp.on_click(lambda _:cp_action(save_cp_action));continue_button.on_click(lambda _:cp_action(continue_cp))
    cp_panel=disclosure('TD accepted-state continuation (CPU)',W.VBox((
        W.HTML('Uses an authoritative saved state and native compatibility checks. Static has no Continue. '
               'A legacy saved result may not contain the required checkpoint. Source state-defining inputs stay locked.'),
        capture,cp_folder,load_cp,save_cp,cp_steps,cp_dt,
        disclosure("Optional illumination changes — native request JSON; engine locks state-defining inputs",cp_json),continue_button,cp_info)))
    app.continuation_panel=cp_panel
    app.editor.children=(engine_workflow,W.HBox((transport,evolution),layout=W.Layout(flex_flow='row wrap')),device,
        W.HBox((app.run_button,reset,release),layout=W.Layout(flex_flow='row wrap')),app.run_reason,help,advanced,cp_panel)
    for widget in (cp_json,preview_samples,precision,products,capture,cp_folder,save_cp,load_cp,cp_steps,cp_dt,engine_workflow,transport,evolution,device,reset,release,risk,unqualified,timeout,detect,apply_json,*[w for w,_ in controls.values()]):
        widget.disabled=s.busy
    app.advanced=advanced
    app.transport_control=transport;app.evolution_control=evolution;app.engine_workflow=engine_workflow
    app.memory_risk=risk;app.unqualified_opt_in=unqualified
    app.resource_summary=summary;app.hardware_status=hardware
    if route in {"pr_unified","pr_published"}:
        transport.disabled=evolution.disabled=True
        transport.tooltip="Model is shown here; edit Unified closure/solver together in Advanced, or choose a classic engine workflow."
    products.disabled=s.busy or not operation.supports_result_policy
    app._update_editor_summary=refresh
    refresh()
