"""Bridge to existing accepted-state continuation, with checkpoint-owned inputs."""
from dataclasses import replace
from functools import partial
from pathlib import Path
from lcprop.runners.base import WorkflowOperation
from .batch_requests import ROUTES


def checkpoint_for(record):
    if record is None:raise ValueError('Open an accepted TD result first')
    if record.route=='pr':
        from lcprop.pr.checkpoint import validate_pr_checkpoint
        cp=getattr(record.result,'checkpoint',None)
        if cp is None:raise ValueError('Saved result does not contain an authoritative reduced TD checkpoint')
        validate_pr_checkpoint(cp);return cp
    if record.route=='pr_xy_td':
        from lcprop.pr.transverse.continuation import checkpoint_from_result
        return checkpoint_from_result(record.request,record.result)
    raise ValueError('Static results have no time-continuation operation')


def describe(route,request,checkpoint):
    if route=='pr_xy_td':
        from lcprop.pr.transverse.continuation import describe_continuation
    elif route=='pr':
        if request.material.normalization_identity == "pr_channel_peak_reference_v1":
            from lcprop.pr.checkpoint import validate_pr_continuation
            validate_pr_continuation(request,checkpoint)
            return {"source":"accepted reduced legacy checkpoint"}
        from lcprop.pr.reduced_continuation import describe_continuation
    else:raise ValueError('Continue requires TD')
    return describe_continuation(request,checkpoint)


def operation_for(route,checkpoint):
    op=ROUTES[route][0]
    if route=='pr_xy_td':
        from lcprop.pr.transverse.continuation import continue_transverse_td
        execute=partial(continue_transverse_td,checkpoint=checkpoint)
    elif route=='pr':
        if checkpoint.request.material.normalization_identity == "pr_channel_peak_reference_v1":
            from lcprop.pr.workflow import continue_pr_timedependent as continuation
        else:
            from lcprop.pr.reduced_continuation import continue_physical_pr as continuation
        execute=partial(continuation,checkpoint=checkpoint)
    else:raise ValueError('Continue requires TD')
    return WorkflowOperation(op.material_id,op.workflow_id,execute,op.to_run_data)


def save_checkpoint(route,checkpoint,path):
    path=Path(path)
    if path.exists():raise FileExistsError('Choose a new checkpoint folder')
    if route=='pr_xy_td':
        from lcprop.pr.transverse.continuation import save_checkpoint as save
    elif route=='pr':
        from lcprop.pr.persistence import save_pr_checkpoint as save
    else:raise ValueError('Continue requires TD')
    save(checkpoint,path)


def load_checkpoint(route,path, *, admission=lambda size: None):
    import zipfile
    root=Path(path)
    size=sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    expanded=size
    for f in root.rglob("*.npz"):
        with zipfile.ZipFile(f) as z:expanded+=sum(i.file_size for i in z.infolist())
    admission(3*expanded)
    if route=='pr_xy_td':
        from lcprop.pr.transverse.continuation import load_checkpoint as load
    elif route=='pr':
        from lcprop.pr.persistence import load_pr_checkpoint as load
    else:raise ValueError('Continue requires TD')
    return load(path)
