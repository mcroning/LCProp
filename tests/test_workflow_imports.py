from lcprop.workflows import run_static, run_timedependent


def test_public_workflow_imports():
    assert callable(run_static)
    assert callable(run_timedependent)
