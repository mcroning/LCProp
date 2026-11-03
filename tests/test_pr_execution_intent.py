"""Execution intent is portable metadata, never observed execution state."""
import json
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.gui.remote_execution import RemoteExecutionDiscovery
from lcprop.persistence import load_experiment, save_experiment
from lcprop.persistence.execution_intent import ExecutionIntent
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.runners.cluster_profiles import ClusterCatalog, ClusterProfile
from lcprop.runners.slurm import SlurmResourceProfile


@pytest.fixture
def window(monkeypatch):
    import lcprop.gui.remote_execution as remote
    app = QApplication.instance() or QApplication([])
    resources = (
        SlurmResourceProfile("CPU", "batch", "normal", "00:10:00", 2, 8),
        SlurmResourceProfile("H200", "gpu", "normal", "00:10:00", 2, 64,
                             gpus=1, require_cupy=True),
    )
    cluster = ClusterProfile("saved-cluster", "user@invalid", "/runs", "/python",
                             "/sources", resources, default_resource_profile="CPU")
    discovery = RemoteExecutionDiscovery(ClusterCatalog((cluster,), cluster.name))
    monkeypatch.setattr(remote, "discover_remote_execution", lambda path=None: discovery)
    w = PRMainWindow()
    yield w
    w.close()


@pytest.mark.parametrize("target", ["local", "slurm"])
@pytest.mark.parametrize("backend,precision", [("numpy", "float64"), ("cupy", "float32")])
def test_execution_intent_roundtrip(window, tmp_path, target, backend, precision):
    w = window
    w.execution_target_selector.setCurrentIndex(0 if target == "local" else 1)
    w.remote_execution_controls.resource_selector.setCurrentIndex(1)
    w.result_policy_selector.setCurrentIndex(w.result_policy_selector.findData("full"))
    w.evolution_panel.set_backend_spec(BackendSpec(backend=backend, precision=precision))
    scientific = w.build_request()
    intent = w._current_execution_intent()
    path = tmp_path / "intent.json"
    w.save_experiment_to(path)
    document = json.loads(path.read_text())
    assert document["execution_intent"] == intent.to_payload()
    assert set(document["execution_intent"]) == set(ExecutionIntent.__dataclass_fields__)
    w.execution_target_selector.setCurrentIndex(0)
    w.result_policy_selector.setCurrentIndex(0)
    w.remote_execution_controls.resource_selector.setCurrentIndex(0)
    loaded = w.load_experiment_from(path)
    assert loaded.execution_intent == intent
    assert w.execution_intent_was_saved
    assert w._current_execution_intent() == intent
    assert w.build_request() == scientific
    assert w.execution_target_selector.currentData() == target
    assert w.runner is (w.slurm_runner if target == "slurm" else w.local_runner)


@pytest.mark.parametrize("missing", ["cluster_profile", "resource_profile"])
def test_unavailable_target_preserved_and_blocked_until_explicit_selection(window, tmp_path, missing):
    w = window
    w.execution_target_selector.setCurrentIndex(1)
    path = tmp_path / "missing.json"
    scientific = w.build_request()
    w.save_experiment_to(path)
    document = json.loads(path.read_text())
    document["execution_intent"][missing] = "unavailable"
    path.write_text(json.dumps(document))
    loaded = w.load_experiment_from(path)
    assert w.build_request() == scientific
    assert w._saved_unresolved_intent == loaded.execution_intent
    assert w.execution_target_selector.currentData() == "slurm"
    assert "UNAVAILABLE/UNRESOLVED" in w.execution_intent_label.text()
    assert "Execution target: Slurm (unavailable/unresolved)" in w.describe_request(scientific)
    assert "Execution target: Local" not in w.describe_request(scientific)
    for call in (w._validate_execution_request, w._run_registered):
        with pytest.raises(ValueError, match="explicitly select"):
            call(scientific)
    copied = tmp_path / "saved-again.json"
    w.save_experiment_to(copied)
    assert load_experiment(copied).execution_intent == loaded.execution_intent
    # Refreshing discovery must not implicitly accept its default target.
    w.remote_execution_controls.refresh(w.remote_execution_controls.discovery)
    assert w._saved_unresolved_intent == loaded.execution_intent
    w.execution_target_selector.setCurrentIndex(0)
    assert w._saved_unresolved_intent is not None  # programmatic changes are not consent
    w.execution_target_selector.activated.emit(0)
    assert w._saved_unresolved_intent is None
    assert w.runner is w.local_runner


def test_legacy_absence_distinct_from_explicit_local(window, tmp_path):
    path = tmp_path / "legacy.json"
    window.save_experiment_to(path)
    assert load_experiment(path).execution_intent.target == "local"
    document = json.loads(path.read_text())
    del document["execution_intent"]
    document["schema_version"] = 2
    path.write_text(json.dumps(document))
    loaded = window.load_experiment_from(path)
    assert loaded.execution_intent is None and loaded.schema_version == 2
    assert not window.execution_intent_was_saved
    assert "absent" in window.execution_intent_label.text()


@pytest.mark.parametrize("change", [{"job_id": "123"}, {"precision": "float32"}])
def test_ephemeral_or_conflicting_intent_rejected_before_mutation(window, tmp_path, change):
    path = tmp_path / "invalid.json"
    window.evolution_panel.set_backend_spec(BackendSpec(backend="numpy", precision="float64"))
    window.save_experiment_to(path)
    before = window.build_request()
    document = json.loads(path.read_text())
    document["execution_intent"].update(change)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="execution intent"):
        window.load_experiment_from(path)
    assert window.build_request() == before


def test_missing_catalog_loads_science_and_requires_explicit_target(window, tmp_path):
    window.execution_target_selector.setCurrentIndex(1)
    path = tmp_path / "slurm.json"
    window.save_experiment_to(path)
    science = window.build_request()
    window.remote_execution_controls.refresh(RemoteExecutionDiscovery(ClusterCatalog()))
    window.load_experiment_from(path)
    assert window.build_request() == science
    assert window.execution_target_selector.currentData() == "slurm"
    with pytest.raises(ValueError, match="unavailable/unresolved"):
        window._validate_execution_request(science)


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("removed", ["resource", "cluster"])
def test_resolved_nondefault_identity_survives_refresh_then_requires_explicit_replacement(
    window, tmp_path, restored, removed,
):
    w = window
    controls = w.remote_execution_controls
    original = controls.discovery.catalog.clusters[0]
    replacement = replace(original, name="replacement")
    controls.refresh(RemoteExecutionDiscovery(ClusterCatalog((original, replacement))))
    w.execution_target_selector.setCurrentIndex(1)
    controls.resource_selector.setCurrentIndex(controls.resource_selector.findData("H200"))
    w.evolution_panel.set_backend_spec(BackendSpec(backend="numpy", precision="float64"))
    path = tmp_path / "resolved.json"
    w.save_experiment_to(path)
    if restored:
        w.load_experiment_from(path)
    expected = w._current_execution_intent()
    assert expected.resource_profile == "H200"
    for cluster in (original, replace(original, poll_interval=17)):
        controls.refresh(RemoteExecutionDiscovery(ClusterCatalog((cluster, replacement))))
        assert controls.cluster_selector.currentData() == original.name
        assert controls.resource_selector.currentData() == "H200"
        assert w._saved_unresolved_intent is None
        assert w.runner is w.slurm_runner
        assert w._remote_runner_kwargs()["resource_profile"] == "H200"
        assert w._current_execution_intent() == expected
        assert "resource=H200" in w.execution_intent_label.text()
    if removed == "resource":
        remaining = (replace(original, resource_profiles=(original.resource_profiles[0],)), replacement)
    else:
        remaining = (replacement,)
    controls.refresh(RemoteExecutionDiscovery(ClusterCatalog(remaining)))
    assert w.execution_target_selector.currentData() == "slurm"
    assert controls.cluster_selector.currentData() == original.name
    assert controls.resource_selector.currentData() == "H200"
    assert "unavailable" in (controls.cluster_selector.currentText() + controls.resource_selector.currentText())
    assert w._saved_unresolved_intent == expected
    assert "UNAVAILABLE/UNRESOLVED" in w.execution_intent_label.text()
    with pytest.raises(ValueError, match="explicitly select"):
        w._validate_execution_request(w.build_request())
    w.save_experiment_to(path)
    assert load_experiment(path).execution_intent == expected
    ci = controls.cluster_selector.findData(replacement.name)
    controls.cluster_selector.setCurrentIndex(ci)
    controls.cluster_selector.activated.emit(ci)
    ri = controls.resource_selector.findData("H200")
    controls.resource_selector.setCurrentIndex(ri)
    controls.resource_selector.activated.emit(ri)
    assert w._saved_unresolved_intent is None
    assert w.runner is w.slurm_runner
    assert "cluster=replacement" in w.execution_intent_label.text()
    w.save_experiment_to(path)
    new = load_experiment(path).execution_intent
    assert new.cluster_profile == "replacement" and new.resource_profile == "H200"


@pytest.mark.parametrize("unresolved", [False, True])
def test_late_load_failure_restores_complete_execution_transaction(window, tmp_path, monkeypatch, unresolved):
    w = window
    controls = w.remote_execution_controls
    a = controls.discovery.catalog.clusters[0]
    b = replace(a, name="cluster-B")
    controls.refresh(RemoteExecutionDiscovery(ClusterCatalog((a, b))))
    # Construct a valid saved B request, then add the existing late widget-precision failure.
    controls.select_profile_identity(b.name, "CPU")
    w.execution_target_selector.setCurrentIndex(1)
    w.result_policy_selector.setCurrentIndex(0)
    w.evolution_panel.set_backend_spec(BackendSpec(backend="numpy", precision="float64"))
    path = tmp_path / "late-failure.json"
    w.save_experiment_to(path)
    document = json.loads(path.read_text())
    # Use the codec to preserve the real payload structure.
    loaded = load_experiment(path)
    request = replace(loaded.request, material=replace(loaded.request.material, gain_length_product=0.1234567891))
    save_experiment(request, path, material_id=loaded.material_id,
                    workflow_id=loaded.workflow_id, execution_intent=loaded.execution_intent,
                    presentation_payload=loaded.presentation_payload)
    controls.select_profile_identity(a.name, "H200")
    w.execution_target_selector.setCurrentIndex(0)
    w.result_policy_selector.setCurrentIndex(w.result_policy_selector.findData("full"))
    w.evolution_panel.set_backend_spec(BackendSpec(backend="numpy", precision="float32"))
    if unresolved:
        intent = ExecutionIntent("slurm", a.name, "missing", "numpy", "float32", "full")
        w._restore_execution_intent(intent)
    before = w._capture_execution_controls()
    restore = w._restore_execution_intent
    reached = []
    def observe(intent):
        restore(intent)
        reached.append(w._capture_execution_controls())
    monkeypatch.setattr(w, "_restore_execution_intent", observe)
    with pytest.raises(ValueError, match="not exactly representable"):
        w.load_experiment_from(path)
    assert reached[-1]["target"] == "slurm"
    assert reached[-1]["cluster"] == b.name and reached[-1]["resource"] == "CPU"
    assert w._capture_execution_controls() == before
    if not unresolved:
        w.execution_target_selector.setCurrentIndex(1)
        w.execution_target_selector.activated.emit(1)
        assert w.runner is w.slurm_runner
        assert controls.selected_cluster().name == a.name
        assert w._remote_runner_kwargs()["resource_profile"] == "H200"
    # A successful load commits B rather than restoring the old snapshot.
    save_experiment(loaded.request, path, material_id=loaded.material_id,
                    workflow_id=loaded.workflow_id, execution_intent=loaded.execution_intent,
                    presentation_payload=loaded.presentation_payload)
    w.load_experiment_from(path)
    assert w.execution_target_selector.currentData() == "slurm"
    assert controls.selected_cluster().name == b.name
    assert controls.selected_resource_name() == "CPU"
    assert w._saved_unresolved_intent is None


def test_fresh_td_policy_is_initialized_before_any_configuration_change(window):
    w = window
    assert w.evolution_panel.workflow_id() == 'pr_timedependent'
    def check():
        combo = w.result_policy_selector
        assert [combo.itemData(i) for i in range(combo.count())] == ['fast', 'full']
        assert combo.itemText(0) == 'Fast / Exploratory'
        assert combo.itemText(1) == 'Full'
        assert not w.analysis_products_button.isEnabled()
    check()
    before = w.build_request()
    w.execution_target_selector.setCurrentIndex(w.execution_target_selector.findData('slurm'))
    check()
    assert w.build_request() == before
    for policy in ('fast', 'full'):
        w.result_policy_selector.setCurrentIndex(w.result_policy_selector.findData(policy))
        assert w._remote_runner_kwargs()['result_policy'] == policy
        assert w._current_execution_intent().retrieval_policy == policy
        assert 'Slurm' in w.describe_request(w.build_request())


@pytest.mark.parametrize('policy', ['fast', 'full'])
def test_td_policy_load_preserves_supported_combo(window, tmp_path, policy):
    w = window
    w._set_product_policy(policy)
    path = tmp_path/'td-policy.json'
    w.save_experiment_to(path)
    w.evolution_panel.set_workflow_id('pr_static')
    w._set_product_policy('analysis:complex_input')
    w.load_experiment_from(path)
    assert w.evolution_panel.workflow_id() == 'pr_timedependent'
    combo = w.result_policy_selector
    assert [combo.itemData(i) for i in range(combo.count())] == ['fast', 'full']
    assert combo.currentData() == policy
