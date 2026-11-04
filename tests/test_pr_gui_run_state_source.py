"""Behavioral checks of isolated methods; no Qt imports, GUI, or calculations.

These tests compile selected methods from the source AST against fake widgets
and runners. They do not replace eventual Qt integration acceptance.
"""
import ast
import __future__
import builtins
from enum import Enum
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/lcprop/pr/gui/main_window.py"


def methods():
    tree = ast.parse(SOURCE.read_text())
    names = {"_PRExecutionSnapshot", "_capture_execution_snapshot",
             "_run_registered", "preview_request_clicked", "run_clicked",
             "_set_configuration_enabled", "_run_with_optional_image_analysis",
             "_update_product_controls", "_finish_background"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_PRExecutionSnapshot":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "PRMainWindow":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in names:
                    method.decorator_list = []
                    selected.append(method)
    # Extract the real policy helpers without importing the production package.
    policy_path = SOURCE.parents[2] / "transport/result_policy.py"
    policy_tree = ast.parse(policy_path.read_text())
    policy_env = {"Enum": Enum}
    policy_nodes = [node for node in policy_tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=policy_nodes, type_ignores=[]),
                 str(policy_path), "exec", flags=__future__.annotations.compiler_flag), policy_env)
    def isolated_import(name, *args, **kwargs):
        if name == "lcprop.transport.result_policy":
            return SimpleNamespace(static_product_selection=policy_env["static_product_selection"])
        raise AssertionError("Unexpected import in extracted method: " + name)
    env = dict(dataclass=dataclass, partial=partial, monotonic=lambda: 0,
               begin_request=lambda *a: None, traceback=SimpleNamespace(format_exc=lambda: "error"),
               report_failure=lambda *a: (_ for _ in ()).throw(AssertionError("unexpected failure")),
               PR_MATERIAL_ID="pr", PRLivePreviewPolicy=lambda: "live")
    env.update(__builtins__={**vars(builtins), "__import__": isolated_import},
               QSignalBlocker=FakeSignalBlocker,
               PR_STATIC_WORKFLOW="static", PR_PUBLISHED_STATIC_WORKFLOW="published",
               UNIFIED_WORKFLOW="unified",
               PUBLISHED_STATIC_ANALYSIS_PRODUCTS=("far_field_intensity", "complex_output"),
               UNIFIED_ANALYSIS_PRODUCTS=("far_field_intensity", "complex_output",
                                          "unified_intensity_volume"))
    for name in ("PRRunRequest", "PRStaticRunRequest", "PublishedStaticRequest",
                 "UnifiedFreshRequest", "PRTransverseStaticRunRequest",
                 "PRBeamPanelImageAmplificationRunRequest",
                 "PRImageAmplificationExperimentRequest"):
        env[name] = type(name, (), {})
    env["PRImageAnalysisSelection"] = lambda pump, signal: (pump, signal)
    env["add_optional_image_analysis"] = lambda base, *a, **k: base
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
    return env


class FakeSignalBlocker:
    """No signals exist in these fakes; only context-manager shape is needed."""
    def __init__(self, control):
        self.control = control

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeControl:
    def __init__(self, value=None, checked=False):
        self.enabled = True
        self.value = value
        self.checked = checked
        self.items = []

    def setEnabled(self, value):
        self.enabled = value

    def setVisible(self, value):
        self.visible = value

    def setToolTip(self, value):
        self.tooltip = value

    def setText(self, value):
        self.text = value

    def currentData(self):
        return self.value

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = value

    def count(self):
        return len(self.items)

    def itemText(self, index):
        return self.items[index][0]

    def itemData(self, index):
        return self.items[index][1]

    def clear(self):
        self.items.clear()
        self.value = None

    def addItem(self, label, value):
        self.items.append((label, value))

    def findData(self, value):
        return next((i for i, item in enumerate(self.items) if item[1] == value), -1)

    def setCurrentIndex(self, index):
        self.value = self.items[index][1] if index >= 0 else None


class RunStateTests(unittest.TestCase):
    def setUp(self):
        self.env = methods()

    def test_snapshot_detaches_remote_options(self):
        options = {"result_policy": "full", "resource_profile": "cpu"}
        runner = object()
        w = SimpleNamespace(runner=runner, slurm_runner=runner,
            result_policy_selector=SimpleNamespace(currentData=lambda: "full"),
            _require_execution_intent_resolved=lambda: None,
            _remote_runner_kwargs=lambda: options)
        snapshot = self.env["_capture_execution_snapshot"](w)
        options["result_policy"] = "fast"
        self.assertEqual(dict(snapshot.remote_kwargs)["result_policy"], "full")
        self.assertIs(snapshot.runner, runner)

    def dispatch(self, remote=False, policy="full"):
        calls = []
        runner = SimpleNamespace(run_registered=lambda *a, **kw: calls.append(kw))
        snapshot = self.env["_PRExecutionSnapshot"](runner, remote, policy,
            (("result_policy", policy), ("resource_profile", "cpu")))
        # Missing GUI attributes make any worker-side widget access fail.
        w = SimpleNamespace(_workflow_id_for_request=lambda request: "static")
        request = self.env["PRStaticRunRequest"]()
        self.env["_run_registered"](w, request, execution_snapshot=snapshot)
        return calls[0]

    def test_local_worker_uses_captured_policy_without_gui(self):
        self.assertEqual(self.dispatch()["_result_policy"], "full")

    def test_remote_worker_uses_captured_options_without_gui(self):
        result = self.dispatch(remote=True)
        self.assertEqual(result["result_policy"], "full")
        self.assertEqual(result["resource_profile"], "cpu")

    def test_inspection_does_not_touch_workspace_during_run_or_close(self):
        for running, closing in ((True, False), (False, True)):
            w = SimpleNamespace(_background_running=running, _close_requested=closing)
            self.env["preview_request_clicked"](w)

    def test_run_binds_snapshot_for_both_analysis_routes(self):
        for optional in (False, True):
            captured = []
            snapshot = object()
            request = object()
            w = SimpleNamespace(_background_running=False, _close_requested=False,
                build_request=lambda: request, describe_request=lambda r: "summary",
                _validate_execution_request=lambda r: None,
                _local_run_cost_guard=lambda r: True, last_checkpoint=None,
                _capture_execution_snapshot=lambda: snapshot,
                _run_registered=lambda r, **kw: kw,
                _run_with_optional_image_analysis=lambda r, **kw: kw,
                input_panel=SimpleNamespace(requests_optional_image_analysis=lambda: optional,
                    pump_channel=SimpleNamespace(currentData=lambda: "pump"),
                    signal_channel=SimpleNamespace(currentData=lambda: "signal")),
                _start_background=lambda *a, **kw: captured.append(kw))
            self.env["run_clicked"](w)
            w._capture_execution_snapshot = lambda: self.fail("late GUI read")
            result = captured[0]["runner_callable"](request)
            self.assertIs(result["execution_snapshot"], snapshot)
            if optional:
                self.assertEqual(result["selection"], ("pump", "signal"))

    def test_image_amplification_uses_captured_runner_and_remote_options(self):
        calls = []
        self.env["run_image_amplification_experiment"] = (
            lambda runner, request, **kw: calls.append((runner, kw)))
        runner = object()
        snapshot = self.env["_PRExecutionSnapshot"](
            runner, True, "full", (("result_policy", "full"),))
        request = self.env["PRImageAmplificationExperimentRequest"]()
        self.env["_run_registered"](object(), request, execution_snapshot=snapshot)
        self.assertIs(calls[0][0], runner)
        self.assertEqual(calls[0][1]["runner_kwargs"], {"result_policy": "full"})

    def product_window(self, workflow, target, policy):
        names = ("material_panel", "beam_panel", "grid_panel", "evolution_panel",
                 "input_panel", "execution_target_selector", "remote_execution_controls",
                 "run_button", "preview_button", "result_policy_selector",
                 "analysis_products_button", "experiment_file_buttons",
                 "load_checkpoint_button", "stop_button", "continue_button",
                 "save_checkpoint_button")
        w = SimpleNamespace(**{name: FakeControl() for name in names})
        w.material_panel.applied_field = FakeControl()
        w.evolution_panel.workflow_id = lambda: workflow
        w.execution_target_selector.value = target
        w.result_policy_selector.value = policy
        w._background_running = False
        w._close_requested = False
        w._analysis_actions = {
            name: FakeControl(checked=True) for name in
            ("input_intensity", "complex_output", "unified_intensity_volume")}
        w._update_product_controls = partial(self.env["_update_product_controls"], w)
        w._set_configuration_enabled = partial(self.env["_set_configuration_enabled"], w)
        w._refresh_checkpoint_controls = lambda: None
        return w

    def product_cases(self):
        # Workflow, execution target, configured policy, expected restored policy.
        return (("static", "local", "analysis:complex_output", "analysis:complex_output,input_intensity"),
                ("static", "slurm", "full", "full"),
                ("published", "local", "analysis:complex_output", "analysis:complex_output"),
                ("timedependent", "local", "analysis:complex_output", "fast"),
                ("timedependent", "slurm", "full", "full"))

    def assert_workflow_controls(self, w, workflow, target, expected_policy):
        supported = workflow in ("static", "published")
        self.assertEqual(w.analysis_products_button.enabled, supported)
        self.assertEqual(w.result_policy_selector.enabled, supported or target == "slurm")
        self.assertEqual(w.result_policy_selector.currentData(), expected_policy)
        self.assertEqual(w._analysis_actions["input_intensity"].enabled, workflow != "published")
        self.assertTrue(w._analysis_actions["complex_output"].enabled)
        self.assertFalse(w._analysis_actions["unified_intensity_volume"].enabled)
        self.assertFalse(w._analysis_actions["unified_intensity_volume"].checked)
        if workflow == "published":
            self.assertFalse(w._analysis_actions["input_intensity"].checked)
        if not supported:
            self.assertEqual([value for _, value in w.result_policy_selector.items], ["fast", "full"])
            # Individual ordinary actions retain their allowed flags, but the
            # unsupported workflow cannot open their disabled parent button.
            self.assertFalse(w.analysis_products_button.enabled)

    def test_product_refresh_respects_workflow_and_execution_target(self):
        for workflow, target, policy, expected in self.product_cases():
            with self.subTest(workflow=workflow, target=target):
                w = self.product_window(workflow, target, policy)
                w._update_product_controls()
                self.assert_workflow_controls(w, workflow, target, expected)

    def test_active_refresh_cannot_reenable_product_controls(self):
        for workflow, target, policy, _ in self.product_cases():
            with self.subTest(workflow=workflow, target=target):
                w = self.product_window(workflow, target, policy)
                w._background_running = True
                w._set_configuration_enabled(False)
                # Deliberately call the actual refresh after startup disabling:
                # this catches a refresh that accidentally re-enables controls.
                w._update_product_controls()
                self.assertFalse(w.preview_button.enabled)
                self.assertFalse(w.result_policy_selector.enabled)
                self.assertFalse(w.analysis_products_button.enabled)
                self.assertTrue(all(not action.enabled for action in w._analysis_actions.values()))

    def test_completion_restores_workflow_appropriate_product_controls(self):
        for workflow, target, policy, expected in self.product_cases():
            with self.subTest(workflow=workflow, target=target):
                w = self.product_window(workflow, target, policy)
                w._background_running = True
                w._set_configuration_enabled(False)
                self.env["_finish_background"](w)
                self.assertFalse(w._background_running)
                self.assertIsNone(w._active_request)
                self.assertTrue(w.preview_button.enabled)
                self.assert_workflow_controls(w, workflow, target, expected)


if __name__ == "__main__":
    unittest.main()
