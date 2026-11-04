"""Behavioral checks of isolated methods; no Qt imports, GUI, or calculations.

These tests compile selected methods from the source AST against fake widgets
and runners. They do not replace eventual Qt integration acceptance.
"""
import ast
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
             "_set_configuration_enabled", "_run_with_optional_image_analysis"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_PRExecutionSnapshot":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "PRMainWindow":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in names:
                    method.decorator_list = []
                    selected.append(method)
    env = dict(dataclass=dataclass, partial=partial, monotonic=lambda: 0,
               begin_request=lambda *a: None, traceback=SimpleNamespace(format_exc=lambda: "error"),
               report_failure=lambda *a: (_ for _ in ()).throw(AssertionError("unexpected failure")),
               PR_MATERIAL_ID="pr", PRLivePreviewPolicy=lambda: "live")
    for name in ("PRRunRequest", "PRStaticRunRequest", "PublishedStaticRequest",
                 "UnifiedFreshRequest", "PRTransverseStaticRunRequest",
                 "PRBeamPanelImageAmplificationRunRequest",
                 "PRImageAmplificationExperimentRequest"):
        env[name] = type(name, (), {})
    env["PRImageAnalysisSelection"] = lambda pump, signal: (pump, signal)
    env["add_optional_image_analysis"] = lambda base, *a, **k: base
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
    return env


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

    def test_running_disables_inspection_and_product_controls(self):
        states = {}
        def control(name):
            return SimpleNamespace(setEnabled=lambda value: states.update({name: value}),
                                   setVisible=lambda value: None)
        names = ("material_panel", "beam_panel", "grid_panel", "evolution_panel",
                 "input_panel", "execution_target_selector", "remote_execution_controls",
                 "run_button", "preview_button", "result_policy_selector",
                 "analysis_products_button", "experiment_file_buttons",
                 "load_checkpoint_button", "stop_button", "continue_button",
                 "save_checkpoint_button")
        w = SimpleNamespace(**{name: control(name) for name in names})
        w._update_product_controls = lambda: (
            states.update(result_policy_selector=True, analysis_products_button=True))
        w._refresh_checkpoint_controls = lambda: None
        self.env["_set_configuration_enabled"](w, False)
        for name in ("preview_button", "result_policy_selector", "analysis_products_button"):
            self.assertFalse(states[name])
        self.env["_set_configuration_enabled"](w, True)
        for name in ("preview_button", "result_policy_selector", "analysis_products_button"):
            self.assertTrue(states[name])


if __name__ == "__main__":
    unittest.main()
