"""Bounded CPU notebook parity; no desktop toolkit, GPU or remote runtime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/pr_reduced_cpu.ipynb"


def code_cells():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
            if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def demo():
    import matplotlib
    matplotlib.use("Agg")
    ns = {}
    for cell in code_cells():
        exec(compile(cell, str(NOTEBOOK), "exec"), ns)
    yield ns
    ns["plt"].close("all")


def assert_array_bytes(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        assert a.shape == b.shape, key
        assert a.dtype == b.dtype, key
        assert a.tobytes() == b.tobytes(), key


def test_request_and_scientific_parity(demo):
    from lcprop.pr.workflow import run_pr_timedependent
    from lcprop.pr.products import pr_result_to_run_data
    codec = demo["PR_TIMEDEPENDENT_TRANSPORT_CODEC"]
    encoded = codec.encode_request(demo["request"])
    reopened = codec.decode_request(encoded.payload.metadata, encoded.payload.arrays)
    assert codec.encode_request(reopened).payload.metadata == encoded.payload.metadata
    # Product retains extra movie products when any progress callback is present.
    # Match that retention contract before comparing the complete codec inventory.
    reference = run_pr_timedependent(reopened, progress_callback=lambda event: None)
    assert reference.status == "completed"
    assert reference.completed_steps == demo["result"].completed_steps == 3
    assert reference.time_normalized == demo["result"].time_normalized == 0.003
    assert_array_bytes(codec.encode_result(reference).payload.arrays,
                       codec.encode_result(demo["result"]).payload.arrays)
    fields = pr_result_to_run_data(reference).fields
    assert_array_bytes({k: v.data for k, v in fields.items()},
                       {k: v.data for k, v in demo["run_data"].fields.items()})
    no_callback = run_pr_timedependent(reopened)
    scientific = ("A_initial", "A_final", "E_initial", "E_final", "source_intensity_stack")
    assert_array_bytes({k: getattr(no_callback, k) for k in scientific},
                       {k: getattr(reference, k) for k in scientific})


def test_progress_cpu_and_plot_coordinates(demo):
    assert demo["request"].backend.backend == "numpy"
    assert [p["completed"] for p in demo["progress_records"]] == [1, 2, 3]
    assert all(p["elapsed_s"] >= 0 for p in demo["progress_records"])
    data = demo["completed_run"].run_data
    xy = data.fields["output_intensity"]
    volume = data.fields["optical_intensity_stack"].data
    xz = volume[:, :, np.argmin(np.abs(data.geometry.y))]
    assert np.shares_memory(xz, volume)
    np.testing.assert_array_equal(demo["im_xy"].get_array(), xy.data.T)
    np.testing.assert_array_equal(demo["im_xz"].get_array(), xz.T)
    assert xy.value_unit == "1/µm²"
    demo["figure"].canvas.draw()


def test_persistence_and_overwrite_protection(demo, tmp_path):
    from lcprop.transport.io import read_result_package, read_request_package
    out = demo["save_demo"](tmp_path / "saved")
    registry = demo["TransportCodecRegistry"]()
    codec = demo["PR_TIMEDEPENDENT_TRANSPORT_CODEC"]
    registry.register(codec)
    reopened = read_result_package(out, registry=registry).result
    assert_array_bytes(codec.encode_result(reopened).payload.arrays,
                       codec.encode_result(demo["result"]).payload.arrays)
    assert read_request_package(out, registry=registry).envelope.execution_target == "local"
    experiments = demo["ExperimentCodecRegistry"]()
    experiments.register(demo["PR_TIMEDEPENDENT_EXPERIMENT_CODEC"])
    loaded = demo["read_experiment_file"](out / "experiment.lcprop.json", registry=experiments)
    assert codec.encode_request(loaded.request).payload.metadata == codec.encode_request(demo["request"]).payload.metadata
    with np.load(out / "selected_intensity.npz", allow_pickle=False) as selected:
        data = demo["completed_run"].run_data
        np.testing.assert_array_equal(selected["output_xy"], data.fields["output_intensity"].data)
        np.testing.assert_array_equal(selected["optical_xz"], data.fields["optical_intensity_stack"].data[:, :, np.argmin(np.abs(data.geometry.y))])
    manifest = json.loads((out / "notebook-manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert (out / "intensity.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        demo["save_demo"](out)


@pytest.mark.parametrize("order", ["parameters_changed", "execute_without_plot",
                                   "plot_without_execute", "mutated_globals"])
def test_completed_snapshot_out_of_order(tmp_path, order):
    from dataclasses import replace
    from lcprop.transport.io import read_request_package, read_result_package
    cells = code_cells()
    ns = {}
    for cell in cells:
        exec(cell, ns)
    first = ns["completed_run"]
    original_xy = first.run_data.fields["output_intensity"].data.copy()
    ns["request"] = replace(ns["request"], material=replace(
        ns["request"].material, gain_length_product=0.2))
    if order == "execute_without_plot":
        exec(cells[2], ns)
        assert ns["completed_run"] is not first
        assert ns["completed_run"].request.material.gain_length_product == 0.2
    elif order == "plot_without_execute":
        exec(cells[3], ns)
        exec(cells[3], ns)
        np.testing.assert_array_equal(ns["im_xy"].get_array(), original_xy.T)
    elif order == "mutated_globals":
        ns["result"].A_final[...] = 0
        ns["run_data"].fields["output_intensity"].data[...] = -1
        ns["environment"]["lcprop_version"] = "wrong"
        ns["figure"].clear()
    expected = ns["completed_run"]
    # Record actual export renderer inputs, including its mesh arrays and title.
    render = ns["plot_completed_run"]
    seen = []
    def record_render(snapshot):
        fig, axes, xy_mesh, xz_mesh = render(snapshot)
        seen.append((snapshot, xy_mesh.get_array().copy(), xz_mesh.get_array().copy(),
                     fig._suptitle.get_text()))
        return fig, axes, xy_mesh, xz_mesh
    ns["plot_completed_run"] = record_render
    out = ns["save_demo"](tmp_path / order)
    assert len(seen) == 1 and seen[0][0] is expected
    data = expected.run_data
    iy = np.argmin(np.abs(data.geometry.y))
    np.testing.assert_array_equal(seen[0][1], data.fields["output_intensity"].data.T)
    np.testing.assert_array_equal(seen[0][2], data.fields["optical_intensity_stack"].data[:, :, iy].T)
    assert f"τ={expected.result.time_normalized:g}" in seen[0][3]
    registry = ns["TransportCodecRegistry"]()
    codec = ns["PR_TIMEDEPENDENT_TRANSPORT_CODEC"]
    registry.register(codec)
    saved_request = read_request_package(out, registry=registry).request
    assert codec.encode_request(saved_request).payload.metadata == codec.encode_request(expected.request).payload.metadata
    saved_result = read_result_package(out, registry=registry).result
    assert_array_bytes(codec.encode_result(saved_result).payload.arrays,
                       codec.encode_result(expected.result).payload.arrays)
    with np.load(out / "selected_intensity.npz") as selected:
        np.testing.assert_array_equal(selected["output_xy"], seen[0][1].T)
        np.testing.assert_array_equal(selected["optical_xz"], seen[0][2].T)
    if order != "execute_without_plot":
        assert expected is first and saved_request.material.gain_length_product == 0.1
    manifest = json.loads((out / "notebook-manifest.json").read_text())
    assert manifest["environment"] == expected.environment
    ns["plt"].close("all")


def test_saving_before_execution_rejected_without_writes(tmp_path):
    ns = {}
    cells = code_cells()
    for index in (0, 1, 4):
        exec(cells[index], ns)
    with pytest.raises(RuntimeError, match="execute a successful run first"):
        ns["save_demo"](tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("missing", ["numpy", "scipy", "matplotlib", "lcprop"])
def test_missing_dependency_message(demo, missing):
    with pytest.raises(RuntimeError, match="Missing notebook dependencies: " + missing):
        demo["require_dependencies"](lambda name: None if name == missing else object())


def test_notebook_python310_grammar():
    for cell in code_cells():
        ast.parse(cell, feature_version=(3, 10))
    nb = json.loads(NOTEBOOK.read_text())
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in nb["cells"])


def test_fresh_kernel_headless_and_exact_outputs(demo, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    kernels = tmp_path / "jupyter" / "kernels" / "lcprop-test"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "LCProp temporary test kernel", "language": "python",
        "env": {"PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(tmp_path / "mpl"), "PYTHONDONTWRITEBYTECODE": "1"},
    }))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter"))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    nb.cells.insert(0, nbformat.v4.new_code_cell('''import sys, importlib.abc
class NoDesktopGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'PyQt6', 'launchplane', 'cupy'}:
            raise ImportError('Forbidden optional backend: ' + fullname)
sys.meta_path.insert(0, NoDesktopGPU())
'''))
    saved = tmp_path / "kernel-output"
    nb.cells.append(nbformat.v4.new_code_cell(
        f"save_demo(Path({str(saved)!r}))\n"
        "assert not ({'PySide6', 'PyQt6', 'launchplane', 'cupy'} & set(sys.modules))\n"
        "print('HEADLESS_CPU_COMPLETE')"))
    executed = nbclient.NotebookClient(nb, timeout=120, kernel_name="lcprop-test",
                                      resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert any("HEADLESS_CPU_COMPLETE" in o.get("text", "") for o in executed.cells[-1].outputs)
    from lcprop.transport.io import read_result_package
    registry = demo["TransportCodecRegistry"]()
    codec = demo["PR_TIMEDEPENDENT_TRANSPORT_CODEC"]
    registry.register(codec)
    result = read_result_package(saved, registry=registry).result
    assert_array_bytes(codec.encode_result(result).payload.arrays,
                       codec.encode_result(demo["result"]).payload.arrays)
    # Optional artifact destination, only for this audit's evidence capture.
    evidence = os.environ.get("LCPROP_NOTEBOOK_EXECUTED_PATH")
    if evidence:
        nbformat.write(executed, evidence)
