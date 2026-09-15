from __future__ import annotations

from dataclasses import dataclass, replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.pr.gui.beam_panel import make_pr_beam_panel
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.grid_panel import PRGridPanel
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.material_panel import PRMaterialPanel
from lcprop.pr.gui.request_adapter import apply_pr_request, build_pr_request
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.specs import (
    PRRunRequest,
    PR_EULER_INTEGRATOR,
    PR_EXACT_MODAL_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)
from lcprop.pr.static_workflow import PRStaticRunRequest
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PRTransverseRunRequest,
)
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunRequest


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _controls():
    grid = PRGridPanel()
    return (
        PRMaterialPanel(),
        make_pr_beam_panel(
            x_aperture_um=grid.x_aperture_um.value(),
            y_aperture_um=grid.y_aperture_um.value(),
        ),
        grid,
        PREvolutionPanel(),
    )


def _build(controls):
    material, beams, grid, evolution = controls
    return build_pr_request(
        material_panel=material,
        beam_panel=beams,
        grid_panel=grid,
        evolution_panel=evolution,
    )


@pytest.mark.parametrize(
    ("evolution", "transport", "response", "request_type"),
    (
        ("static", "reduced_x", PR_MATERIAL_RESPONSE_NONLINEAR, PRStaticRunRequest),
        ("static", "reduced_x", PR_MATERIAL_RESPONSE_LINEARIZED, PRStaticRunRequest),
        (
            "static",
            "full_transverse",
            PR_MATERIAL_RESPONSE_NONLINEAR,
            PRTransverseStaticRunRequest,
        ),
        (
            "static",
            "full_transverse",
            PR_MATERIAL_RESPONSE_LINEARIZED,
            PRTransverseStaticRunRequest,
        ),
        (
            "time_dependent",
            "reduced_x",
            PR_MATERIAL_RESPONSE_NONLINEAR,
            PRRunRequest,
        ),
        (
            "time_dependent",
            "reduced_x",
            PR_MATERIAL_RESPONSE_LINEARIZED,
            PRRunRequest,
        ),
        (
            "time_dependent",
            "full_transverse",
            PR_MATERIAL_RESPONSE_NONLINEAR,
            PRTransverseRunRequest,
        ),
        (
            "time_dependent",
            "full_transverse",
            PR_MATERIAL_RESPONSE_LINEARIZED,
            PRTransverseRunRequest,
        ),
    ),
)
def test_independent_model_axes_build_all_eight_production_cells(
    app, evolution, transport, response, request_type
):
    controls = _controls()
    panel = controls[-1]
    panel.evolution.setCurrentIndex(panel.evolution.findData(evolution))
    panel.transport_model.setCurrentIndex(
        panel.transport_model.findData(transport)
    )
    panel.material_response.setCurrentIndex(
        panel.material_response.findData(response)
    )

    request = _build(controls)

    assert type(request) is request_type
    assert request.material_response.model == response
    assert "x-only" in panel.transport_model.itemText(0)
    assert "x-y" in panel.transport_model.itemText(1)
    assert "Experimental" not in panel.material_response.currentText()
    assert "Production model selection" in panel.algorithm_status.text()


def test_integrator_choices_follow_the_selected_td_model(app):
    panel = PREvolutionPanel()
    panel.evolution.setCurrentIndex(panel.evolution.findData("time_dependent"))

    def choices():
        return tuple(panel.integrator.itemData(i) for i in range(panel.integrator.count()))

    panel.transport_model.setCurrentIndex(panel.transport_model.findData("reduced_x"))
    panel.material_response.setCurrentIndex(
        panel.material_response.findData(PR_MATERIAL_RESPONSE_NONLINEAR)
    )
    assert choices() == (PR_SEMI_IMPLICIT_INTEGRATOR, PR_EULER_INTEGRATOR)

    panel.material_response.setCurrentIndex(
        panel.material_response.findData(PR_MATERIAL_RESPONSE_LINEARIZED)
    )
    assert choices() == (PR_EXACT_MODAL_INTEGRATOR,)
    assert panel.integrator.currentText() == "Exact modal evolution"

    panel.transport_model.setCurrentIndex(
        panel.transport_model.findData("full_transverse")
    )
    assert choices() == (PR_TRANSVERSE_IMEX_EULER,)
    assert panel.integrator.currentText() == "Exact modal evolution"

    panel.material_response.setCurrentIndex(
        panel.material_response.findData(PR_MATERIAL_RESPONSE_NONLINEAR)
    )
    assert choices() == (
        PR_TRANSVERSE_IMEX_EULER,
        PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    )


def test_image_amplification_status_is_mode_specific_not_a_model_label(app):
    panel = PREvolutionPanel()
    panel.set_image_amplification_mode(True)
    panel.material_response.setCurrentIndex(
        panel.material_response.findData(PR_MATERIAL_RESPONSE_LINEARIZED)
    )

    assert panel.material_response.currentText() == "Linearized"
    assert panel.image_amplification_validation_status() == (
        "compatible_validation_pending"
    )
    assert "Experimental" in panel.algorithm_status.text()


def test_scattering_controls_round_trip_where_production_supports_them(app):
    source = _controls()
    panel = source[-1]
    scattering = PRCanonicalScatteringSpec(
        epsilon=0.025,
        transverse_correlation_um=1.5,
        realization_seed=123456,
        canonical_dz_um=1.0,
        algorithm_version=PR_CANONICAL_SCATTERING_V2,
    )
    panel.set_scattering_spec(scattering)
    request = _build(source)
    assert request.scattering == scattering

    restored = _controls()
    apply_pr_request(
        request,
        material_panel=restored[0],
        beam_panel=restored[1],
        grid_panel=restored[2],
        evolution_panel=restored[3],
    )
    assert _build(restored) == request

    panel.evolution.setCurrentIndex(panel.evolution.findData("static"))
    assert panel.scattering_enabled.isHidden()
    assert not hasattr(_build(source), "scattering")


def test_legacy_full_linearized_td_integrator_token_round_trips(app):
    source = _controls()
    panel = source[-1]
    panel.transport_model.setCurrentIndex(
        panel.transport_model.findData("full_transverse")
    )
    panel.material_response.setCurrentIndex(
        panel.material_response.findData(PR_MATERIAL_RESPONSE_LINEARIZED)
    )
    request = _build(source)
    legacy = replace(
        request,
        solver=replace(
            request.solver,
            integrator=PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
        ),
    )
    restored = _controls()

    apply_pr_request(
        legacy,
        material_panel=restored[0],
        beam_panel=restored[1],
        grid_panel=restored[2],
        evolution_panel=restored[3],
    )

    assert restored[-1].integrator.currentText() == "Exact modal evolution"
    assert _build(restored) == legacy


@dataclass(frozen=True)
class _Profile:
    gpus: int
    require_cupy: bool


class _Cluster:
    def profile(self, name):
        return {
            "GPU": _Profile(gpus=1, require_cupy=True),
            "CPU": _Profile(gpus=0, require_cupy=False),
        }[name]


class _SlurmRunner:
    name = "Slurm"


def _remote_window():
    window = PRMainWindow(slurm_runner=_SlurmRunner())
    selected = {"resource": "GPU"}
    window.remote_execution_controls.selected_cluster = lambda: _Cluster()
    window.remote_execution_controls.selected_resource_name = (
        lambda: selected["resource"]
    )
    window.remote_execution_controls.create_runner = (
        lambda: window._explicit_slurm_runner
    )
    return window, selected


def _set_target(window, target):
    window.execution_target_selector.setCurrentIndex(
        window.execution_target_selector.findData(target)
    )


def _select_resource(window, selected, resource):
    selected["resource"] = resource
    window._remote_profile_changed()


def _saved_experiment(tmp_path, *, backend):
    source = PRMainWindow()
    source.evolution_panel.set_backend_spec(
        BackendSpec(backend, "float64", False)
    )
    path = tmp_path / f"saved-{backend}.lcprop.json"
    source.save_experiment_to(path)
    source.close()
    return path


def test_fresh_local_gpu_slurm_local_restores_implicit_numpy(app):
    window, _selected = _remote_window()
    assert window.evolution_panel.backend.currentText() == "numpy"
    assert window.evolution_panel.backend_origin == "implicit_local_default"

    _set_target(window, "slurm")
    assert window.evolution_panel.backend.currentText() == "cupy"
    assert window.evolution_panel.backend_origin == "automatic_target_default"
    assert not window.evolution_panel.backend_explicitly_selected

    _set_target(window, "local")
    assert window.evolution_panel.backend.currentText() == "numpy"
    assert window.evolution_panel.backend_origin == "implicit_local_default"
    assert not window.evolution_panel.backend_explicitly_selected
    window.close()


def test_repeated_local_gpu_slurm_transitions_are_reversible(app):
    window, _selected = _remote_window()
    for _ in range(3):
        _set_target(window, "slurm")
        assert window.evolution_panel.backend.currentText() == "cupy"
        assert window.evolution_panel.backend_origin == "automatic_target_default"
        _set_target(window, "local")
        assert window.evolution_panel.backend.currentText() == "numpy"
        assert window.evolution_panel.backend_origin == "implicit_local_default"
    window.close()


def test_gpu_cpu_gpu_slurm_resource_changes_follow_automatic_defaults(app):
    window, selected = _remote_window()
    _set_target(window, "slurm")
    assert window.evolution_panel.backend.currentText() == "cupy"

    _select_resource(window, selected, "CPU")
    assert window.evolution_panel.backend.currentText() == "numpy"
    assert window.evolution_panel.backend_origin == "automatic_target_default"
    _select_resource(window, selected, "GPU")
    assert window.evolution_panel.backend.currentText() == "cupy"
    assert not window.evolution_panel.backend_explicitly_selected
    window.close()


def test_explicit_numpy_on_gpu_slurm_is_never_overwritten(app):
    window, selected = _remote_window()
    _set_target(window, "slurm")
    window.evolution_panel.backend.setCurrentText("numpy")
    assert window.evolution_panel.backend_origin == "explicit_user_or_loaded"

    _select_resource(window, selected, "CPU")
    _select_resource(window, selected, "GPU")
    _set_target(window, "local")
    _set_target(window, "slurm")
    assert window.evolution_panel.backend.currentText() == "numpy"
    window.close()


def test_explicit_cupy_is_preserved_when_returning_to_local(app):
    window, _selected = _remote_window()
    window.evolution_panel.backend.setCurrentText("cupy")
    assert window.evolution_panel.backend_explicitly_selected
    _set_target(window, "slurm")
    _set_target(window, "local")
    assert window.evolution_panel.backend.currentText() == "cupy"
    assert window.evolution_panel.backend_origin == "explicit_user_or_loaded"
    window.close()


def test_experiment_load_while_local_establishes_explicit_intent(app, tmp_path):
    path = _saved_experiment(tmp_path, backend="auto")
    window, selected = _remote_window()
    loaded = window.load_experiment_from(path)

    assert loaded.request.backend.backend == "auto"
    assert window.evolution_panel.backend.currentText() == "auto"
    assert window.evolution_panel.backend_origin == "explicit_user_or_loaded"
    _set_target(window, "slurm")
    _select_resource(window, selected, "CPU")
    _select_resource(window, selected, "GPU")
    assert window.evolution_panel.backend.currentText() == "auto"
    window.close()


def test_experiment_load_while_gpu_slurm_is_not_overwritten(app, tmp_path):
    path = _saved_experiment(tmp_path, backend="numpy")
    window, selected = _remote_window()
    _set_target(window, "slurm")
    assert window.evolution_panel.backend.currentText() == "cupy"

    loaded = window.load_experiment_from(path)
    assert loaded.request.backend.backend == "numpy"
    assert window.evolution_panel.backend.currentText() == "numpy"
    assert window.evolution_panel.backend_origin == "explicit_user_or_loaded"
    _set_target(window, "local")
    _select_resource(window, selected, "CPU")
    _set_target(window, "slurm")
    _select_resource(window, selected, "GPU")
    assert window.evolution_panel.backend.currentText() == "numpy"
    window.close()


def test_reselecting_current_numpy_is_explicit_intent(app):
    window, _selected = _remote_window()
    window.evolution_panel.backend.activated.emit(
        window.evolution_panel.backend.currentIndex()
    )
    assert window.evolution_panel.backend_origin == "explicit_user_or_loaded"

    _set_target(window, "slurm")
    assert window.evolution_panel.backend.currentText() == "numpy"
    window.close()
