# RC1 hosted-Colab presentation correction

2026-10-17. Notebook-only correction prepared for hosted acceptance. Qualified package/artifacts remain unchanged.

## Evidence and scope

The user reported a backend-selection error resolved after session restart. Normal initialization then showed Run/Results without Configure. Direct display of the existing preview host rendered correctly, and direct display of the existing Configure body exposed the controls. The user exercised Beam 2 toggling, Move/Tilt and numeric edits successfully. These are user-reported browser observations, not newly automated hosted results. They isolate a presentation-path failure; the precise Colab accordion frontend cause is not established.

## Changes

- `notebooks/lc_static_pypi_rc1_colab_qualification.ipynb`: one retained notebook-owned VBox displays existing Configure contents, Run and Results directly. Original accordion is not displayed. No new controller, scientific widgets or callbacks. Container and HTML layouts use natural height, visible overflow and no flex shrinking; outer width is capped at 1000px. Canvas/output internals and Beam 2 display state are untouched. Repeated initialization reuses the view and controller.
- `notebooks/lc_static_pypi_rc1_colab_qualification.md`: initial-install manual session restart, verification rerun, view ownership and remaining acceptance instructions.
- `tests/test_rc1_colab_presentation.py`: three bounded tests using real ipywidgets and a fake controller; no scientific execution.
- This report.

After first installation, restart the session, rerun setup verification, then initialize. Do not delete the runtime. Ordinary compatible-session setup/initialization reruns need no restart. No automatic restart or live module reload is introduced. Existing completed records remain in the same controller; any manual restart loses in-memory records, so export first if needed.

## Validation

` .codex-work/beam-rc1/colab-env/bin/python -m pytest --noconftest tests/test_rc1_colab_presentation.py -q`: **3 passed**. Tests cover repeated view/widget identity, completed-reference preservation, callback count, hidden Beam 2 state, untouched preview-host layout, static backend ordering, empty outputs and cell compilation.

The setup/verification cell is byte-for-byte identical as a JSON cell to the published parent notebook. Exact TestPyPI version, both artifact hashes, production-PyPI dependency resolution and 269-file verification are unchanged. Production 0.3.0 notebook bytes match HEAD. No src/package/dependency changes, builds, installations or propagation.

Notebook SHA-256 before publication: `b8d2e01a7540e86a99d2a7937fefc4d6b3f1f42991eeb522b3e0f132048b125e`. Publication commit and anonymous retrieval receipt are recorded after commit in local `.codex-work/beam-stage3f-correction/publication.json` and the final response, avoiding a self-referential commit identifier.

## Publication and remaining checks

Publish only these four files on the existing RC1 qualification branch after staged/outgoing-history audit. The previous public tip is `d1a54ffc30f4a64ba6d67bacf732a9b46c22987d`; no earlier unpublished ancestor is required.

Fresh hosted acceptance remains pending: initial install/restart/verification, full Configure/Run/Results visibility, accessible controls without unnecessary nested scrollbars, first/repeated Beam 2 activation, Move/Tilt, numeric edits, repeated initialization and explicit static fallback. Browser layout is not proven by widget-tree tests. Do not also display diagnostic copies in other cells. No scientific calculation is needed for presentation qualification. Completed-run isolation/export acceptance in Colab remains pending separately authorized bounded execution.
