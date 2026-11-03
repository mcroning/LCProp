# Bootstrap-only acceptance

The previous acceptance was blocked because `test_notebook_binding` depended on the
excluded candidate notebook. The test is now retained intact in the deferred notebook
integration test file; all seven other tests are moved unchanged into the standalone
`tests/test_lc_colab_bootstrap.py`. No test skip or assertion removal was introduced.

Seven bootstrap tests passed against an isolated file view containing the committed
baseline notebook and candidate bootstrap. The notebook was obtained from Git HEAD;
the bootstrap tests do not read it. Tests cover verified downloads, 264 installed file
identities, preimported dependency reuse, missing/stale/broken dependencies, installer
failure, real widget initialization and same-object/snapshot preservation on rerun.
One deferred notebook integration test passes separately against the candidate notebook,
retaining bootstrap SHA-256, engine/helper pin and collapsed-cell checks.

No scientific execution or installation. Existing disposable installed environment
was reused. Logs are under `.codex-work/bootstrap-separation/`. Bootstrap and scientific
helper bytes remain unchanged by test separation. No new numerical qualification.

Approved commit inventory:

- `examples/lc_colab_bootstrap.py`
- `examples/lc_colab_bootstrap.md`
- `tests/test_lc_colab_bootstrap.py`
- `results/Research/lcprop-colab-interactive-stage-1d-v1/BOOTSTRAP_ACCEPTANCE.md`

Excluded: modified notebook, notebook usage guide, deferred integration test changes,
main Stage 1D report remediation, unrelated tracked changes, untracked results and
protected evidence. No push. Notebook revision pinning and publication remain pending.
