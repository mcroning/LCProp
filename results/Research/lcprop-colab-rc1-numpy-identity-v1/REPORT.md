# Stage 3J — Hosted NumPy identity qualification

2026-10-17. **Identity gate PASS following user-supplied hosted origin verification; see closure below.** The initial review held scientific qualification pending that evidence. Strong evidence favors a duplicate-metadata inventory collision, rather than NumPy replacement or a broken active numerical stack. No dependency repair is warranted on current evidence.

## Findings

Stage 3I archive SHA-256: `f8a4b23cdc0e053876fb087ca5087a9a3a72a440e2750aed31cd67d5aa325c54`. Inspected in place, without executing or extracting archive contents.

- Loaded NumPy reports 2.1.3 before and after installation.
- The flattened installed-distribution inventory reports 1.26.4 at both times.
- pip explicitly reports: `Requirement already satisfied: numpy in /usr/local/lib/python3.13/dist-packages ... (2.1.3)`. SciPy is similarly reported as 1.16.3 at that location.
- pip's transaction contains only new LCProp, ipympl and jedi installations. No NumPy installation, replacement or removal occurs.
- The diagnostic's `installed()` function collapses all `metadata.distributions()` entries into a normalized-name dictionary. Multiple records with the same name overwrite each other in enumeration order. It does not retain their roots or enumeration order.
- The archive retains module names and loaded versions, but not NumPy's `__file__`, distribution roots or metadata-directory paths. Consequently it cannot establish the current duplicate directories or which filesystem package Python imported. A pip location is corroborating evidence, not a substitute for the loaded module path.

**Identified capture defect:** distribution identities from potentially different locations are merged by name. This makes the flattened value unsuitable as an authoritative loaded-library version. The observed result is strongly consistent with a repeat of the previously proven collision below. The exact RC1 filesystem cause remains to be confirmed; neither a genuine NumPy version mismatch nor a clean runtime should be asserted yet.

## Comparison with resolved 0.3.0 case

The retained `../lcprop-pypi-rc1-hosted-results-review-v1/hosted-dependency-clarification.json` and its report establish the earlier same-kernel observations:

| Identity | Version | Origin |
| --- | --- | --- |
| Loaded NumPy and matching metadata | 2.1.3 | `/usr/local/lib/python3.13/dist-packages` |
| Loaded SciPy and matching metadata | 1.16.3 | `/usr/local/lib/python3.13/dist-packages` |
| Additional NumPy metadata, enumerated later | 1.26.4 | `/usr/lib/python3/dist-packages` |

The later system record overwrote the earlier record in a name-keyed inventory. This was resolved without reinstalling dependencies or rerunning science. It did not demonstrate two active NumPy implementations in one process. An empty metadata_files list on the system record did not establish that metadata was absent; system packaging may use a different representation/file inventory.

The RC1 symptoms, versions and pip-selected root match that prior case. Historical clarification is not current-runtime evidence, so it cannot close RC1's origin check by itself. The Stage 3I characterization of the baseline as inconsistent was too broad if read as a confirmed numerical-stack fault: what is confirmed is disagreement between a loaded version and a lossy inventory.

## Exact read-only pre-run procedure

Use the accepted RC1 hosted notebook: install, perform the required manual session restart, and rerun its exact artifact/version/269-file/import-ownership verification. Do not run science. In that same kernel, run the following cell before pressing Run. It imports numerical libraries but performs no numerical calculation, installation, removal, reload or path modification. Preserve the complete JSON output, including duplicate metadata records, together with the RC1 setup provenance.

```python
import sys, json, importlib
from pathlib import Path
from importlib import metadata as md

names = ("numpy", "scipy", "lcprop")
modules = {name: importlib.import_module(name) for name in names}
loaded = {
    name: {"version": getattr(module, "__version__", None),
           "file": str(Path(module.__file__).resolve())}
    for name, module in modules.items()
}
records = []
for distribution in md.distributions():
    name = distribution.metadata.get("Name", "").lower().replace("_", "-")
    if name not in names:
        continue
    records.append({
        "name": name,
        "version": distribution.version,
        "root": str(Path(distribution.locate_file("")).resolve()),
        "package_init": str(Path(distribution.locate_file(name + "/__init__.py")).resolve()),
        "metadata_files": [str(distribution.locate_file(f))
            for f in (distribution.files or [])
            if str(f).endswith((".dist-info/METADATA", ".egg-info/PKG-INFO"))],
    })
# Keep every record; do not build a name-keyed version inventory.
selected = {name: {"version": md.version(name),
    "root": str(Path(md.distribution(name).locate_file("")).resolve())}
    for name in names}
checks = {}
for name in ("numpy", "scipy"):
    same_origin = [r for r in records
                   if r["name"] == name and r["package_init"] == loaded[name]["file"]]
    checks[name] = bool(same_origin) and all(
        r["version"] == loaded[name]["version"] for r in same_origin)
    checks[name + "_selected_metadata"] = (
        selected[name]["version"] == loaded[name]["version"] and
        Path(selected[name]["root"], name, "__init__.py").resolve()
        == Path(loaded[name]["file"]))
checks["lcprop_rc1"] = selected["lcprop"]["version"] == "0.4.0rc1"
report = {"python": sys.version, "executable": sys.executable,
          "sys_path": list(sys.path), "loaded": loaded,
          "selected_metadata": selected, "all_distributions": records,
          "checks": checks}
print(json.dumps(report, indent=2))
if not all(checks.values()):
    raise RuntimeError("Identity gate incomplete: retain output for review; do not run science.")
print("Library origin/version checks passed; retain RC1 artifact verification separately.")
```

This check intentionally fails closed if importlib's selected metadata disagrees with active library ownership, even if another matching record exists. Review that output rather than removing packages automatically. LCProp package byte identity and import ownership remain enforced by its existing 269-file gate; a version string alone cannot replace those checks.

## Acceptance interpretation

- If loaded NumPy and selected/matching metadata are 2.1.3 at `/usr/local`, and an additional 1.26.4 record is present at `/usr/lib`, the collision is confirmed. Record both; use 2.1.3 with its loaded origin for numerical provenance. Do not uninstall the system metadata to tidy the inventory.
- Require equivalent origin/version agreement for SciPy, successful RC1 package integrity checks, and the retained dependency-consistency gate. A different current Colab numerical stack must be recorded and reviewed rather than silently treated as the previous environment.
- If the origins or versions conflict, stop before science and preserve the full output. No installation, upgrade, deletion or reload is authorized by this audit.
- Passing this identity gate permits consideration of a separately authorized bounded scientific qualification; it does not itself prove cross-platform numerical parity, ABI completeness, convergence or snapshot/export correctness.

## Scope and remaining uncertainty

Local inspection verified the archive's existing pip logs, capture implementation and historical closure evidence. No current hosted kernel is accessible here; the required same-kernel origin output has not been supplied. The code above was syntax-checked locally, not executed against the hosted runtime. Only this report was added; existing evidence, notebooks, package/controller sources and unrelated work remain unchanged. No simulations, dependency changes, commits or pushes occurred.

**Decision: NO-GO pending the short read-only origin check; likely an inventory artifact, with no evidence supporting dependency replacement as the cause.** The manual restart remains independently required for backend discovery.

## Closure — hosted origins verified

The user supplied the requested same-kernel output. It is retained as `hosted-origin-verification.json`; Markdown-rendered `**init**.py` paths were normalized to `__init__.py`. This is user-supplied evidence, not an independent remote inspection. Earlier archive contents and reports remain preserved.

Loaded NumPy 2.1.3 and SciPy 1.16.3 both resolve under `/usr/local/lib/python3.13/dist-packages`, agreeing with selected distribution metadata and same-origin metadata records. LCProp resolves under that same root with selected distribution version 0.4.0rc1. Its module-level version is null because the check found no __version__ attribute; this is not a distribution-version failure and does not replace the separate 269-file verification.

An additional NumPy 1.26.4 metadata record under `/usr/lib/python3/dist-packages` appears later in enumeration. Flattening these records into the Stage 3I name-keyed dictionary retains 1.26.4, despite Python loading 2.1.3. This confirms the same inventory-key collision as the earlier 0.3.0 case in the checked hosted environment. It does not indicate two actively imported NumPy versions. The empty system metadata_files list does not prove metadata or package absence.

All five reported origin/version checks passed. **GO on the numerical-library identity prerequisite for separately authorized bounded hosted scientific qualification**, provided the accepted RC1 artifact/runtime verification has passed in that session. No reinstall, upgrade or removal is needed. Numerical parity, convergence, frozen-run ownership and export validation are not established by this identity check and remain separate gates. The manual restart requirement for interactive backend initialization remains unchanged. No scientific execution was authorized or performed by recording this closure.
