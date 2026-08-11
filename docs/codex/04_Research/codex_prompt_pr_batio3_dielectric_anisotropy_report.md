# Codex Task: Create Archival Research Report for the BaTiO₃ Dielectric-Anisotropy Extension

## Objective

Create a comprehensive archival research report documenting the bounded BaTiO₃ dielectric-anisotropy extension to the already validated 2D transverse photorefractive hopping reference model.

This report should be comparable in scope, scientific depth, and archival value to the existing **Full Transverse Hopping-Model Recovery** report.

The goal is not to add new PR physics in this task. The goal is to preserve, explain, and validate the physics already implemented.

Do not modify production PR workflows, GUI, GPU support, or the governing transport equations. Do not add new numerical features.

---

## Output file

Create:

```text
docs/research/pr_batio3_dielectric_anisotropy_extension_2026-08-10.md
```

Use Markdown with VS Code-compatible math:

- inline math: `$...$`
- display math: `$$...$$`

Use the built-in VS Code Markdown renderer conventions.

Do not use `\(...\)` or `\[...\]`.

---

## Source material

Use the following repository artifacts as authoritative implementation sources:

```text
src/lcprop/pr/transverse_reference.py
tests/test_pr_transverse_reference.py
scripts/checks/pr_transverse_reference.py
docs/architecture/pr_transverse_reference_model.md
```

Also use the existing archival transverse-model report as context and style reference:

```text
docs/development/full_transverse_hopping_model_recovery_vscode.md
```

Do not silently rewrite or alter the original transverse-hopping derivation.

The new document should describe the anisotropy extension as a separate physics milestone built on top of that recovered model.

---

# Required report structure

## 1. Title and provenance

Use a title such as:

```text
# BaTiO₃ Dielectric Anisotropy in the Full Transverse Photorefractive Hopping Model
```

Include:

- date;
- repository/project name;
- statement that this is an archival research record;
- relationship to the prior full-transverse hopping-model recovery;
- statement that the implementation remains a bounded CPU reference model, not a production solver.

Clearly distinguish:

- physics derived analytically;
- implementation details extracted from source code;
- validation evidence extracted from tests/scripts;
- editorial explanatory text;
- verbatim Codex implementation log, if included.

---

## 2. Executive summary

Explain concisely:

- why the original 2D reference model used isotropic dielectric closure;
- why this was scientifically useful as a validation baseline;
- the subsequent recognition that tetragonal BaTiO₃ has strong dielectric anisotropy;
- why dielectric anisotropy affects the drift-driving electric field, not merely a downstream optical response;
- that the extension introduces crystal-axis rotation in the simulation x-z plane as the physically meaningful input;
- that the isotropic reference remains unchanged;
- that the new model remains deliberately limited to a rotated clamped dielectric tensor.

State explicitly that this extension does **not** implement the complete wavevector-dependent photorefractive effective dielectric response.

---

## 3. Relationship to the recovered transverse hopping model

Briefly restate the governing normalized transport/electrostatic system:

$$
\frac{\partial n}{\partial\tau}
=
\nabla_\perp\cdot
\left[
\mathbf M
\left(
\nabla_\perp(nI)-nI\mathbf e
\right)
\right]
$$

with

$$
n-1
=
\nabla_\perp\cdot(\mathbf H\mathbf e)
$$

and

$$
\nabla_\perp\times\mathbf e=0.
$$

For the potential formulation,

$$
\mathbf e
=
\mathbf e_h-\nabla_\perp\psi
$$

and

$$
n-1
=
-\nabla_\perp\cdot
\left(
\mathbf H\nabla_\perp\psi
\right).
$$

Explain that the original minimal reference model used

$$
\mathbf H=\mathbf I,
$$

which was an explicitly isotropic validation assumption.

Then explain why the dielectric tensor appears inside the electrostatic closure and therefore changes the field that feeds back into the drift term.

---

## 4. Physical significance of dielectric anisotropy

Explain carefully that the carrier flux contains a field-driven drift contribution:

$$
\mathbf j_p
=
\mathbf M
\left[
nI\mathbf e-\nabla_\perp(nI)
\right]
$$

in normalized form.

Therefore, changing the dielectric tensor changes the solution of the electrostatic problem, which changes:

- $E_x$;
- $E_y$;
- the direction and magnitude of carrier drift;
- subsequent carrier redistribution.

Emphasize that dielectric anisotropy is part of the nonlinear feedback loop.

Do not describe it as merely an optical-index correction.

---

## 5. BaTiO₃ crystal-frame dielectric tensor

Describe tetragonal BaTiO₃ as uniaxial for this bounded reference model.

Use:

$$
\boldsymbol{\varepsilon}_{\rm crystal}
=
\operatorname{diag}
(\varepsilon_a,\varepsilon_a,\varepsilon_c).
$$

Document the implemented reference values:

$$
\varepsilon_a=2200,
\qquad
\varepsilon_c=56.
$$

Explain that these are approximate room-temperature clamped dielectric reference values used for the bounded reference implementation.

State explicitly that:

- they are not universal constants;
- they are not the full photorefractive effective dielectric response;
- free/unclamped dielectric values are not being used;
- electromechanical contributions are outside the present implementation.

Do not introduce new values not already present in the implementation.

---

## 6. Coordinate systems and crystal-axis angle

Define the simulation axes explicitly:

- x and y are transverse;
- z is the optical propagation direction.

Define the crystal $c$-axis angle parameter exactly as implemented:

```text
c_axis_xz_angle_deg
```

Use the convention:

$$
\hat{\mathbf c}
=
(\cos\gamma,0,\sin\gamma).
$$

Therefore:

- $\gamma=0^\circ$: crystal c-axis along simulation x;
- $\gamma=90^\circ$: crystal c-axis along propagation z.

Include a small ASCII or Markdown schematic if useful, but do not generate external artwork.

Explain why this is a better physical parameter than directly exposing $h_y$ as the primary user-facing quantity.

State that direct $h_y$ remains only as a low-level/reference override.

---

## 7. Tensor rotation derivation

Derive the rotated dielectric tensor from

$$
\boldsymbol{\varepsilon}
=
\varepsilon_a\mathbf I
+
(\varepsilon_c-\varepsilon_a)
\hat{\mathbf c}\hat{\mathbf c}^{T}.
$$

Show the full rotated tensor, including the off-diagonal $xz$ term.

Explicitly derive:

$$
\varepsilon_{xx}
=
\varepsilon_a\sin^2\gamma
+
\varepsilon_c\cos^2\gamma,
$$

$$
\varepsilon_{yy}
=
\varepsilon_a,
$$

and

$$
\varepsilon_{xz}
=
(\varepsilon_c-\varepsilon_a)
\sin\gamma\cos\gamma
$$

with the sign consistent with the implemented rotation convention.

Then explain the transverse-slice approximation.

The 2D reference model solves only the transverse x-y electrostatic problem. Therefore the implemented transverse block is

$$
\boldsymbol{\varepsilon}_\perp
=
\begin{pmatrix}
\varepsilon_{xx} & 0\\
0 & \varepsilon_{yy}
\end{pmatrix}.
$$

The rotated full tensor does contain $\varepsilon_{xz}$, but that component is outside the present transverse-slice model because longitudinal material derivatives and longitudinal space-charge dynamics are omitted.

This point must be stated explicitly.

Do not imply that the fully rotated 3D tensor is diagonal.

---

## 8. Normalized transverse dielectric tensor

Explain that the solver normalizes by $\varepsilon_{xx}$:

$$
\mathbf H
=
\frac{\boldsymbol{\varepsilon}_\perp}{\varepsilon_{xx}}.
$$

Therefore,

$$
H_{xx}=1,
$$

and

$$
H_{yy}=h_y
=
\frac{\varepsilon_{yy}}{\varepsilon_{xx}}.
$$

Derive:

$$
\boxed{
h_y(\gamma)
=
\frac{\varepsilon_a}
{
\varepsilon_a\sin^2\gamma
+
\varepsilon_c\cos^2\gamma
}
}
$$

for the implemented angle convention.

Explain physically what this does to the elliptic operator.

For constant coefficients,

$$
-\psi_{xx}
-h_y\psi_{yy}
=
\rho'
$$

schematically, and in Fourier space:

$$
\hat\psi(k_x,k_y)
\propto
\frac{\hat\rho(k_x,k_y)}
{k_x^2+h_yk_y^2}.
$$

Explain that this makes the electrostatic response direction-dependent.

---

## 9. Analytic limiting cases

Document and derive all three implemented checks.

### 9.1 c-axis along x

For

$$
\gamma=0^\circ,
$$

show:

$$
\varepsilon_{xx}=56,
$$

$$
\varepsilon_{yy}=2200,
$$

$$
h_y=39.2857142857.
$$

### 9.2 45-degree rotation

For

$$
\gamma=45^\circ,
$$

show:

$$
\varepsilon_{xx}=1128,
$$

$$
\varepsilon_{yy}=2200,
$$

$$
h_y=1.95035460993.
$$

### 9.3 c-axis along z

For

$$
\gamma=90^\circ,
$$

show:

$$
\varepsilon_{xx}=2200,
$$

$$
\varepsilon_{yy}=2200,
$$

$$
h_y=1.
$$

Explain why the $\gamma=90^\circ$ case becomes transversely isotropic even though the full crystal remains anisotropic.

---

## 10. Implementation

Describe the implementation in prose, mapped directly to source.

Document:

### `RotatedUniaxialDielectric`

Explain:

```python
BATIO3_CLAMPED_EPSILON_A = 2200.0
BATIO3_CLAMPED_EPSILON_C = 56.0
```

and the dataclass:

```python
RotatedUniaxialDielectric
```

Describe how it derives:

- `epsilon_xx`;
- `epsilon_yy`;
- `h_y`.

### `TransverseReferenceOptions.from_barium_titanate_c_axis`

Explain the new constructor and its parameters.

Clarify that the constructor derives $h_y$ from the physical crystal orientation.

### Preservation of legacy/reference behavior

State explicitly that:

```python
TransverseReferenceOptions(..., h_y=1.0)
```

remains the default and is numerically unchanged.

Explain that direct $h_y$ specification remains available for low-level reference/testing use.

---

## 11. Validation tests

Describe each new test individually and explain what it establishes.

Include:

- `test_rotated_barium_titanate_dielectric_limits[0°]`
- `test_rotated_barium_titanate_dielectric_limits[45°]`
- `test_rotated_barium_titanate_dielectric_limits[90°]`
- `test_crystal_axis_constructor_exposes_derived_dielectric_and_preserves_default`
- `test_rotated_dielectric_changes_localized_transverse_electrostatics`

Explain the physics significance of each test rather than merely listing names.

---

## 12. Localized isotropic-vs-anisotropic comparison

Document the comparison case exactly as implemented:

- grid: 64×56;
- same localized intensity profile;
- $dt=0.002$;
- 100 steps;
- all parameters identical except dielectric model.

Compare:

1. isotropic reference:
   $h_y=1$;

2. BaTiO₃ tensor case:
   $\gamma=45^\circ$.

Include a table with every metric actually present in the script output.

The known values include:

| Quantity | Isotropic | BaTiO₃ $45^\circ$ |
|---|---:|---:|
| RMS $E_x$ | 0.005784893801 | 0.004548323925 |
| RMS $E_y$ | 0.009912676998 | 0.005646646440 |
| RMS $n-1$ | 0.009507603001 | 0.009742440973 |

Also include the actual maximum $E_x$ and maximum $E_y$ values from the recorded output or `metrics.json`.

Do not leave those rows blank.

If the values are stored in:

```text
/private/tmp/lcprop-pr-transverse-dielectric-reference/
```

read them from the generated metrics output.

Include the relative field differences:

$$
\Delta E_x^{\rm rel}=0.2798629641,
$$

$$
\Delta E_y^{\rm rel}=0.4386519584.
$$

Explain precisely how the relative differences are defined, based on the implementation.

Do not guess the norm definition; inspect the script/code and state it accurately.

---

## 13. Scientific interpretation

Discuss the important result:

- $E_x$ changes by roughly 28%;
- $E_y$ changes by roughly 44%;
- RMS carrier perturbation changes much less.

Explain that this demonstrates that dielectric anisotropy strongly changes the electrostatic field geometry even when the total magnitude of carrier redistribution changes modestly.

Explain why this matters for carrier drift.

Discuss, carefully and without overclaiming, that this provides a plausible mechanism for strong transverse spectral effects in BaTiO₃.

Do not claim that the historical $k_y$-narrowing calculation has been reproduced.

---

## 14. Relationship to the full photorefractive effective dielectric response

Add a scientifically careful section explaining that the present implementation uses a rotated clamped crystal-frame dielectric tensor.

State explicitly that more complete photorefractive treatments can introduce an angle- or wavevector-dependent effective dielectric response.

The present reference solver does not implement:

- $\varepsilon_{\rm eff}^{PR}(\mathbf k)$;
- electromechanical coupling;
- piezoelectric contributions to the effective dielectric response;
- arbitrary grating-orientation-dependent corrections.

Explain that implementing such a model would require a more general spectral dielectric operator rather than a single constant diagonal $\mathbf H$.

Do not implement it.

This section is explanatory only.

---

## 15. Optical response remains unchanged

State clearly that:

$$
E_{\rm active}=E_x
$$

remains the reference optical projection.

The crystal-axis angle introduced here currently affects only the dielectric electrostatic closure.

Explain that a future full Pockels-tensor optical model would also depend on crystal orientation, but that this is intentionally outside scope.

---

## 16. Deliberate limitations

Include and explain:

- no transport-anisotropy calibration;
- no historical $D_y/D_x$ recovery;
- no wavevector-dependent effective dielectric response;
- no electromechanical response;
- no full Pockels tensor;
- no arbitrary 3D crystal rotations;
- no GPU support;
- no implicit integrator;
- no production static solver;
- no GUI;
- no production optical-workflow integration;
- no historical $k_y$-narrowing reconstruction.

Explain that these are deliberate scope boundaries, not accidental omissions.

---

## 17. Validation summary

Include exact test results:

Focused test:

```text
python -m pytest -q tests/test_pr_transverse_reference.py
12 passed in 0.45s
```

Complete PR validation:

```text
194 passed, 9 skipped, 1 deselected in 35.60s
```

Saved 3720 µm snapshot:

```text
1 passed in 5.88s
```

Effective total:

```text
195 passed, 9 skipped
```

Also record that whitespace/no-index checks passed if verified from the implementation log.

---

## 18. Research milestone conclusion

Conclude that:

- the full transverse hopping model is now accompanied by a physically motivated anisotropic dielectric reference;
- crystal orientation in the x-z plane is a meaningful physical input;
- the anisotropy materially changes the drift-driving field;
- the isotropic model remains an important validation baseline;
- the current implementation remains intentionally a reference model rather than a production BaTiO₃ solver.

State explicitly that this completes the intended bounded dielectric-anisotropy extension.

Do not propose new implementation work in the conclusion.

---

# Appendix A — Verbatim Codex implementation report

Include the full Codex completion report for this anisotropy milestone verbatim in substance.

Clean only UI artifacts such as:

- timestamps;
- “Worked for …”;
- “Edited N files”;
- review/undo UI text.

Preserve:

- equations;
- file lists;
- metrics;
- test results;
- implementation notes;
- limitations.

Format it cleanly as Markdown.

---

# Appendix B — File inventory

List:

```text
src/lcprop/pr/transverse_reference.py
tests/test_pr_transverse_reference.py
scripts/checks/pr_transverse_reference.py
docs/architecture/pr_transverse_reference_model.md
```

Briefly describe the role of each.

---

# Appendix C — Validation outputs

Record the generated comparison-output directory:

```text
/private/tmp/lcprop-pr-transverse-dielectric-reference/
```

List the files actually generated there.

If `metrics.json` exists, include a compact Markdown table reproducing its contents.

Do not invent missing outputs.

---

## Important preservation rules

Do not:

- change the PR implementation;
- alter numerical values;
- invent literature values;
- invent missing metrics;
- rewrite the original transverse-model derivation;
- add new physics;
- run large parameter sweeps;
- change production behavior.

This task is documentation and archival synthesis only.

Use code, tests, generated metrics, and existing documentation as the source of truth.

If any requested metric or detail is absent, state that explicitly rather than guessing.

---

## Completion report

When done, report:

- file created;
- approximate line count;
- sections included;
- whether all equations render correctly in built-in VS Code Markdown Preview;
- whether maximum $E_x$ and $E_y$ metrics were recovered;
- whether the $\varepsilon_{xz}$ transverse-slice caveat was documented;
- whether Appendix A contains the full completion log;
- whether any repository source files were modified.

Do not commit unless explicitly instructed.
