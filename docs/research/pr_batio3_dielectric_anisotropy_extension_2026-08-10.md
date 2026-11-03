# BaTiO₃ Dielectric Anisotropy in the Full Transverse Photorefractive Hopping Model

| Provenance field | Value |
| --- | --- |
| Date | 2026-08-10 |
| Project | LCProp |
| Branch at validation | `feature/pr-second-order-static` |
| Base Git HEAD | `e118616e7e72d3346e7df0cceac9df079c6b69b8` |
| Record type | Archival research and implementation record |

This document records the bounded BaTiO₃ dielectric-anisotropy extension to
LCProp's full transverse photorefractive hopping reference model. It is a
separate milestone built on the derivation preserved in
`docs/development/full_transverse_hopping_model_recovery_vscode.md`; it does
not replace or silently revise that derivation.

The implementation remains an isolated CPU/NumPy reference model. It is not a
production PR solver, is not connected to the production optical workflows,
and does not establish a complete tensor-calibrated BaTiO₃ material model.

## 1. Provenance and evidence classes

The implementation was validated from an uncommitted working-tree snapshot on
the base revision shown above. Four repository artifacts are authoritative for
what was implemented:

- `src/lcprop/pr/transverse_reference.py`;
- `tests/test_pr_transverse_reference.py`;
- `scripts/checks/pr_transverse_reference.py`;
- `docs/architecture/pr_transverse_reference_model.md`.

The following distinctions apply throughout this report:

- **Analytically derived physics** consists of the tensor rotation, transverse
  tensor extraction, normalization, and limiting-orientation calculations.
- **Implementation details** are statements read directly from the Python
  source and its public dataclasses and functions.
- **Validation evidence** consists of test results and values recorded by the
  comparison script in
  `/private/tmp/lcprop-pr-transverse-dielectric-reference/metrics.json`.
- **Editorial explanation** connects those sources and explains their
  significance without adding new physics.
- **Appendix A** preserves the Codex implementation completion report in
  substance, with chat-interface artifacts omitted.

## 2. Executive summary

The first full-transverse reference implementation deliberately used an
isotropic dielectric closure, $\mathbf H=\mathbf I$. That was the smallest
defensible model with which to validate the recovered two-dimensional hopping
equations, periodic electrostatics, carrier conservation, curl-free field
construction, Gauss closure, and exact one-dimensional reduction to the
production A7 equation. Keeping transport and electrostatics isotropic made
those structural tests interpretable.

Tetragonal BaTiO₃, however, has a strongly anisotropic dielectric tensor. In
the recovered transverse system, the dielectric tensor appears inside the
electrostatic closure. It therefore changes the electric field that drives
carrier drift; it is not merely a correction applied later to the optical
index.

The bounded extension introduces the orientation of the crystal c-axis in the
simulation x-z plane as the physical input. The parameter is
`c_axis_xz_angle_deg`. From this angle and the reference clamped dielectric
values, the implementation derives $\varepsilon_{xx}$,
$\varepsilon_{yy}$, and the normalized transverse ratio $h_y$. The original
isotropic path remains the default and is numerically unchanged.

The new option uses only a rotated, uniaxial, clamped crystal-frame dielectric
tensor. It does **not** implement the complete wavevector-dependent
photorefractive effective dielectric response
$\varepsilon_{\mathrm{eff}}^{PR}(\mathbf k)$, electromechanical coupling, or
a full electro-optic tensor contraction.

## 3. Relationship to the recovered transverse hopping model

Let the normalized mobile-carrier ratio be

$$
P=\frac{p}{p_d},
$$

where $P=1$ is the uniform neutral reference state. The normalized transverse
carrier equation recovered from the hopping model is

$$
\frac{\partial P}{\partial\tau}
=
\nabla_\perp\cdot
\left[
\mathbf M
\left(
\nabla_\perp(PI)-PI\mathbf e
\right)
\right].
$$

The carrier state is closed electrostatically through

$$
P-1
=
\nabla_\perp\cdot(\mathbf H\mathbf e),
$$

with the electrostatic constraint

$$
\nabla_\perp\times\mathbf e=0.
$$

For a prescribed harmonic applied-field component $\mathbf e_h$, the periodic
potential representation is

$$
\mathbf e
=
\mathbf e_h-\nabla_\perp\psi,
$$

and therefore

$$
P-1
=
-\nabla_\perp\cdot
\left(
\mathbf H\nabla_\perp\psi
\right).
$$

The original minimal reference used

$$
\mathbf H=\mathbf I.
$$

That was an explicit validation assumption rather than a claim that BaTiO₃ is
dielectrically isotropic. Because $\mathbf H$ is inside the elliptic closure,
changing it changes the potential, both transverse electric-field components,
and the carrier density derived from the potential. Those fields then feed
back into the transport equation.

The earlier recovery report remains authoritative for the derivation from the
discrete hopping equations, the continuum limit, and the proof that the local
scalar A7 equation cannot be generalized to two dimensions by merely adding
y derivatives.

## 4. Physical significance of dielectric anisotropy

In the normalized convention used by the recovered model, the carrier flux can
be written

$$
\mathbf j_p
=
\mathbf M
\left[
PI\mathbf e-\nabla_\perp(PI)
\right].
$$

The two terms represent field-driven drift and intensity/carrier-gradient
diffusion. The dielectric tensor does not appear explicitly in this flux
formula, but it determines $\mathbf e$ through the electrostatic closure.
Consequently, a dielectric change alters:

- $E_x$;
- $E_y$;
- the direction and magnitude of the drift term $PI\mathbf e$;
- the subsequent redistribution of carriers;
- the next electrostatic solution obtained from that redistributed charge.

Dielectric anisotropy is therefore part of the nonlinear transport-feedback
loop. Treating it only as a downstream optical-index factor would omit its
effect on carrier motion.

## 5. BaTiO₃ crystal-frame dielectric tensor

For this bounded reference model, tetragonal BaTiO₃ is represented as a
uniaxial dielectric with crystal-frame tensor

$$
\boldsymbol{\varepsilon}_{\rm crystal}
=
\operatorname{diag}
(\varepsilon_a,\varepsilon_a,\varepsilon_c).
$$

The implemented reference values are

$$
\varepsilon_a=2200,
\qquad
\varepsilon_c=56.
$$

They are approximate room-temperature clamped dielectric reference values.
They are exposed in the implementation as
`BATIO3_CLAMPED_EPSILON_A` and `BATIO3_CLAMPED_EPSILON_C`, and callers may
override them in the bounded constructor.

These numbers are not treated as universal constants. In particular:

- they are not free or unclamped dielectric values;
- they are not a complete photorefractive effective dielectric response;
- electromechanical contributions are not represented;
- their use does not calibrate the entire BaTiO₃ photorefractive model.

No additional dielectric values are introduced in this milestone.

## 6. Coordinate systems and crystal-axis angle

The simulation axes are:

- x: first transverse direction and the direction of the scalar active field;
- y: second transverse direction;
- z: optical propagation direction.

The physical orientation parameter is

```text
c_axis_xz_angle_deg
```

and its convention is

$$
\hat{\mathbf c}
=
(\cos\gamma,0,\sin\gamma).
$$

Thus:

- $\gamma=0^\circ$ places the crystal c-axis along simulation x;
- $\gamma=90^\circ$ places the crystal c-axis along propagation z.

A schematic view of the x-z plane is

```text
                       z (propagation)
                       ^
                       |      c-hat
                       |     /
                       |    / gamma
                       |   /
                       +----------------> x

gamma = 0 degrees:  c-hat parallel to x
gamma = 90 degrees: c-hat parallel to z
```

The angle is a better physical input than $h_y$ because it specifies a crystal
orientation from which the transverse dielectric coefficients follow. A raw
$h_y$ does not identify the crystal tensor or the geometry that produced it.
Direct $h_y$ input nevertheless remains available as a low-level validation
and testing override, preserving the original reference API.

## 7. Tensor rotation derivation

For a uniaxial material whose c-axis is $\hat{\mathbf c}$, the dielectric
tensor in simulation coordinates can be written without constructing an
explicit rotation matrix:

$$
\boldsymbol{\varepsilon}
=
\varepsilon_a\mathbf I
+
(\varepsilon_c-\varepsilon_a)
\hat{\mathbf c}\hat{\mathbf c}^{T}.
$$

For

$$
\hat{\mathbf c}
=
(\cos\gamma,0,\sin\gamma),
$$

the dyadic product is

$$
\hat{\mathbf c}\hat{\mathbf c}^{T}
=
\begin{pmatrix}
\cos^2\gamma & 0 & \sin\gamma\cos\gamma\\
0 & 0 & 0\\
\sin\gamma\cos\gamma & 0 & \sin^2\gamma
\end{pmatrix}.
$$

The fully rotated tensor is therefore

$$
\boldsymbol{\varepsilon}
=
\begin{pmatrix}
\varepsilon_a\sin^2\gamma+\varepsilon_c\cos^2\gamma
& 0
& (\varepsilon_c-\varepsilon_a)\sin\gamma\cos\gamma\\
0
& \varepsilon_a
& 0\\
(\varepsilon_c-\varepsilon_a)\sin\gamma\cos\gamma
& 0
& \varepsilon_a\cos^2\gamma+\varepsilon_c\sin^2\gamma
\end{pmatrix}.
$$

In particular,

$$
\varepsilon_{xx}
=
\varepsilon_a\sin^2\gamma
+
\varepsilon_c\cos^2\gamma,
$$

$$
\varepsilon_{yy}=\varepsilon_a,
$$

and

$$
\varepsilon_{xz}
=
(\varepsilon_c-\varepsilon_a)
\sin\gamma\cos\gamma.
$$

The sign of $\varepsilon_{xz}$ follows directly from the implemented
$\hat{\mathbf c}=(\cos\gamma,0,\sin\gamma)$ convention. Since
$\varepsilon_c<\varepsilon_a$ for the reference constants, the component is
negative for angles between zero and 90 degrees.

### 7.1 Transverse-slice approximation

The reference model solves only the transverse x-y electrostatic problem. It
omits longitudinal material derivatives and longitudinal space-charge
dynamics. The implemented block is consequently

$$
\boldsymbol{\varepsilon}_\perp
=
\begin{pmatrix}
\varepsilon_{xx} & 0\\
0 & \varepsilon_{yy}
\end{pmatrix}.
$$

The full rotated three-dimensional tensor is not diagonal: it contains
$\varepsilon_{xz}$. That component is outside the transverse-slice model; it
has not been discarded because it is mathematically zero. A model retaining
longitudinal material dynamics would have to revisit this approximation.

## 8. Normalized transverse dielectric tensor

The solver normalizes the transverse block by $\varepsilon_{xx}$:

$$
\mathbf H
=
\frac{\boldsymbol{\varepsilon}_\perp}{\varepsilon_{xx}}.
$$

It follows that

$$
H_{xx}=1,
$$

and

$$
H_{yy}=h_y
=
\frac{\varepsilon_{yy}}{\varepsilon_{xx}}.
$$

For the implemented angle convention,

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
}.
$$

For constant coefficients, the electrostatic closure contains the anisotropic
elliptic operator

$$
-\psi_{xx}-h_y\psi_{yy}=\rho'
$$

schematically, where $\rho'$ denotes the normalized carrier perturbation. In
Fourier space, its inversion has the form

$$
\hat\psi(k_x,k_y)
\propto
\frac{\hat\rho(k_x,k_y)}
{k_x^2+h_yk_y^2}.
$$

The denominator weights transverse Fourier directions differently whenever
$h_y\ne1$. The same carrier perturbation can therefore produce a different
potential and different $E_x=-\psi_x$ and $E_y=-\psi_y$ fields.

## 9. Analytic limiting cases

### 9.1 Crystal c-axis along x

For

$$
\gamma=0^\circ,
$$

$\sin\gamma=0$ and $\cos\gamma=1$, giving

$$
\varepsilon_{xx}=\varepsilon_c=56,
$$

$$
\varepsilon_{yy}=\varepsilon_a=2200,
$$

and

$$
h_y=\frac{2200}{56}=39.2857142857.
$$

### 9.2 Forty-five-degree rotation

For

$$
\gamma=45^\circ,
$$

$\sin^2\gamma=\cos^2\gamma=1/2$, so

$$
\varepsilon_{xx}
=
\frac{\varepsilon_a+\varepsilon_c}{2}
=1128,
$$

$$
\varepsilon_{yy}=2200,
$$

and

$$
h_y
=
\frac{2200}{1128}
=1.95035460993.
$$

### 9.3 Crystal c-axis along z

For

$$
\gamma=90^\circ,
$$

$\sin\gamma=1$ and $\cos\gamma=0$, giving

$$
\varepsilon_{xx}=\varepsilon_a=2200,
$$

$$
\varepsilon_{yy}=\varepsilon_a=2200,
$$

and

$$
h_y=1.
$$

The full crystal remains anisotropic because its c-axis is along z and its
longitudinal dielectric component differs from the a-axis value. The x-y block
seen by the transverse-slice model is nevertheless isotropic.

## 10. Implementation

### 10.1 `RotatedUniaxialDielectric`

The source declares

```python
BATIO3_CLAMPED_EPSILON_A = 2200.0
BATIO3_CLAMPED_EPSILON_C = 56.0
```

and the frozen dataclass

```python
RotatedUniaxialDielectric
```

with fields:

- `c_axis_xz_angle_deg`;
- `epsilon_a`;
- `epsilon_c`.

Construction validates that all three values are finite and that both
dielectric constants are positive. Read-only properties calculate:

- `epsilon_xx` from the rotated x component;
- `epsilon_yy` as `epsilon_a`;
- `h_y` as `epsilon_yy / epsilon_xx`.

The object preserves the physical inputs and exposes the derived quantities
used by the numerical reference model.

### 10.2 `TransverseReferenceOptions.from_barium_titanate_c_axis`

The named constructor accepts:

- `dx_normalized`;
- `dy_normalized`;
- `dt_normalized`;
- `steps`;
- `c_axis_xz_angle_deg`;
- the existing optional `m_y` and `applied_field_x` controls;
- optional overrides for `epsilon_a` and `epsilon_c`.

It constructs a `RotatedUniaxialDielectric`, assigns its derived `h_y` to the
existing solver option, and retains the dielectric object as provenance on the
options instance. Validation rejects an options object whose stored `h_y`
does not match its retained dielectric derivation.

No new transport model is introduced. The existing `m_y` remains one by
default, and no transport-anisotropy calibration occurs.

### 10.3 Numerical path

The pre-existing potential solver remains unchanged in structure. Its Fourier
elliptic denominator is

```python
denominator = kx * kx + options.h_y * ky * ky
```

for both state reconstruction and potential-rate inversion. The anisotropic
constructor therefore reaches the existing constant-tensor numerical path by
supplying a physically derived `h_y`; it does not duplicate or bypass the
electrostatic operator.

### 10.4 Preservation of the isotropic reference

The direct path

```python
TransverseReferenceOptions(..., h_y=1.0)
```

remains the default. Its `dielectric` metadata is `None`, and all earlier
reference tests continue to use the same numerical equations and values.
Direct `h_y` specification remains available for low-level numerical tests and
controlled reference cases.

## 11. Validation tests

### 11.1 `test_rotated_barium_titanate_dielectric_limits[0°]`

This parameterized case verifies that a c-axis along x exposes the c-axis
dielectric value in the simulation x direction and derives
$h_y=\varepsilon_a/\varepsilon_c$. It checks the most anisotropic of the three
analytic orientations.

### 11.2 `test_rotated_barium_titanate_dielectric_limits[45°]`

This case verifies the equal sine-squared and cosine-squared weighting at 45
degrees. It establishes $\varepsilon_{xx}=1128$ and
$h_y=2200/1128$ without relying on the evolution solver.

### 11.3 `test_rotated_barium_titanate_dielectric_limits[90°]`

This case verifies that placing the c-axis along z leaves an isotropic x-y
block, $\varepsilon_{xx}=\varepsilon_{yy}=\varepsilon_a$, and hence $h_y=1$.
It tests the angle convention as well as the limiting algebra.

### 11.4 `test_crystal_axis_constructor_exposes_derived_dielectric_and_preserves_default`

This test compares the low-level default with the named physical constructor.
It verifies that the original default remains $h_y=1$ with no dielectric
metadata, while the 45-degree constructor retains its physical dielectric
object and exposes $\varepsilon_{xx}=1128$ and the matching derived $h_y$.

### 11.5 `test_rotated_dielectric_changes_localized_transverse_electrostatics`

This evolution test applies the same localized two-dimensional intensity and
all the same numerical controls to isotropic and 45-degree dielectric cases.
It requires non-negligible relative differences in both $E_x$ and $E_y$. This
establishes that the new physical input changes the electrostatic solution,
not merely stored metadata.

The earlier reduction, conservation, curl/Gauss, active-field projection, and
input-validation tests remain active and passed in the same focused suite.

## 12. Localized isotropic-versus-anisotropic comparison

The comparison script uses:

- grid: 64 by 56;
- normalized spacings: `dx=dy=0.5`;
- a displaced, elliptical localized intensity plus uniform level 0.1;
- explicit Euler integration;
- normalized timestep $dt=0.002$;
- 100 steps;
- final normalized time 0.2;
- $m_y=1$ and zero applied field in both cases;
- active-field projection $(1,0)$, meaning $E_{\rm active}=E_x$.

Only the dielectric configuration differs:

1. isotropic reference, $h_y=1$;
2. rotated BaTiO₃ reference, $\gamma=45^\circ$ and
   $h_y=1.9503546099290785$.

The recorded metrics are:

For each field, the reported maximum is the maximum absolute sampled value,
implemented as `np.max(np.abs(field))`. It is not the maximum signed value
`np.max(field)`.

| Quantity | Isotropic | BaTiO₃ $45^\circ$ |
| --- | ---: | ---: |
| $h_y$ | 1.0 | 1.9503546099290785 |
| RMS $E_x$ | 0.005784893800870564 | 0.004548323925454413 |
| Maximum $\lvert E_x\rvert$ | 0.04285247526208585 | 0.032166094893785836 |
| RMS $E_y$ | 0.009912676998390971 | 0.005646646439823533 |
| Maximum $\lvert E_y\rvert$ | 0.07389087289865454 | 0.04285239891644005 |
| RMS $P-1$ | 0.009507603001085742 | 0.009742440973425737 |
| Relative carrier drift | $1.2688263138573217\times10^{-16}$ | 0.0 |

For a field $F$, the script defines the anisotropic difference relative to
the isotropic field as the array Euclidean/Frobenius norm

$$
\Delta F^{\rm rel}
=
\frac{
\left\|F_{\rm iso}-F_{\rm aniso}\right\|_2
}{
\left\|F_{\rm iso}\right\|_2
}.
$$

It records

$$
\Delta E_x^{\rm rel}=0.27986296408469163,
$$

and

$$
\Delta E_y^{\rm rel}=0.43865195842039445.
$$

The generated figure uses shared color limits for corresponding isotropic and
anisotropic fields, preventing independent rescaling from hiding their
amplitude differences.

## 13. Scientific interpretation

The 45-degree dielectric model changes $E_x$ by approximately 28 percent and
$E_y$ by approximately 44 percent in relative field norm. Both field RMS
values and both maximum magnitudes decrease in this comparison. The RMS
carrier perturbation changes much less, from approximately 0.009508 to
0.009742.

This distinction is important. A similar overall magnitude of carrier
redistribution does not imply a similar electric-field geometry. The
anisotropic elliptic operator redistributes the potential response in Fourier
space and changes its gradients. Since those gradients are the electric field,
they change the drift term that controls subsequent carrier motion.

The comparison demonstrates a physically plausible route by which dielectric
anisotropy could have substantial transverse spectral consequences in
BaTiO₃. It does not establish that this constant-tensor model reproduces the
historical $k_y$-narrowing calculation, whose coefficient set and original
implementation were not recovered.

## 14. Relationship to the full photorefractive effective dielectric response

The implemented tensor is the rotation of a constant clamped crystal-frame
tensor. More complete photorefractive treatments can produce an effective
dielectric response that depends on grating direction or wavevector.

The reference solver does not implement:

- $\varepsilon_{\rm eff}^{PR}(\mathbf k)$;
- electromechanical coupling;
- piezoelectric contributions to an effective dielectric response;
- arbitrary grating-orientation corrections.

Such a model would generally require a more general spectral dielectric
operator whose response changes with $\mathbf k$, rather than the single
constant diagonal matrix $\mathbf H=\operatorname{diag}(1,h_y)$ used here.
This report records that distinction but does not specify or implement such an
operator.

## 15. Optical response remains unchanged

The reference optical projection remains

$$
E_{\rm active}=E_x.
$$

The crystal-axis angle introduced in this milestone affects only the
dielectric electrostatic closure. It is not passed into a Pockels-tensor
contraction, and $E_y$ is not independently applied to an optical phase
screen.

A complete electro-optic model would also depend on crystal orientation,
optical polarization, and the relevant Pockels coefficients. That calculation
is intentionally outside this bounded extension.

## 16. Deliberate limitations

The following boundaries are deliberate rather than accidental omissions:

- no transport-anisotropy calibration;
- no recovery or fitting of the historical $D_y/D_x$;
- no wavevector-dependent effective dielectric response;
- no electromechanical or piezoelectric response;
- no full Pockels-tensor optical model;
- no arbitrary three-dimensional crystal rotations or Euler angles;
- no GPU or CuPy implementation;
- no implicit time integrator;
- no production static solver;
- no GUI controls;
- no production optical-workflow integration;
- no reconstruction of the historical $k_y$-narrowing calculation.

These limits preserve the role of the implementation as a transparent physics
reference. They prevent a validated tensor-rotation seam from being mistaken
for a complete production BaTiO₃ model.

## Architectural status

`src/lcprop/pr/transverse_reference.py` is the validated PR reference model for
the recovered full-transverse hopping physics and the bounded rotated-
dielectric extension. It is intentionally isolated from production workflows:
the production PR path continues to use its established reduced model, and no
production request, result, persistence, GUI, execution, or optical-workflow
interface depends on this reference module.

That isolation is an architectural boundary, not an indication that the
reference is unfinished. It allows the recovered equations and their physics
gates to remain inspectable while avoiding an unvalidated replacement of
production behavior. A future package reorganization may place PR in a peer
material channel alongside LC, but this milestone neither performs nor
requires that relocation. It introduces no plugin framework or new public
material API.

## 17. Validation summary

The focused reference command was

```text
python -m pytest -q tests/test_pr_transverse_reference.py
```

with result

```text
12 passed in 0.45s
```

The complete PR validation excluding the separately invoked large saved
snapshot was

```text
194 passed, 9 skipped, 1 deselected in 35.60s
```

The saved 3720 micrometer snapshot test was then run independently:

```text
1 passed in 5.88s
```

The effective combined result was

```text
195 passed, 9 skipped
```

Python syntax compilation passed for the reference source, tests, and
comparison script. No-index whitespace checks produced no errors for all four
reference artifacts.

## 18. Research milestone conclusion

The recovered full transverse hopping model is now accompanied by a
physically motivated anisotropic dielectric reference. Crystal c-axis
orientation in the simulation x-z plane is the primary physical input, and the
implementation derives the normalized transverse dielectric response from
that orientation.

The localized comparison confirms that this anisotropy materially changes the
drift-driving $E_x$ and $E_y$ fields while changing the RMS carrier
perturbation much less. The isotropic model remains unchanged and continues to
serve as the essential structural-validation baseline.

The implementation remains intentionally bounded: it is a CPU reference for a
rotated clamped dielectric tensor, not a production or fully calibrated
BaTiO₃ photorefractive solver. This completes the intended dielectric-
anisotropy extension.

---

# Appendix A — Codex implementation completion report

Implemented the bounded crystal-axis-controlled dielectric anisotropy
extension. Production PR workflows and physics remain unchanged, and nothing
was committed.

## Files changed

- `src/lcprop/pr/transverse_reference.py`
- `tests/test_pr_transverse_reference.py`
- `scripts/checks/pr_transverse_reference.py`
- `docs/architecture/pr_transverse_reference_model.md`

The earlier completion/recovery documents were not changed.

## Resulting implementation

The principal implementation addition is:

```python
BATIO3_CLAMPED_EPSILON_A = 2200.0
BATIO3_CLAMPED_EPSILON_C = 56.0

@dataclass(frozen=True)
class RotatedUniaxialDielectric:
    c_axis_xz_angle_deg: float
    epsilon_a: float = BATIO3_CLAMPED_EPSILON_A
    epsilon_c: float = BATIO3_CLAMPED_EPSILON_C

    @property
    def epsilon_xx(self) -> float:
        gamma = math.radians(float(self.c_axis_xz_angle_deg))
        return (
            self.epsilon_a * math.sin(gamma) ** 2
            + self.epsilon_c * math.cos(gamma) ** 2
        )

    @property
    def epsilon_yy(self) -> float:
        return float(self.epsilon_a)

    @property
    def h_y(self) -> float:
        return self.epsilon_yy / self.epsilon_xx
```

`TransverseReferenceOptions` gained the explicit physical constructor:

```python
@classmethod
def from_barium_titanate_c_axis(
    cls,
    *,
    dx_normalized: float,
    dy_normalized: float,
    dt_normalized: float,
    steps: int,
    c_axis_xz_angle_deg: float,
    m_y: float = 1.0,
    applied_field_x: float = 0.0,
    epsilon_a: float = 2200.0,
    epsilon_c: float = 56.0,
) -> "TransverseReferenceOptions":
    dielectric = RotatedUniaxialDielectric(...)
    return cls(..., h_y=dielectric.h_y, dielectric=dielectric)
```

The tests add:

```text
test_rotated_barium_titanate_dielectric_limits[0°]
test_rotated_barium_titanate_dielectric_limits[45°]
test_rotated_barium_titanate_dielectric_limits[90°]
test_crystal_axis_constructor_exposes_derived_dielectric_and_preserves_default
test_rotated_dielectric_changes_localized_transverse_electrostatics
```

The comparison script now runs the isotropic and 45-degree cases with
identical inputs, records both fields, and uses shared per-field color limits.

The four reference artifacts were already untracked from the preceding
reference-solver milestone, so Git had no committed baseline for an
incremental tracked-file diff. No-index whitespace checks passed for the
complete resulting files.

## Angle convention and formulas

$$
\hat{\mathbf c}=(\cos\gamma,0,\sin\gamma).
$$

- $\gamma=0^\circ$: c-axis along simulation x.
- $\gamma=90^\circ$: c-axis along propagation z.

The implemented tensor is

$$
\boldsymbol{\varepsilon}
=\varepsilon_a\mathbf I+
(\varepsilon_c-\varepsilon_a)\hat{\mathbf c}\hat{\mathbf c}^{T},
$$

giving

$$
\varepsilon_{xx}
=\varepsilon_a\sin^2\gamma+\varepsilon_c\cos^2\gamma,
\qquad
\varepsilon_{yy}=\varepsilon_a,
$$

and

$$
h_y(\gamma)=
\frac{\varepsilon_a}
{\varepsilon_a\sin^2\gamma+\varepsilon_c\cos^2\gamma}.
$$

Reference values:

$$
\varepsilon_a=2200,
\qquad
\varepsilon_c=56.
$$

| Angle | $\varepsilon_{xx}$ | $\varepsilon_{yy}$ | $h_y$ |
| --- | ---: | ---: | ---: |
| $0^\circ$ | 56 | 2200 | 39.2857142857 |
| $45^\circ$ | 1128 | 2200 | 1.95035460993 |
| $90^\circ$ | 2200 | 2200 | 1 |

## Localized comparison

The same 64 by 56 intensity profile was evolved with $dt=0.002$ for 100
steps:

| Quantity | Isotropic | BaTiO₃ $45^\circ$ |
| --- | ---: | ---: |
| RMS $E_x$ | 0.005784893801 | 0.004548323925 |
| Maximum $\lvert E_x\rvert$ | 0.04285247526 | 0.03216609489 |
| RMS $E_y$ | 0.009912676998 | 0.005646646440 |
| Maximum $\lvert E_y\rvert$ | 0.07389087290 | 0.04285239892 |
| RMS $P-1$ | 0.009507603001 | 0.009742440973 |

Relative differences:

- $E_x$: 0.2798629641;
- $E_y$: 0.4386519584.

Comparison outputs are under
`/private/tmp/lcprop-pr-transverse-dielectric-reference/`.

## Validation

```text
python -m pytest -q tests/test_pr_transverse_reference.py
12 passed in 0.45s
```

Complete PR validation:

```text
194 passed, 9 skipped, 1 deselected in 35.60s
```

Saved 3720 micrometer snapshot:

```text
1 passed in 5.88s
```

Effective total: **195 passed, 9 skipped**.

## Existing behavior

The default remains:

```python
TransverseReferenceOptions(..., h_y=1.0)
```

It remains isotropic and numerically unchanged. Direct `h_y` specification is
retained as a low-level reference/testing override. Transport anisotropy,
optical projection $E_{\rm active}=E_x$, integration, boundaries, and
production PR behavior were not changed.

## Deliberate limitations

The implementation intentionally remains a rotated, clamped dielectric
reference, not a complete $\mathbf k$-dependent photorefractive effective
dielectric model. No electromechanical response, transport calibration, full
Pockels contraction, GPU support, or production integration was added.

No commit or push was performed.

---

# Appendix B — File inventory

| File | Role |
| --- | --- |
| `src/lcprop/pr/transverse_reference.py` | Defines the CPU transverse potential model, rotated dielectric value object, physical constructor, and derived electrostatic fields. |
| `tests/test_pr_transverse_reference.py` | Preserves the original physics gates and validates the three angle limits, default compatibility, and localized anisotropic response. |
| `scripts/checks/pr_transverse_reference.py` | Runs the deterministic isotropic/45-degree comparison and writes compact fields, a shared-scale figure, and JSON metrics. |
| `docs/architecture/pr_transverse_reference_model.md` | Documents the equations, angle convention, normalization, approximations, validation values, and scope boundary. |

# Appendix C — Validation outputs

The generated comparison directory is

```text
/private/tmp/lcprop-pr-transverse-dielectric-reference/
```

It contains exactly:

```text
metrics.json
transverse_reference_fields.npz
transverse_reference_fields.png
```

The compact contents of `metrics.json` are:

| Group | Metric | Value |
| --- | --- | ---: |
| Model | authoritative state | `psi` |
| Grid | $N_x$ | 64 |
| Grid | $N_y$ | 56 |
| Grid | $dx$ | 0.5 |
| Grid | $dy$ | 0.5 |
| Solver | integrator | `explicit_euler` |
| Solver | $dt$ | 0.002 |
| Solver | steps | 100 |
| Solver | final normalized time | 0.2 |
| Isotropic tensor | $m_y$ | 1.0 |
| Isotropic tensor | $h_y$ | 1.0 |
| Rotated tensor | c-axis angle | 45.0 degrees |
| Rotated tensor | $\varepsilon_a$ | 2200.0 |
| Rotated tensor | $\varepsilon_c$ | 56.0 |
| Rotated tensor | $\varepsilon_{xx}$ | 1127.9999999999998 |
| Rotated tensor | $\varepsilon_{yy}$ | 2200.0 |
| Rotated tensor | $h_y$ | 1.9503546099290785 |
| Rotated tensor | $m_y$ | 1.0 |
| Applied field | x component | 0.0 |
| Applied field | y component | 0.0 |
| Active projection | x weight | 1.0 |
| Active projection | y weight | 0.0 |
| Isotropic | RMS $E_x$ | 0.005784893800870564 |
| Isotropic | maximum $\lvert E_x\rvert$ | 0.04285247526208585 |
| Isotropic | RMS $E_y$ | 0.009912676998390971 |
| Isotropic | maximum $\lvert E_y\rvert$ | 0.07389087289865454 |
| Isotropic | RMS $P-1$ | 0.009507603001085742 |
| Isotropic | relative carrier drift | $1.2688263138573217\times10^{-16}$ |
| Anisotropic | RMS $E_x$ | 0.004548323925454413 |
| Anisotropic | maximum $\lvert E_x\rvert$ | 0.032166094893785836 |
| Anisotropic | RMS $E_y$ | 0.005646646439823533 |
| Anisotropic | maximum $\lvert E_y\rvert$ | 0.04285239891644005 |
| Anisotropic | RMS $P-1$ | 0.009742440973425737 |
| Anisotropic | relative carrier drift | 0.0 |
| Relative difference | $E_x$ | 0.27986296408469163 |
| Relative difference | $E_y$ | 0.43865195842039445 |

The JSON also records the output filenames
`transverse_reference_fields.png` and `transverse_reference_fields.npz`.

# Appendix D — Normalized-carrier notation semantic audit

The normalized mobile-carrier notation was changed from $n$ to
$P=p/p_d$, with $P=1$ defined as the uniform neutral reference state. The
cleanup was semantic rather than global: refractive-index notation and the
lattice-site index in the recovered discrete hopping equation were preserved.

Counts below are mathematical symbol occurrences in the pre-edit baseline of
the six audited source, test, script, architecture, and archival-report files.
They exclude ordinary letters in words, Python identifier substrings, the JSON
newline escape, and the retrospective notation labels in this appendix. The
53 changed occurrences do not include four new explanatory $P$ occurrences
added while defining $P=p/p_d$ and $P=1$ at first use in the reports.

| Category | Count | Action |
| --- | ---: | --- |
| Normalized carrier $n\rightarrow P$ | 53 | Changed |
| Refractive-index $n$ | 3 | Preserved |
| Other $n$ usages | 14 | Preserved |

Compatibility-sensitive implementation names were already semantic rather
than symbolic: `carrier_density`, `carrier_perturbation`, and
`carrier_perturbation_rms`. They were preserved, as were the existing NPZ and
JSON keys. The cleanup changes mathematical docstrings, report equations,
validation-table labels, and plot titles without changing a public API or a
machine-readable output schema. The test module required no notation edit
because it already used descriptive carrier names.

The three refractive-index occurrences are $\Delta n_{\rm eff}$,
$n_{\rm eff}$, and $n_{\rm NL}$. The 14 other occurrences are the lattice-site
index $n$ in Equation A1 and its immediate explanation. No active normalized-
carrier $n$ remains in the audited mathematical model, implementation
docstrings, plot labels, or validation tables; references to the old symbol in
this appendix exist only to record the completed change.
