# Future Research Package: Beyond Paraxial Forward-Marching Photorefractive Fanning

**Status:** Research concept / near-future extension --- deliberately
separate from completion of the conventional LCProp product\
**Date:** 2026-10-06\
**Purpose:** Preserve the research insight before implementation choices
harden around the present forward-marching model.

## Repository terminology clarification

Current LCProp PR optical propagation already uses the full scalar homogeneous
angular-spectrum kernel, not the Fresnel/paraxial diffraction approximation.
See the [committed migration contract](../../development/lcprop_pr_full_angular_spectrum_migration.md)
and [current PR model contracts](../../science/pr_model_contracts.md).
Its positive-k_z, one-way optical march and transverse material representation
remain distinct limitations; full scalar dispersion alone does not supply
backward coupling or longitudinal material transport. References below to
paraxial models describe the historical literature or small-angle overlap,
not the current Product's homogeneous propagation kernel.

This preservation pass checks repository terminology only. The literature
claims and proposed hypotheses below remain research notes, not newly verified
scientific conclusions or implementation authorization.

## 1. Core insight

A conventional forward-marching photorefractive beam-propagation model
has a structural large-angle limitation that cannot necessarily be cured
merely by increasing transverse resolution.

As photorefractive fanning grows toward large propagation angles, the
optical field and the induced charge/index gratings acquire increasingly
important longitudinal spatial structure. In the limiting case of
radiation approaching transverse propagation (near 90° to the nominal
propagation axis), the longitudinal component of the optical wavevector
approaches zero for the scattered wave, while interference with the
original beam can generate material gratings with substantial
longitudinal wavevector content. Backward radiation introduces
reflection gratings with still larger longitudinal spatial frequencies.

Therefore a model whose optical evolution is intrinsically a one-way
march in `z`, and whose material response is represented only as
transverse fields attached to coarse longitudinal optical slices, may
cease to represent the relevant physics before the angular spectrum
reaches the most interesting sideways/backward regime.

**The research question is consequently not just "how small must dz be?"
It is "when does z cease to be a valid evolution coordinate for the
coupled optical-material problem?"**

That is the key insight to preserve.

## 2. Why this is not already solved by the classic fanning models

### Zozulya--Saffman--Anderson (1994)

The 1994 PRL performs a two-dimensional optical propagation calculation
and reproduces striking photorefractive fanning, self-bending, and
phase-conjugate geometries. But its material reduction assumes that
material quantities predominantly vary in the transverse direction, and
its optical equations are explicitly paraxial/small-angle equations.

Its successful fanning calculation therefore demonstrates how much
physics can be captured inside the paraxial regime; it does **not**
establish a formulation valid continuously through sideways or backward
propagation.

The numerical fanning example is nevertheless important for LCProp
validation: it used a 5 mm longitudinal region, 2 mm transverse region,
12,000 transverse points, and 3,000 longitudinal steps, with a noisy
Gaussian input and strong nonlinearity. This is a valuable historical
benchmark for the conventional product.

### Saffman--Zozulya--Anderson (1994 JOSA B)

This work treats transverse instability of counterpropagating waves
through coupled pump/sideband amplitudes and explicitly includes
reflection gratings. It therefore exposes physics that a purely forward
field cannot represent: forward/backward coupling and gratings with
approximately twice the optical longitudinal wavevector.

But it is a coupled-mode/linear-instability formulation, not an
arbitrary-angle real-space propagation algorithm.

### Xie et al. (1998)

Xie and coworkers take a particularly revealing alternative route. They
use separate forward and backward angular fields in a two-dimensional
x-z geometry and distinguish:

-   transmission gratings coupling forward-forward and backward-backward
    angular components;
-   reflection gratings coupling forward-backward components.

The material response depends on the actual grating-vector magnitude.
Their formulation therefore handles large-angle and backward channels by
changing the representation of the optical problem rather than asking
one forward-marching field to turn through 90°.

They also show that transmission and reflection gratings can have
dramatically different response times; in their BaTiO3:Ce example a
representative reflection grating has a response time around `0.01 tau`,
versus about `0.76 tau` for a transmission grating.

This is a major clue for a future LCProp extension: **large-angle
fanning may require an angular-sector or bidirectional optical
representation coupled to grating-vector-aware material dynamics.**

## 3. Proposed research hypothesis

> Strong photorefractive fanning may have a representation transition.
> LCProp already uses full scalar homogeneous angular-spectrum dispersion,
> but its positive-k_z, one-way z march and transverse material fields on
> longitudinal slices may be adequate only while backward coupling and
> unresolved longitudinal material structure are negligible. Sufficiently
> strong fanning may require a bidirectional or full-wavevector formulation
> with explicit longitudinal material structure and transmission/reflection
> gratings. This would extend the coupled model, not merely replace a
> paraxial homogeneous propagation kernel.

This hypothesis is testable.

A second, stronger possibility is:

> The apparent saturation, angular cutoff, or morphology of fanning
> predicted by a conventional forward-marching calculation may in some
> regimes be numerical/model-formulation limited rather than physical.

That claim must **not** be assumed. It should be tested by controlled
convergence and cross-formulation experiments.

## 4. Two future research tracks

### Track A --- 2-D x-z large-angle / bidirectional model

This should probably come first.

Goal: create a scientifically transparent bridge between the present
LCProp forward-marching product and arbitrary-angle propagation in the plane
containing the dominant photorefractive coupling.

This proposed x-z model is only two-dimensional. It should be computationally
inexpensive enough to support extensive spatial, angular, and material-time
convergence matrices and, where useful, an independent 2-D
bidirectional/Helmholtz oracle. High throughput is part of the scientific
validation strategy: it should make systematic convergence and independent
cross-formulation comparisons practical. This is a research design objective,
not a measured feasibility claim or a Product performance requirement.

Candidate formulations to study:

1.  forward/backward angular-spectrum sectors, following the conceptual
    organization of Xie et al.;
2.  the existing full scalar angular-spectrum, positive-k_z propagation
    as a unidirectional control, separating homogeneous dispersion from
    one-way coupling and material-slice approximations;
3.  bidirectional Helmholtz-type propagation;
4.  coupled angular bins/rays with explicit transmission and reflection
    grating variables;
5.  a full x-z electrostatic/material solve whose charge/potential
    fields are no longer assumed transverse-only.

Critical requirements:

-   preserve the physical grating wavevector, including its z component;
-   allow forward, near-transverse, and backward optical sectors;
-   avoid a singularity or artificial loss of accuracy as `k_z -> 0`;
-   distinguish transmission from reflection gratings;
-   make material response time depend on the physically appropriate
    grating scale when the underlying transport model requires it;
-   conserve/diagnose optical power appropriately across angular
    sectors;
-   provide convergence tests in angle, x, and z;
-   recover the existing LCProp solution in the small-angle limit.

#### Candidate: material-time evolution with a global isotropic optical solve

Investigate material time as the only evolution variable, without choosing
x or z as an optical marching coordinate. At each accepted material-time
state:

1. hold the material distribution fixed;
2. solve the steady monochromatic optical problem globally over the full
   x-z plane;
3. derive optical intensity, grating information, and material drive from
   that solution;
4. advance the PR material state by one material-time step;
5. repeat.

This candidate exploits the separation between optical transit time and PR
response time. Forward, grazing, sideways, and backward radiation would be
directions within one global optical solution rather than separate spatial
marching algorithms. The validity of that time-scale separation remains part
of the model assessment. This is one candidate alongside the angular-sector
and coupled-wave candidates above, not a selected implementation architecture.

For the 2-D Track-A problem, investigate the outgoing Helmholtz Green function

\[
G_{2D}(r)=\frac{i}{4}H_0^{(1)}(k_0r),
\]

with the eventual 3-D analogue

\[
G_{3D}(r)=\frac{e^{ik_0r}}{4\pi r}.
\]

Here \(k_0\) denotes the homogeneous background optical wavenumber for this
candidate; its mapping to Product conventions must be specified in a future
derivation. Track A thus uses an isotropic outgoing cylindrical-wave kernel;
the eventual 3-D problem has spherical-wave propagation. Outgoing labels here
assume a compatible time-harmonic convention, to be made explicit below.

A Lippmann–Schwinger-type candidate is

\[
E=E_{\rm inc}+G*(VE),
\]

where \(E\) denotes the monochromatic **optical** field and \(V\) the optical
scattering potential, with sign and normalization to be derived from the
chosen Helmholtz convention. Strong multiple scattering and PR feedback
generally require a self-consistent solution, not a single Born application.
Self-consistency of the fixed-material optical solve must be distinguished
from the subsequent material-time update.

**Real-space singularity.** The Green function is singular at its origin:
logarithmically in 2-D and as \(1/r\) in 3-D. These singularities are locally
integrable and must not be implemented by evaluating \(G(0)\) pointwise.
Investigate finite-cell/self-term integration or an equivalent controlled
diagonal correction, for example the 2-D cell average

\[
G_{\rm self}=\frac{1}{\Delta A}\int_{\rm cell}G(\mathbf r)\,dA,
\]

with the appropriate cell-area weight in the discrete convolution and explicit
convergence tests. Preserve the central distinction:

- **Green-function singularity:** a controllable operator/discretization
  problem, requiring a justified self-term treatment;
- **\(k_z\to0\) in a one-way z march:** potentially a coordinate/representation
  problem, not simply a local kernel value needing regularization.

This distinction may be a central reason to investigate material-time
marching with a global optical solve; it does not itself establish that the
candidate is accurate or practical.

**Fourier-space pole and radiation condition.** Record the Fourier
representation schematically as

\[
\widetilde G(k_x,k_z)\propto
\frac{1}{k_0^2-k_x^2-k_z^2+i0},
\qquad +i0\equiv\lim_{\epsilon\to0^+}+i\epsilon.
\]

The denominator vanishes on \(k_x^2+k_z^2=k_0^2\). The infinitesimal
displacement is a radiation/pole prescription, not an arbitrary numerical
stabilizer. It specifies how the Fourier inversion contour passes the pole.
Cauchy's theorem/residue calculus, together with the adopted Fourier and
time-harmonic sign conventions, determines whether the solution is outgoing
or incoming. For the displayed plus prescription, the distribution identity is

\[
\frac{1}{x+i0}=\operatorname{PV}\frac{1}{x}-i\pi\delta(x).
\]

The delta-term sign reverses for the minus prescription. The sign of \(i0\),
the overall Green-function normalization, and the choice of \(H_0^{(1)}\)
versus \(H_0^{(2)}\) must be derived consistently from the adopted conventions;
the schematic denominator is not a convention-independent implementation rule.

An implementation must **not** simply substitute an arbitrarily small
floating-point epsilon for \(i0\). Finite epsilon introduces damping and
changes the problem. The discrete solver must demonstrate convergence to the
appropriate outgoing-radiation, zero-loss limit, accounting for domain and
resolution as well as damping.

**Future Track-A validation.** Include:

- small-angle parity with conventional LCProp, whose homogeneous optical
  kernel already uses full scalar angular-spectrum dispersion;
- behavior approaching grazing and backward propagation;
- self-cell singularity convergence;
- spatial, material-time, and iterative-solver convergence;
- equivalence with an independent 2-D Helmholtz/PML oracle;
- convergence as finite damping/regularization tends to the outgoing
  zero-loss limit;
- conditioning and scientifically neutral preconditioning of the
  Lippmann–Schwinger solve under strong PR feedback;
- whether transmission/reflection-grating physics emerges naturally or
  requires explicit grating variables.

Because Track A is only two-dimensional, FFT-accelerated global optical
operators may be extremely fast and permit extensive convergence matrices
and independent-oracle comparisons. This is a **hypothesis to benchmark**,
not a performance claim or Product performance requirement. High throughput
is a proposed means of making scientific validation thorough. All existing
implementation prohibitions remain in force.

### Track B --- true 3-D arbitrary-angle model

Goal: ultimately allow a fan to develop on the full wavevector sphere
rather than in a prescribed plane.

This is not simply "LCProp with Ny added." It requires deciding how
optical direction, 3-D material charge/potential structure, and
electro-optic tensor projection are represented when there is no
privileged propagation direction.

Candidate architectures include:

-   multi-sector angular-spectrum propagation;
-   directional domain decomposition;
-   bidirectional/full Helmholtz solvers;
-   hybrid real-space/material + k-space/optical representations;
-   eventually, if justified, a full-wave or envelope formulation with
    no single global propagation axis.

The 3-D model should be attempted only after the 2-D model identifies
which physical and numerical ingredients are actually necessary.

## 5. Immediate experiments that do NOT require the future solver

Before changing the architecture, the conventional LCProp product can
map its own domain of validity.

Recommended studies:

-   repeat fanning calculations while reducing `dz`;
-   compare far-field angular distributions versus `dz`;
-   track off-carrier fraction versus time and `dz`;
-   quantify population approaching grazing propagation, where a one-way
    z evolution coordinate and coarse longitudinal material slices may
    become inadequate despite full scalar homogeneous dispersion;
-   distinguish angular support/resolution from missing backward coupling
    and unresolved longitudinal charge/index-grating structure;
-   compare square versus anisotropic transverse grids;
-   compare the present full-transverse material model with the
    legacy/reduced model only where both are numerically trustworthy;
-   reproduce a Zozulya-style small-angle fanning benchmark as a
    historical validation case;
-   establish a "forward-marching/material-model validity envelope"
    rather than silently extrapolating the coupled model.

These studies belong to validation of the conventional product and can
later provide the overlap data needed to validate the new formulation.

## 6. Design principle for LCProp

Do **not** distort the nearly finished conventional product to solve
this future problem.

The present product should be finished as a clearly defined
forward-marching scalar angular-spectrum photorefractive propagation tool with explicit validity
diagnostics.

The large-angle/bidirectional project should be a peer research workflow
or future material/optics architecture, sharing trusted infrastructure
where appropriate but not forcing arbitrary-angle physics through APIs
designed for a one-way z march.

## 7. Scientific milestones

**R0 --- Literature reconstruction.** Re-derive exactly what the
Zozulya, Saffman, Xie, Valley, Segev, and related models assume about
optical direction and material dimensionality.

**R1 --- Validity map of current LCProp.** Determine numerically where
results become sensitive to longitudinal step and angular support.

**R2 --- 2-D linear benchmark.** Implement a minimal x-z
bidirectional/large-angle linear scattering model with no PR feedback.

**R3 --- Prescribed-grating benchmark.** Propagate through known
transmission and reflection gratings and verify angular coupling.

**R4 --- Dynamic PR grating model.** Couple optical angular sectors to
time-dependent photorefractive material response.

**R5 --- Small-angle parity.** Demonstrate convergence to current LCProp
in the overlap regime.

**R6 --- Sideways-fanning experiment.** Ask whether an initially forward
beam can populate and amplify near-transverse optical states without a
coordinate singularity or imposed angular cutoff.

**R7 --- Backward-fanning benchmark.** Compare qualitatively and
quantitatively with Xie-style backward-fanning results.

**R8 --- 3-D design review.** Only after R0--R7 decide whether full 3-D
implementation is scientifically justified and what representation it
should use.

## 8. Falsifiable outcomes

This project remains worthwhile even if the dramatic hypothesis is
wrong.

Possible outcomes include:

-   the conventional forward model remains quantitatively valid
    throughout experimentally relevant forward fanning;
-   the existing full scalar angular-spectrum kernel remains adequate,
    with only improved longitudinal coupling or material resolution needed;
-   longitudinal material resolution changes the fan substantially
    before optical propagation itself fails;
-   explicit backward/reflection-grating channels are essential only for
    a separate class of experiments;
-   or the full arbitrary-angle treatment reveals qualitatively new
    evolution toward sideways/backward propagation.

Each outcome is scientifically informative.

## 9. Priority relative to current work

**Do not begin implementation now.**

First finish and commission the conventional PR product, including
continuation, diagnostics, movies, stable full-transverse TD execution,
and clearly stated validity limits.

Preserve this document as the research seed for the next-generation
fanning project.

------------------------------------------------------------------------

## Source papers that motivated this package

1.  A. A. Zozulya, M. Saffman, and D. Z. Anderson, "Propagation of Light
    Beams in Photorefractive Media: Fanning, Self-Bending, and Formation
    of Self-Pumped Four-Wave-Mixing Phase Conjugation Geometries,"
    *Physical Review Letters* **73**, 818 (1994).
2.  M. Saffman, A. A. Zozulya, and D. Z. Anderson, "Transverse
    instability of energy-exchanging counterpropagating waves in
    photorefractive media," *Journal of the Optical Society of America
    B* **11**, 1409 (1994).
3.  P. Xie, P.-Y. Wang, J.-H. Dai, and H.-J. Zhang, "Backward beam
    fanning in photorefractive crystals," *Journal of the Optical
    Society of America B* **15**, 1521 (1998).

## Repository destination

`docs/research/future/pr-large-angle-bidirectional-fanning-research-seed.md`

This is a research note, not an accepted Product specification or
implementation authorization.

## Short Codex preservation prompt

Review the attached research-seed Markdown against the current LCProp
repository architecture. Do not implement physics or modify Product
behavior. Preserve the note under an appropriate future-research
documentation path, correcting only repository-specific naming/path
issues. Add cross-links only if an existing future-work/research index
clearly warrants them. Report exact files changed and diff summary. Run
documentation-only checks if available. Do not simulate, use CUDA/Slurm,
push, or alter protected research evidence.
