# Physical total-illumination PR normalization

`pr_integral_total_illumination_mean_irradiance_v1` is an explicit scientific
normalization identity for unified Static, reduced TD and full-transverse TD.
It does not change their material equations or integration methods.

Physical dark-equivalent and uniform-background irradiances are mandatory
inputs in W/cm². Zero is permitted; no nonzero default is supplied. The GUI
accepts mW/cm² and converts to W/cm². Derived dimensionless fractions are not
independent inputs. Programmatic callers use `PRMaterialSpec.physical(...)`.

For the accepted launch, intensities interfere within each coherence group
before groups are summed. The numerical launch's fractional-power field is
converted to physical irradiance with `total_power_mW * 1e5`; integrating a
unit-fraction field over µm² then recovers its actual power. Thus absolute beam
power is restored before constructing the transport source, and coherent cross
terms on the represented aperture are included.

On the periodic cell-centered grid, with pixel area in cm²,

```
A = Nx * Ny * dx_um * dy_um * 1e-8
Popt = sum(Iopt_W_cm2) * dx_um * dy_um * 1e-8
Pref = Popt + A * (Id + Ib)                     # W
Ibar = Pref / A                                # W/cm²
Itransport = (Iopt_W_cm2 + Id + Ib) / Ibar
id = Id / Ibar; ib = Ib / Ibar
```

The aperture mean of the reference launch source is one. Pref must be positive.
The reference remains fixed through the propagation/time segment; evolved
fields are not independently renormalized. Equal integrated power gives equal
reference power regardless of beam shape or sampled peak. Changing optical
power changes its relative contribution when fixed physical backgrounds are
nonzero. With both backgrounds zero, a common optical power scale cancels by
definition. No separate reference is assigned to independent reduced columns.

The A7 normalized current is `Eapp * (id + ib)`, derived after launch acceptance;
the solved internal harmonic field is not Eapp. Accepted result provenance
binds the actual derived closure and physical reference. Physical reference
metadata is retained for results, source inversion and future calibration.

`pr_channel_peak_reference_v1` remains the explicit legacy convention:
`sum_g |sum_j A_j|² / sum_j max|A_j(0)|² + dark_intensity + uniform_background`.
The legacy constructor and historical archives keep dimensionless backgrounds
and their old meaning. Loading them does not migrate them. Prior native
certifications and trajectory studies qualify that predecessor convention,
not this new one. The GUI shows the selected mode and its units explicitly.

Physical TD currently supports fresh beam-defined launches. Arbitrary prepared
TD fields are rejected because that interface lacks a physical amplitude-scale
contract. Prepared unified Static fields require an explicit irradiance scale.
Dark-only prepared Static is valid when total reference power is positive.
Continuation remains deferred: a future illumination intervention must preserve
the accepted material state and explicitly recompute the next segment reference.

TD time remains characteristic time. No mapping to seconds, proportional or
otherwise, is introduced. Pref, Ibar, aperture and physical illumination are
retained to support a separately validated future calibration.
