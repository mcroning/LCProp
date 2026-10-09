# Stage 2A — hosted Colab reduced PR CPU qualification closed

**QUALIFIED for the tested reduced nonlinear PR CPU workflow.** This closure records
the recovered hosted installation metadata supplied by the user and the previously
verified saved-data comparisons. It supersedes the engine-provenance limitation in
the diagnostic report; that historical report and all original evidence remain
unchanged. No scientific calculation was rerun for closure.

## Distinct source identities

| Component | Identity |
|---|---|
| Installed hosted scientific engine | `985b80fbbc05616d4fe92f974ef0da24bf84e94c` |
| Accepted Stage 1 notebook implementation | `580a12b8933a6bfa39bd8bbd5ccff47fef5a5b28` |
| Notebook file SHA-256 | `7f8c06d8a18fa685c192deaa8a48712433e4a1cd41ea8710b4ad8a33ba17e6b4` |

The installed Git revision is recovered metadata reported by the user for this
closure; the original result archive contains only the version/baseline label, not
the installation receipt. No new receipt file was supplied or independently parsed
in this turn. Git comparison established that `src/` and `pyproject.toml` are
identical between these two revisions. The Stage 1 commit adds the notebook,
README, tests and validation report; it does not change the scientific engine.
The hosted engine is therefore not relabeled as installed from `580a12b…`.

## Hosted environment and demonstrated behavior

- Python **3.13.16**, NumPy **2.1.3**, SciPy **1.16.3**.
- **Linux x86-64**, **OpenBLAS 0.3.27**. The BLAS version is recovered metadata
  supplied by the user; it was absent from the original archive.
- Archive records Linux 6.6.122+, glibc 2.39, Matplotlib 3.10.0 and LCProp 0.1.0
  imported from `/usr/local/lib/python3.13/dist-packages/lcprop`.
- Successful NumPy float64 execution: **3/3 accepted steps**, characteristic
  material time **τ=0.003**, status **completed**. Recorded runner/product elapsed
  time was **0.9674 s**, not a performance guarantee.
- Successful xy/xz plotting and artifact export. The archive contains the rendered
  PNG, selected intensity arrays, experiment, Product request/result packages and
  checksummed notebook manifest. User closure confirms successful plotting;
  retained file/hash checks substantiate the exported artifacts.

The tested request is the unchanged small reduced nonlinear `pr_timedependent`
example: 64×48 transverse grid, 20 longitudinal planes, NumPy float64,
`semi_implicit_trapezoidal`, dt=0.001, one normal-incidence 1 mW Gaussian,
0.633 µm wavelength, gain-length product 0.1, physical dark-equivalent irradiance
0.01 W/cm², zero background, no scattering, and published optical-first coupling.

## Scientific request and numerical parity

The saved hosted experiment is byte-identical to the local reference, SHA-256
`8b5110c6df2cc07ed56431dd0b5b910c516d97b075fd0a0bcdb8b6fc830e3a60`.
Scientific request payloads, numerical settings, coordinates, accepted observation
times and saved result diagnostics match. Original local arrays were recovered
and compared directly; none were regenerated.

| Scientific quantity | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| Complex output optical field | 4.267984×10⁻¹⁶ | 4.890911×10⁻¹⁵ |
| Final material field | 2.363561×10⁻¹⁷ | 4.120882×10⁻¹⁵ |
| Transport source intensity | 2.060574×10⁻¹³ | 3.513869×10⁻¹⁵ |
| Output xy intensity | 4.250073×10⁻¹⁷ | 5.775778×10⁻¹⁵ |
| Optical xz intensity | 4.250073×10⁻¹⁷ | 5.684104×10⁻¹⁵ |

All compared scientific values are finite. Initial optical and initial material
arrays match exactly; checkpoint differences duplicate their corresponding state
differences. Four of the nine encoded-array hashes match exactly. Of the five
differences, four are scientific state/source entries and one is an encoded MP4.
The movie contains different FFmpeg/x264 version strings, so compressed movie-byte
identity is not a scientific parity requirement. Retained numerical movie samples
were compared separately in the diagnostic.

The observed scientific discrepancies are consistent with floating-point variation
between the retained macOS arm64 environment and hosted Linux x86-64 environment
with different numerical-library versions. Differences are already observable in
the retained τ=0 optical products before material evolution. The exact first
rounding operation was not isolated; OpenBLAS is not asserted to be the cause.

**This is acceptance of the observed cross-platform numerical parity for this
fixture, not a claim of bitwise equality or a newly imposed general tolerance.**
The user's qualification-closure instruction accepts these documented results.
No physical gate, integration tolerance, normalization or algorithm was changed.

## Evidence and scope

- [Preparation procedure](../lcprop-colab-cpu-stage-2-preparation-v1/REPORT.md).
- [Saved-data diagnostic and complete metrics](../lcprop-colab-cpu-stage-2-parity-diagnostic-v1/REPORT.md),
  with `comparison.json` in that directory.
- [Stage 1 validation](../lcprop-portable-notebook-stage-1-v1/REPORT.md):
  retained 14 notebook tests plus 24 existing regressions; detached completed-run
  provenance, real export paths, canonical scientific parity and dependency isolation.
- Hosted archive: `/Users/mcroning/Downloads/lcprop_results.zip`, SHA-256
  `d73c49a8dbc4debac53bb1eb3f3085195c7cb13da80629853fda22a5f8584997`.
- Both local and hosted artifact manifests, checksum files and READY bindings
  were independently verified during the diagnostic.

Qualification covers this **hosted Colab CPU reduced-PR notebook fixture and its
tested environment**. It does not qualify arbitrary grids/timesteps, full-transverse
TD, GPU/CuPy, live visualization, other Colab images, Slurm, fanning, backpropagation
or LC workflows. Exact BLAS threading and FFT build details remain unrecorded;
they are not silently inferred from the recovered BLAS version.

Only this qualification documentation is prepared for commit. Scientific source,
unrelated tracked modifications, prior reports, archives and protected evidence
are unchanged. No simulation, installation, cluster access or push was performed.

**LCPROP COLAB STAGE 2A QUALIFIED — TESTED REDUCED PR CPU WORKFLOW**
