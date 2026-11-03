# Contributing to LCProp

## Work independently

1. Fork the repository into your own account. Do not request upstream write access
   just to experiment. Use your own clone, branch and virtual environment.
2. Start from a stated commit. The tested reference is
   `4299ec618810f63ed39431ad1cb374f8f7e6cb93`; branch names and the `0.4.0` label
   alone do not identify the qualified tree. See the [quick start](docs/developer_quickstart.md).
3. Discuss the scope of new physics or public-interface changes before broad work.
   Keep LC and PR semantics intact; do not make a new material depend on a GUI.
4. Test the affected behavior and state the limits. Numerical changes need
   physical units, normalization, assumptions and appropriate reference/convergence
   checks. A passing import or GUI startup is not scientific validation.
5. Open a pull request against `main` once it is established as the public entry.
   State your base commit, problem, changes, checks, warnings and numerical scope.
   Do not include private cluster paths, credentials or bulk research outputs.
   Opening a pull request starts review; it does not authorize merging it.

## Owner approval and the current-account model

The maintainer is `@mcroning`. External contributions require maintainer review.
The intended protected-branch policy uses pull requests, owner review, stale
approval dismissal and separate history protections. The presence of this file
or CODEOWNERS does not itself activate GitHub rules; inspect the repository's
actual rules for enforcement status.

The present workflow also uses the maintainer's account for authorized agent work.
GitHub cannot distinguish the maintainer from an agent using those credentials,
and an author cannot approve their own pull request. An owner-authorized exception
may therefore merge an owner-authored pull request through a **pull-request-only**
review bypass. It must apply to the exact reviewed head revision and receive fresh
permission if that revision changes. No bypass is intended for history-safety rules.

This is a procedural human-approval boundary, not independently authenticated
human review. Changing to a separate automation identity requires a separate
maintainer decision. Do not add collaborators or enable automatic merging,
publication, dependency upgrades or installation changes as part of a contribution.

## Protect existing work

A pull request merge changes repository history, not the maintainer's preserved
working installation. Do not alter existing environments, launchers, cluster jobs,
result trees or editor paths. Installation or release promotion is a separate
explicit decision. Experiments remain in your fork/environment until accepted.

Never force-push, delete or repoint canonical/history refs to simplify a review.
If a policy or restore problem occurs, stop and propose a concrete corrective
plan; do not disable protections automatically.

## Extension routes

See the [extension map](docs/developer_extension_map.md) before changing physics,
products or a frontend. Small examples are preferred to a new plugin framework.
Use the [concrete test tiers](docs/developer_testing.md) to select focused checks
in your new environment before proposing a broader suite. The tiers identify
real paths, prerequisites, optional device checks, and exclusions. State skipped,
large, GPU and cluster checks explicitly rather than implying they were exercised.
