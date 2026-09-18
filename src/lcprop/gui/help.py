"""Small material-neutral in-application help surface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QMenu,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class HelpTopic:
    key: str
    title: str
    markdown: str


HELP_TOPICS = (
    HelpTopic(
        "quick_start", "Quick Start",
        """Choose **LC** for liquid-crystal director response or **PR** for photorefractive
charge transport. Define beams in **Beam**, choose a modest grid, run locally,
and inspect Results. Save the experiment before large runs.
Use modest settings for quick exploration. Verify grid, step-size, timestep,
precision, and model convergence as applicable before quantitative conclusions.
Open the [rendered Quick Start](https://github.com/mcroning/LCProp/blob/feature/pr-second-order-static/docs/user/quick_start.md)
for complete LCProp and LaunchPlane installation instructions. Online guides
follow the published branch and can lag local development; built-in Help
ships with this installation.
""",
    ),
    HelpTopic(
        "model_choices", "Model choices",
        """LC offers static propagation, time evolution, stationary solitons and sweeps.
PR offers independent choices of Static/Time dependent, Reduced x-only/Full
transverse transport, and Fully nonlinear/Linearized response. Choose the model
whose assumptions match your experiment; linearized response requires a positive
reference intensity. Current support is a Product capability, not evidence that
a particular grid or parameter choice is converged.
""",
    ),
    HelpTopic(
        "beam_focusing", "Beam focusing",
        """Every Gaussian beam has a waist. **Collimated Gaussian** specifies the entrance
waist and no focusing curvature there; it does not mean an infinite beam with
no waist. **Focused Gaussian** specifies the waist at a signed focus position:
zero is the interaction entrance, positive is downstream, negative upstream.
LCProp derives entrance size and curvature from that waist, focus, wavelength
and medium; x/y waists can differ. Use the focus position appropriate to the
sample. These remain separate Product profile choices.
""",
    ),
    HelpTopic(
        "boundary_conditions", "Optical edge treatment",
        """FFT propagation uses a **periodic computational domain** in every case.
Choose **None** for no edge attenuation/apodization, **Sponge** for smooth
attenuation accumulated with propagation distance, or **Tukey** for the existing
discrete apodization window. Tukey is not a per-distance absorption rate.
Sponge accumulation is invariant to optical substep subdivision. Edge treatment
reduces unwanted wraparound but does not replace aperture/grid convergence.
LC stationary soliton and existence workflows require **None** (the persisted
`periodic` choice); their restriction is unchanged.
""",
    ),
    HelpTopic(
        "fanning_scattering", "Fanning and scattering setup",
        """### Set up the current Product model

Scattering introduces weak perturbations into supported PR calculations.
Configure the controls deliberately and save the request:

- **Strength**: perturbation magnitude; zero disables it.
- **Transverse correlation length**: transverse texture scale.
- **Random seed**: reproducible realization.
- **Longitudinal canonical slab spacing**: longitudinal partition defining the realization.

Record these with focus, edge treatment, material-z sampling and optical
substeps. They are model/setup parameters, not universal material constants.

### Interpret with care

PR canonical scattering equivalence to the legacy/Photonics model remains on
**SCIENTIFIC HOLD / unestablished**. A visible or absent fanning pattern does
not establish that equivalence. No change of defaults is implied.

### Current support

Full-transverse static and supported TD workflows support canonical scattering.
Reduced static currently does not; this is a Product capability limitation,
not a physical prohibition. See the User Guide for detailed setup and sampling
limitations.
""",
    ),
    HelpTopic(
        "slurm", "Running on Slurm",
        """### Choose where to run

**Local** runs on this computer. **Slurm** submits to a configured cluster.

1. Open **Configure Remote Execution** and choose or create a cluster/resource.
2. Select **Slurm**, then inspect the configured execution before running.
3. Choose the scientific backend appropriate to that resource, then **Run**.

Profiles are user-local configuration, separate from the installed package;
the configuration dialog shows their location.

### CPU or GPU computation

NumPy uses the CPU. CuPy needs a supported GPU environment. Allocating a GPU
does not make NumPy use it; explicit backend choices are preserved.

### Installation and current limits

Ordinary non-editable installation supports Local execution. Installed LCProp
can also use Slurm when its source deployment is configured correctly:
automatic deployment requires a suitable clean deployable LCProp Git checkout.
The User Guide explains supplying that checkout or selecting a pinned existing
remote source. PR Continue currently runs **Local only**.
""",
    ),
    HelpTopic(
        "runtime_estimates", "Runtime estimates",
        """Use **Run Planning → Configured execution** to check the selected run target.
Local Mac/NumPy and H200/CuPy numbers are comparison estimates, not selections.
Estimates are advisory ranges; they do not change the backend, tolerances,
grid, retention or target. Full-transverse nonlinear static cost is especially
sensitive to continuation and convergence.
""",
    ),
    HelpTopic(
        "fast_full", "Fast vs Full",
        """PR Slurm **Fast** retrieval keeps optical endpoints, compact diagnostics,
exact full-resolution longitudinal cuts nearest x=0 and y=0, and a bounded
visualization-only preview for linked orthogonal slices. **Full** retains the
supported complete scientific volumes. Fast TD retains the final 3-D preview
and a compact material-time movie, not a full time history of 3-D volumes.
Choose Full when you need the supported volume data for quantitative analysis;
Full output alone is not a remote-continuation contract.
""",
    ),
    HelpTopic(
        "results", "Understanding Results",
        """### Find the result you need

- **Fields**: images and linked x-y, x-z and y-z slices.
- **Curves**: quantitative traces.
- **Samples / Tables**: retained member values and convergence gates; an empty
  explanation means this result has no retained tables.
- **Diagnostics**: numerical and carrier-power details.
- **Request**: requested configuration, internal request identity and displayed-result ownership.
- **Console**: timestamped operation boundaries, progress and failures.

### Display scaling

**Auto each frame** follows the current view. **Lock scale across frames** keeps
a common range. **Manual limits** applies your Minimum and Maximum. These change
only presentation, never scientific values. Exact numeric values remain available
in table tooltips; meaningful near-unity overlap precision is preserved.

### Two-beam interpretation

Brightness alone is not quantitative energy transfer. Inspect each carrier's
input and output powers and available gain where spatial-frequency separation is
sufficient. Unavailable-gain reasons explain when a reliable ratio cannot be
reported. General two-beam coupling does not require the screen or symmetry
conditions of specialized **Image Amplification**.

### Execution and convergence

Execution completion and solver convergence are different. The member summary
identifies termination, iteration budget and failed gates; retain the detailed
Convergence gates table for expert inspection. Nonconvergence does not establish
physical nonexistence or instability. Continuation initializes from a previously
converged member for branch following; it does not waive a convergence criterion.
See the User Guide for interpretation and workflow limits.
""",
    ),
)

_TOPICS_BY_KEY = {topic.key: topic for topic in HELP_TOPICS}


class ProductHelpDialog(QDialog):
    """Non-modal renderer for one concise built-in help topic."""

    def __init__(
        self,
        topic: HelpTopic,
        *,
        application: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if application not in {"lc", "pr"}:
            raise ValueError("application must be 'lc' or 'pr'")
        self.topic = topic
        self.application = application
        self.setWindowTitle(f"LCProp Help — {topic.title}")
        self.resize(620, 420)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        material_name = "Liquid Crystal" if application == "lc" else "Photorefractive"
        self.browser.setMarkdown(
            f"# {topic.title}\n\n**Current application**: {material_name}\n\n"
            + topic.markdown
            + ("\n\nCurrent LC Slurm capability: canonical NumPy static propagation on a CPU resource; other LC workflows run locally."
               if application == "lc" and topic.key == "slurm" else "")
            + "\n\n[Read the rendered User Guide](https://github.com/mcroning/LCProp/blob/feature/pr-second-order-static/docs/user/user_guide.md)"
            + " · [Documentation index](https://github.com/mcroning/LCProp/blob/feature/pr-second-order-static/docs/README.md)"
        )
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)


class ProductHelpButton(QToolButton):
    """Shared Help menu whose actions never mutate the scientific request."""

    def __init__(
        self,
        *,
        application: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if application not in {"lc", "pr"}:
            raise ValueError("application must be 'lc' or 'pr'")
        self.application = application
        self.setText("Help")
        self.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        for topic in HELP_TOPICS:
            action = menu.addAction(topic.title)
            action.setData(topic.key)
            action.triggered.connect(partial(self.open_topic, topic.key))
        self.setMenu(menu)
        self._active_dialog: ProductHelpDialog | None = None

    @property
    def topic_keys(self) -> tuple[str, ...]:
        return tuple(topic.key for topic in HELP_TOPICS)

    def open_topic(self, key: str, _checked: bool = False) -> ProductHelpDialog:
        try:
            topic = _TOPICS_BY_KEY[key]
        except KeyError as exc:
            raise ValueError(f"unknown help topic: {key!r}") from exc
        if self._active_dialog is not None:
            self._active_dialog.close()
        dialog = ProductHelpDialog(
            topic,
            application=self.application,
            parent=self,
        )
        self._active_dialog = dialog
        dialog.finished.connect(self._dialog_finished)
        dialog.open()
        return dialog

    def _dialog_finished(self, _result: int) -> None:
        self._active_dialog = None


__all__ = [
    "HELP_TOPICS",
    "HelpTopic",
    "ProductHelpButton",
    "ProductHelpDialog",
]
