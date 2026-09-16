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
        "quick_start",
        "Quick Start",
        """Choose **LC** for director propagation or **PR** for photorefractive
transport. Define the beam in **Beam**, choose a modest grid, run locally, and
inspect **Fields**, **Curves**, and **Diagnostics**. Save the experiment before
large runs. The complete walkthrough is `docs/user/quick_start.md`.
""",
    ),
    HelpTopic(
        "model_choices",
        "Model choices",
        """LC offers static, time-dependent, soliton, and soliton-sweep
workflows. PR model choice has three independent axes: evolution, transverse
transport, and material response. All eight PR combinations are production
choices; validation and hardware commissioning are separate evidence. See
`docs/user/user_guide.md` and `docs/science/pr_model_contracts.md`.
""",
    ),
    HelpTopic(
        "beam_focusing",
        "Beam focusing",
        """**Collimated Gaussian** specifies the entrance-plane waist.
**Focused Gaussian** specifies the waist at its focus and a signed focus
position measured from the interaction entrance plane; a negative position is
upstream. LaunchPlane expresses beam intent while LCProp resolves it on the
selected grid.
""",
    ),
    HelpTopic(
        "boundary_conditions",
        "Boundary conditions",
        """**Periodic** applies no optical absorber. **Sponge** is a smooth
amplitude-absorption rate accumulated with propagation distance and is
invariant to subdivision into optical substeps. **Tukey** is a discrete
apodization window, not a rate. Boundaries do not replace aperture and grid
convergence checks.
""",
    ),
    HelpTopic(
        "fanning_scattering",
        "Fanning and scattering setup",
        """PR fanning studies require an explicit scattering model, strength,
correlation length, seed, and canonical slab spacing. Record the beam/focus,
boundary treatment, material-z sampling, and optical substeps independently.
Reduced static requests do not carry the canonical scattering specification.
""",
    ),
    HelpTopic(
        "slurm",
        "Running on Slurm",
        """Execution target and scientific backend are distinct. Configure and
test a remote profile, select Slurm, then review the immutable request summary.
An automatically chosen GPU backend is reversible; an explicit or
experiment-loaded backend is preserved. LC Slurm support is currently limited
to canonical static propagation.
""",
    ),
    HelpTopic(
        "runtime_estimates",
        "Runtime estimates",
        """The PR **Run Planning** estimate is advisory. It reports ranges and
confidence from committed calibration plus transparent scaling. It does not
change the request, backend, tolerances, grid, retention mode, or execution
target. Full-transverse nonlinear static cost is especially continuation
sensitive.
""",
    ),
    HelpTopic(
        "fast_full",
        "Fast vs Full",
        """PR Slurm **Fast** retrieval keeps optical endpoints, compact
diagnostics, exact nearest-zero quantitative longitudinal cuts, and a bounded
visualization-only MPR preview. **Full** retains the complete supported
scientific volumes. Fast TD keeps only the final 3-D preview plus a compact
material-time movie, never a 4-D preview history.
""",
    ),
    HelpTopic(
        "results",
        "Understanding Results",
        """Use **Fields** for transverse images and linked MPR slices,
**Curves** for accepted-state histories, **Diagnostics** for convergence and
carrier-power data, **Request** for the executed configuration, and **Console**
for progress. Autoscaled image brightness is not a quantitative two-beam
energy-transfer measure; use carrier-resolved Fourier-space power.
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
            f"# {topic.title}\n\n**Current application:** {material_name}\n\n"
            + topic.markdown
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
