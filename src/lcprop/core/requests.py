from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamStack


StaticStrategy = Literal[
    "fixed_theta",
    "local_self_consistent",
]

ThetaSolver = Literal[
    "none",
    "picard_cn",
]

OpticsSolver = Literal[
    "splitstep",
]

CouplingMode = Literal[
    "frozen",
    "self_consistent",
]


@dataclass(frozen=True)
class StaticWorkflowOptions:
    """
    High-level static workflow choices.

    strategy:
        fixed_theta:
            propagate through prepared theta without solving self-consistency.

        local_self_consistent:
            alternate local static theta relaxation with split-step optics.

    theta_solver:
        none:
            no theta update.

        picard_cn:
            Picard-corrected CN theta relaxation.

    optics_solver:
        splitstep:
            channel-stack split-step propagation.

    coupling:
        frozen:
            theta/intensity coupling is not iterated.

        self_consistent:
            theta and optical intensity are iterated.
    """

    strategy: StaticStrategy = "fixed_theta"
    theta_solver: ThetaSolver = "none"
    optics_solver: OpticsSolver = "splitstep"
    coupling: CouplingMode = "frozen"


@dataclass(frozen=True)
class StaticSolverOptions:
    workflow: StaticWorkflowOptions = StaticWorkflowOptions()
    max_iterations: int = 100
    tolerance_rms: Optional[float] = None
    tolerance_max: Optional[float] = None

    @property
    def method(self) -> str:
        """
        Backward-compatible label for reports/tests.

        Prefer workflow.strategy in new code.
        """
        return self.workflow.strategy


@dataclass(frozen=True)
class OutputOptions:
    run_dir: Optional[Path] = None
    save_slices: bool = True
    save_full: bool = False


@dataclass(frozen=True)
class StaticRunRequest:
    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec
    beams: BeamStack
    solver: StaticSolverOptions
    output: OutputOptions


@dataclass(frozen=True)
class TimeDependentSolverOptions:
    workflow: StaticWorkflowOptions = StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )
    Nt: int = 20
    dt: float = 7.5e-4
    gamma_z: float = 0.0
    max_picard_iter: int = 4
    tolerance_update: float = 1e-6


@dataclass(frozen=True)
class TimeDependentRunRequest:
    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec
    beams: BeamStack
    solver: TimeDependentSolverOptions
    output: OutputOptions
