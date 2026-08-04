"""Explicit composition of material-owned checkpoint codecs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable


CheckpointSaveCallable = Callable[[Any, str | Path], Path]
CheckpointLoadCallable = Callable[[str | Path], Any]


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty, trimmed string")


@dataclass(frozen=True)
class CheckpointCodec:
    """Material/workflow identity plus callables for one checkpoint type."""

    material_id: str
    workflow_id: str
    checkpoint_type: type
    save: CheckpointSaveCallable
    load: CheckpointLoadCallable

    def __post_init__(self) -> None:
        _validate_identifier("material_id", self.material_id)
        _validate_identifier("workflow_id", self.workflow_id)
        if not isinstance(self.checkpoint_type, type):
            raise TypeError("checkpoint_type must be a type")
        if not callable(self.save):
            raise TypeError("save must be callable")
        if not callable(self.load):
            raise TypeError("load must be callable")

    @property
    def key(self) -> tuple[str, str]:
        return self.material_id, self.workflow_id


class CheckpointCodecRegistry:
    """Explicit, in-process registry for checkpoint persistence callables."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], CheckpointCodec] = {}
        self._by_type: dict[type, CheckpointCodec] = {}
        self._legacy_workflows: dict[str, tuple[str, str]] = {}

    def register(self, codec: CheckpointCodec) -> None:
        if not isinstance(codec, CheckpointCodec):
            raise TypeError("codec must be a CheckpointCodec")
        if codec.key in self._by_key:
            raise ValueError(
                "checkpoint codec already registered for "
                f"material={codec.material_id!r}, workflow={codec.workflow_id!r}"
            )
        if codec.checkpoint_type in self._by_type:
            existing = self._by_type[codec.checkpoint_type]
            raise ValueError(
                "checkpoint type already registered for "
                f"material={existing.material_id!r}, "
                f"workflow={existing.workflow_id!r}"
            )
        self._by_key[codec.key] = codec
        self._by_type[codec.checkpoint_type] = codec

    def register_legacy_workflow(
        self,
        workflow_id: str,
        *,
        material_id: str,
    ) -> None:
        """Map material-less historical metadata to one registered codec."""

        _validate_identifier("workflow_id", workflow_id)
        _validate_identifier("material_id", material_id)
        key = (material_id, workflow_id)
        if key not in self._by_key:
            raise ValueError(
                "legacy workflow target is not a registered checkpoint codec: "
                f"material={material_id!r}, workflow={workflow_id!r}"
            )
        if workflow_id in self._legacy_workflows:
            raise ValueError(
                f"legacy checkpoint workflow already registered: {workflow_id!r}"
            )
        self._legacy_workflows[workflow_id] = key

    @property
    def codecs(self) -> tuple[CheckpointCodec, ...]:
        return tuple(self._by_key.values())

    def save_checkpoint(self, checkpoint: Any, run_dir: str | Path) -> Path:
        codec = self._by_type.get(type(checkpoint))
        if codec is None:
            compatible = tuple(
                candidate
                for checkpoint_type, candidate in self._by_type.items()
                if isinstance(checkpoint, checkpoint_type)
            )
            if not compatible:
                raise TypeError(
                    f"unsupported checkpoint type: {type(checkpoint).__name__}"
                )
            if len(compatible) > 1:
                raise TypeError(
                    "ambiguous checkpoint type: "
                    f"{type(checkpoint).__name__} matches multiple codecs"
                )
            codec = compatible[0]
        return codec.save(checkpoint, run_dir)

    def load_checkpoint(self, run_dir: str | Path) -> Any:
        directory = Path(run_dir)
        provenance = json.loads(
            (directory / "provenance.json").read_text(encoding="utf-8")
        )
        workflow_id = provenance.get("workflow")
        _validate_identifier("checkpoint workflow", workflow_id)
        material_id = provenance.get("material")
        if material_id is None:
            key = self._legacy_workflows.get(workflow_id)
            if key is None:
                raise ValueError(
                    "unsupported legacy checkpoint workflow: "
                    f"{workflow_id!r}"
                )
        else:
            _validate_identifier("checkpoint material", material_id)
            key = (material_id, workflow_id)

        codec = self._by_key.get(key)
        if codec is None:
            raise ValueError(
                "unsupported checkpoint codec identity: "
                f"material={key[0]!r}, workflow={key[1]!r}"
            )
        checkpoint = codec.load(directory)
        if not isinstance(checkpoint, codec.checkpoint_type):
            raise TypeError(
                "checkpoint codec returned an incompatible type: "
                f"expected {codec.checkpoint_type.__name__}, "
                f"got {type(checkpoint).__name__}"
            )
        return checkpoint


__all__ = [
    "CheckpointCodec",
    "CheckpointCodecRegistry",
    "CheckpointLoadCallable",
    "CheckpointSaveCallable",
]
