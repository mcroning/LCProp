"""Verified JSON/NPZ artifact bundles for remote scientific transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from lcprop.transport.envelopes import (
    MANIFEST_FORMAT,
    MANIFEST_SCHEMA_VERSION,
    TransportFormatError,
    TransportVerificationError,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        encoded = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise TransportFormatError(f"metadata is not portable JSON: {exc}") from exc
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _safe_relative_filename(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TransportFormatError("artifact filename must be non-empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise TransportFormatError(f"unsafe artifact filename {value!r}")
    return value


def normalize_host_arrays(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise TransportFormatError("array keys must be non-empty trimmed strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TransportFormatError(f"array {key!r} has forbidden object dtype")
        result[key] = np.ascontiguousarray(array)
    return result


def array_manifest(
    arrays: Mapping[str, Any],
    *,
    filename: str,
) -> dict[str, dict[str, Any]]:
    filename = _safe_relative_filename(filename)
    normalized = normalize_host_arrays(arrays)
    return {
        key: {
            "filename": filename,
            "member": key,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
            "required": True,
        }
        for key, value in sorted(normalized.items())
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def write_artifact_bundle(
    directory: str | Path,
    *,
    run_id: str,
    envelope_filename: str,
    envelope: Mapping[str, Any],
    arrays_filename: str,
    arrays: Mapping[str, Any],
    ready_filename: str,
) -> Path:
    """Write one immutable payload bundle and its verification metadata."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    envelope_filename = _safe_relative_filename(envelope_filename)
    arrays_filename = _safe_relative_filename(arrays_filename)
    ready_filename = _safe_relative_filename(ready_filename)
    normalized = normalize_host_arrays(arrays)

    envelope_path = output / envelope_filename
    _atomic_json(envelope_path, envelope)
    payload_paths = [envelope_path]
    roles = {envelope_filename: "envelope"}
    if normalized:
        arrays_path = output / arrays_filename
        _write_npz(arrays_path, normalized)
        payload_paths.append(arrays_path)
        roles[arrays_filename] = "host_arrays"

    entries = []
    for path in sorted(payload_paths, key=lambda item: item.name):
        entries.append({
            "filename": path.name,
            "role": roles[path.name],
            "required": True,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "format": MANIFEST_FORMAT,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "files": entries,
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)

    checksum_lines = [
        f"{entry['sha256']}  {entry['filename']}" for entry in entries
    ]
    checksum_lines.append(f"{sha256_file(manifest_path)}  manifest.json")
    checksums_path = output / "checksums.sha256"
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    _atomic_json(output / ready_filename, {
        "run_id": run_id,
        "manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(checksums_path),
    })
    return output


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise TransportVerificationError(f"symlink artifacts are forbidden: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportVerificationError(f"could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransportVerificationError(f"{path.name} must contain a JSON object")
    return value


def verify_artifact_bundle(
    directory: str | Path,
    *,
    envelope_filename: str,
    arrays_filename: str,
    ready_filename: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify a complete bundle before loading any scientific arrays."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise TransportVerificationError("artifact bundle is not a real directory")
    envelope_filename = _safe_relative_filename(envelope_filename)
    arrays_filename = _safe_relative_filename(arrays_filename)
    ready_filename = _safe_relative_filename(ready_filename)
    manifest_path = root / "manifest.json"
    checksums_path = root / "checksums.sha256"
    ready_path = root / ready_filename
    for path in (manifest_path, checksums_path, ready_path):
        if not path.is_file() or path.is_symlink():
            raise TransportVerificationError(f"missing required artifact {path.name}")

    ready = _read_json(ready_path)
    if ready.get("manifest_sha256") != sha256_file(manifest_path):
        raise TransportVerificationError("manifest checksum verification failed")
    if ready.get("checksums_sha256") != sha256_file(checksums_path):
        raise TransportVerificationError("checksum-list verification failed")

    manifest = _read_json(manifest_path)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise TransportVerificationError("invalid artifact manifest marker")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TransportVerificationError("unsupported artifact manifest version")
    if ready.get("run_id") != manifest.get("run_id"):
        raise TransportVerificationError("completion marker run_id mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise TransportVerificationError("artifact manifest files must be a list")

    declared: dict[str, dict[str, Any]] = {}
    expected_checksum_lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TransportVerificationError("invalid artifact manifest entry")
        name = _safe_relative_filename(entry.get("filename"))
        if name in declared:
            raise TransportVerificationError(f"duplicate artifact {name!r}")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise TransportVerificationError(f"missing required artifact {name}")
        if path.stat().st_size != entry.get("byte_size"):
            raise TransportVerificationError(f"artifact size mismatch for {name}")
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise TransportVerificationError(f"artifact checksum mismatch for {name}")
        declared[name] = entry
        expected_checksum_lines.append(f"{digest}  {name}")
    expected_checksum_lines.append(f"{sha256_file(manifest_path)}  manifest.json")
    actual_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    if actual_lines != expected_checksum_lines:
        raise TransportVerificationError("checksums.sha256 does not match manifest")
    if envelope_filename not in declared:
        raise TransportVerificationError("envelope is not declared by manifest")

    envelope = _read_json(root / envelope_filename)
    if envelope.get("run_id") != manifest.get("run_id"):
        raise TransportVerificationError("envelope/manifest run_id mismatch")
    array_specs = envelope.get("array_manifest", {})
    if not isinstance(array_specs, dict):
        raise TransportVerificationError("array_manifest must be a JSON object")
    arrays: dict[str, np.ndarray] = {}
    if array_specs:
        if arrays_filename not in declared:
            raise TransportVerificationError("array artifact is not declared")
        arrays_path = root / arrays_filename
        try:
            with np.load(arrays_path, allow_pickle=False) as loaded:
                if set(loaded.files) != set(array_specs):
                    raise TransportVerificationError("NPZ members do not match array manifest")
                for key, spec in array_specs.items():
                    if not isinstance(spec, dict):
                        raise TransportVerificationError(f"invalid array manifest for {key!r}")
                    if spec.get("filename") != arrays_filename or spec.get("member") != key:
                        raise TransportVerificationError(f"invalid array reference for {key!r}")
                    value = np.asarray(loaded[key])
                    if value.dtype.hasobject:
                        raise TransportVerificationError(f"object array forbidden for {key!r}")
                    if str(value.dtype) != spec.get("dtype"):
                        raise TransportVerificationError(f"array dtype mismatch for {key!r}")
                    if list(value.shape) != spec.get("shape"):
                        raise TransportVerificationError(f"array shape mismatch for {key!r}")
                    if int(value.nbytes) != spec.get("nbytes"):
                        raise TransportVerificationError(f"array size mismatch for {key!r}")
                    arrays[key] = value.copy()
        except TransportVerificationError:
            raise
        except Exception as exc:
            raise TransportVerificationError(f"could not load host arrays: {exc}") from exc
    elif arrays_filename in declared:
        raise TransportVerificationError("undeclared array artifact is present")
    return envelope, arrays


__all__ = [
    "array_manifest",
    "normalize_host_arrays",
    "sha256_file",
    "verify_artifact_bundle",
    "write_artifact_bundle",
]
