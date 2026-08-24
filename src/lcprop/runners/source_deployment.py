"""Material-neutral deployment of immutable committed LCProp source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Protocol
from uuid import uuid4

from lcprop.runners.slurm import validate_remote_path


DEPLOYABLE_PATHS = ("src", "pyproject.toml")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_CHECKSUM = re.compile(r"[0-9a-f]{64}")
_LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"
_LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:[0-9a-f]{64}\r?\n"
    rb"size (?:0|[1-9][0-9]*)\r?\n?"
)
_MAX_LFS_POINTER_BYTES = 1024


class SourceDeploymentError(RuntimeError):
    """One categorized failure before scientific job submission."""

    def __init__(self, category: str, reason: str) -> None:
        self.category = category
        self.reason = reason
        super().__init__(f"{category}: {reason}")


class SourceRemoteTransport(Protocol):
    def ssh(self, host: str, *arguments: str) -> str: ...
    def upload(self, host: str, local: Path, remote: str) -> None: ...


@dataclass(frozen=True)
class GitSourceIdentity:
    repository_root: Path
    git_sha: str
    deployable_paths: tuple[str, ...] = DEPLOYABLE_PATHS


@dataclass(frozen=True)
class ResolvedSourceDeployment:
    source_kind: str
    source_git_sha: str
    remote_source_path: str
    source_checksum: str | None
    reused_existing_snapshot: bool
    provenance: dict[str, object]


def _git(repository: Path, *arguments: str, text: bool = True):
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = getattr(exc, "stderr", None) or str(exc)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise SourceDeploymentError("source_identity_failed", message.strip()) from exc


def _is_git_lfs_pointer(blob: bytes) -> bool:
    """Return whether a complete committed blob is a canonical Git LFS pointer."""

    return (
        len(blob) <= _MAX_LFS_POINTER_BYTES and _LFS_POINTER.fullmatch(blob) is not None
    )


def _git_lfs_pointer_paths(
    root: Path,
    sha: str,
    deployable_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Inspect complete committed blobs selected by a cheap signature search."""

    candidates = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "grep",
            "-z",
            "-l",
            "-F",
            _LFS_SIGNATURE.decode("ascii"),
            sha,
            "--",
            *deployable_paths,
        ],
        capture_output=True,
    )
    if candidates.returncode not in {0, 1}:
        raise SourceDeploymentError(
            "source_identity_failed",
            candidates.stderr.decode("utf-8", errors="replace").strip()
            or "Git LFS candidate probe failed",
        )
    prefix = (sha + ":").encode("ascii")
    pointers: list[str] = []
    for reference in candidates.stdout.split(b"\0"):
        if not reference:
            continue
        if not reference.startswith(prefix):
            raise SourceDeploymentError(
                "source_identity_failed",
                "Git returned an invalid LFS candidate reference",
            )
        path = reference[len(prefix) :].decode("utf-8", errors="surrogateescape")
        blob = _git(root, "show", f"{sha}:{path}", text=False).stdout
        if _is_git_lfs_pointer(blob):
            pointers.append(path)
    return tuple(pointers)


def resolve_git_source(
    local_source: str | Path,
    *,
    deployable_paths: tuple[str, ...] = DEPLOYABLE_PATHS,
) -> GitSourceIdentity:
    """Resolve a clean committed source identity without reading untracked content."""

    candidate = Path(local_source).expanduser().resolve()
    try:
        root_result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceDeploymentError(
            "source_not_git_checkout",
            f"{candidate} is not inside a Git checkout; use an explicit pre-staged source override",
        ) from exc
    root = Path(root_result.stdout.strip()).resolve()
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()
    if not _FULL_SHA.fullmatch(sha):
        raise SourceDeploymentError(
            "source_identity_failed", f"Git returned invalid HEAD identity {sha!r}"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *deployable_paths,
    ).stdout
    if status:
        staged = any(line and line[0] not in {" ", "?"} for line in status.splitlines())
        detail = "staged or tracked" if staged else "untracked"
        raise SourceDeploymentError(
            "source_dirty",
            f"deployable source contains {detail} changes:\n{status.rstrip()}",
        )
    members = _git(
        root, "ls-tree", "-r", "--name-only", sha, "--", *deployable_paths
    ).stdout.splitlines()
    if "pyproject.toml" not in members or not any(
        value.startswith("src/lcprop/") for value in members
    ):
        raise SourceDeploymentError(
            "source_identity_failed",
            "selected commit does not contain pyproject.toml and src/lcprop",
        )
    tree = _git(
        root, "ls-tree", "-r", sha, "--", *deployable_paths
    ).stdout.splitlines()
    unsupported_modes = [
        line for line in tree if line.startswith(("120000 ", "160000 "))
    ]
    if unsupported_modes:
        raise SourceDeploymentError(
            "source_archive_failed",
            "deployable source contains unsupported symbolic-link or submodule "
            f"entries:\n{chr(10).join(unsupported_modes)}",
        )
    lfs_pointers = _git_lfs_pointer_paths(root, sha, tuple(deployable_paths))
    if lfs_pointers:
        raise SourceDeploymentError(
            "source_archive_failed",
            "deployable source contains Git LFS pointer files, which automatic "
            "source staging does not support:\n" + "\n".join(lfs_pointers),
        )
    return GitSourceIdentity(root, sha, tuple(deployable_paths))


def build_committed_source_archive(
    identity: GitSourceIdentity,
    destination: str | Path,
) -> str:
    """Build a deterministic tar archive from committed Git objects only."""

    archive = Path(destination)
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(identity.repository_root),
                "archive",
                "--format=tar",
                f"--output={archive}",
                identity.git_sha,
                "--",
                *identity.deployable_paths,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = getattr(exc, "stderr", None) or str(exc)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise SourceDeploymentError("source_archive_failed", message.strip()) from exc
    digest = hashlib.sha256()
    try:
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceDeploymentError("source_archive_failed", str(exc)) from exc
    return digest.hexdigest()


class SourceDeploymentManager:
    """Resolve and atomically stage exact committed source for one SSH cluster."""

    def __init__(
        self,
        *,
        host: str,
        source_root: str,
        local_source: str | Path,
        transport: SourceRemoteTransport | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
            raise ValueError("host must be an SSH hostname or user@hostname")
        self.host = host
        self.source_root = validate_remote_path(source_root)
        self.local_source = Path(local_source).expanduser().resolve()
        if transport is None:
            from lcprop.runners.slurm import SubprocessRemoteTransport

            transport = SubprocessRemoteTransport()
        self._transport = transport

    @staticmethod
    def snapshot_name(git_sha: str) -> str:
        if not _FULL_SHA.fullmatch(git_sha):
            raise ValueError("git_sha must be an exact 40-character SHA")
        return f"git-{git_sha}"

    def _exists(self, path: str) -> bool:
        try:
            self._transport.ssh(self.host, "test", "-e", path)
        except Exception:
            return False
        return True

    def _verify_existing(
        self,
        remote_snapshot: str,
        *,
        git_sha: str,
        checksum: str,
    ) -> bool:
        if not self._exists(remote_snapshot):
            return False
        try:
            remote_sha = self._transport.ssh(
                self.host, "cat", f"{remote_snapshot}/.lcprop-source-sha"
            ).strip()
            remote_checksum = self._transport.ssh(
                self.host, "cat", f"{remote_snapshot}/.lcprop-source-checksum"
            ).strip()
            marker_text = self._transport.ssh(
                self.host, "cat", f"{remote_snapshot}/.lcprop-source.json"
            )
            self._transport.ssh(
                self.host, "test", "-d", f"{remote_snapshot}/src/lcprop"
            )
        except Exception as exc:
            raise SourceDeploymentError(
                "snapshot_identity_mismatch",
                f"existing snapshot {remote_snapshot} is incomplete: {exc}",
            ) from exc
        try:
            marker = json.loads(marker_text)
        except json.JSONDecodeError as exc:
            raise SourceDeploymentError(
                "snapshot_identity_mismatch",
                f"existing snapshot has invalid versioned marker: {exc}",
            ) from exc
        expected_marker = {
            "archive_sha256": checksum,
            "deployable_paths": list(DEPLOYABLE_PATHS),
            "git_sha": git_sha,
            "source_kind": "committed_git_archive",
            "source_marker_version": 1,
        }
        if (
            remote_sha != git_sha
            or remote_checksum != checksum
            or marker != expected_marker
        ):
            raise SourceDeploymentError(
                "snapshot_identity_mismatch",
                "existing snapshot identity does not match the selected committed source: "
                f"SHA {remote_sha!r}, checksum {remote_checksum!r}",
            )
        return True

    def _cleanup_staging(self, staging: str) -> None:
        try:
            self._transport.ssh(self.host, "chmod", "-R", "u+w", staging)
        except Exception:
            pass
        try:
            self._transport.ssh(self.host, "rm", "-rf", staging)
        except Exception:
            pass

    def _stage(
        self,
        *,
        archive: Path,
        checksum: str,
        git_sha: str,
        remote_snapshot: str,
        marker_directory: Path,
    ) -> bool:
        token = uuid4().hex
        staging = validate_remote_path(f"{self.source_root}/.staging-{token}")
        remote_archive = f"{staging}/source.tar"
        failure_category = "source_upload_failed"
        try:
            self._transport.ssh(self.host, "mkdir", "-p", self.source_root)
            self._transport.ssh(self.host, "mkdir", staging)
            self._transport.upload(self.host, archive, remote_archive)
            self._transport.upload(
                self.host,
                marker_directory / ".lcprop-source-sha",
                f"{staging}/.lcprop-source-sha",
            )
            self._transport.upload(
                self.host,
                marker_directory / ".lcprop-source-checksum",
                f"{staging}/.lcprop-source-checksum",
            )
            self._transport.upload(
                self.host,
                marker_directory / ".lcprop-source.json",
                f"{staging}/.lcprop-source.json",
            )
            failure_category = "source_checksum_failed"
            actual = self._transport.ssh(
                self.host, "sha256sum", remote_archive
            ).split()[0]
            if not _CHECKSUM.fullmatch(actual) or actual != checksum:
                raise SourceDeploymentError(
                    "source_checksum_failed",
                    f"uploaded archive checksum {actual!r} does not match {checksum}",
                )
            failure_category = "snapshot_extract_failed"
            self._transport.ssh(
                self.host, "tar", "-xf", remote_archive, "-C", staging
            )
            self._transport.ssh(
                self.host, "test", "-d", f"{staging}/src/lcprop"
            )
            self._transport.ssh(self.host, "rm", "-f", remote_archive)
            failure_category = "snapshot_finalize_failed"
            self._transport.ssh(self.host, "chmod", "-R", "a-w", staging)
            try:
                self._transport.ssh(
                    self.host,
                    "ln",
                    "-s",
                    PurePosixPath(staging).name,
                    remote_snapshot,
                )
            except Exception:
                if self._verify_existing(
                    remote_snapshot, git_sha=git_sha, checksum=checksum
                ):
                    self._cleanup_staging(staging)
                    return True
                raise
        except SourceDeploymentError:
            self._cleanup_staging(staging)
            raise
        except Exception as exc:
            self._cleanup_staging(staging)
            raise SourceDeploymentError(failure_category, str(exc)) from exc
        return False

    def resolve_or_stage(self) -> ResolvedSourceDeployment:
        identity = resolve_git_source(self.local_source)
        remote_snapshot = validate_remote_path(
            f"{self.source_root}/{self.snapshot_name(identity.git_sha)}"
        )
        with tempfile.TemporaryDirectory(prefix="lcprop-source-") as directory:
            local = Path(directory)
            archive = local / "source.tar"
            checksum = build_committed_source_archive(identity, archive)
            if self._verify_existing(
                remote_snapshot, git_sha=identity.git_sha, checksum=checksum
            ):
                reused = True
            else:
                (local / ".lcprop-source-sha").write_text(
                    identity.git_sha + "\n", encoding="ascii"
                )
                (local / ".lcprop-source-checksum").write_text(
                    checksum + "\n", encoding="ascii"
                )
                (local / ".lcprop-source.json").write_text(
                    json.dumps(
                        {
                            "archive_sha256": checksum,
                            "deployable_paths": list(DEPLOYABLE_PATHS),
                            "git_sha": identity.git_sha,
                            "source_kind": "committed_git_archive",
                            "source_marker_version": 1,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                reused = self._stage(
                    archive=archive,
                    checksum=checksum,
                    git_sha=identity.git_sha,
                    remote_snapshot=remote_snapshot,
                    marker_directory=local,
                )
                if not self._verify_existing(
                    remote_snapshot,
                    git_sha=identity.git_sha,
                    checksum=checksum,
                ):  # pragma: no cover - final link was just published
                    raise SourceDeploymentError(
                        "snapshot_finalize_failed",
                        "final snapshot disappeared before post-publication verification",
                    )
        provenance: dict[str, object] = {
            "source_kind": "committed_git_archive",
            "source_git_sha": identity.git_sha,
            "remote_source_path": remote_snapshot,
            "source_checksum_sha256": checksum,
            "snapshot_reused": reused,
            "source_marker_version": 1,
        }
        return ResolvedSourceDeployment(
            source_kind="committed_git_archive",
            source_git_sha=identity.git_sha,
            remote_source_path=remote_snapshot,
            source_checksum=checksum,
            reused_existing_snapshot=reused,
            provenance=provenance,
        )


def explicit_source_deployment(
    *, remote_source_path: str, source_git_sha: str
) -> ResolvedSourceDeployment:
    """Represent one advanced-user/CI pre-staged source override."""

    path = validate_remote_path(remote_source_path)
    if not _FULL_SHA.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be an exact 40-character SHA")
    provenance: dict[str, object] = {
        "source_kind": "explicit_pre_staged",
        "source_git_sha": source_git_sha,
        "remote_source_path": path,
        "source_checksum_sha256": None,
        "snapshot_reused": True,
    }
    return ResolvedSourceDeployment(
        source_kind="explicit_pre_staged",
        source_git_sha=source_git_sha,
        remote_source_path=path,
        source_checksum=None,
        reused_existing_snapshot=True,
        provenance=provenance,
    )


__all__ = [
    "DEPLOYABLE_PATHS",
    "GitSourceIdentity",
    "ResolvedSourceDeployment",
    "SourceDeploymentError",
    "SourceDeploymentManager",
    "build_committed_source_archive",
    "explicit_source_deployment",
    "resolve_git_source",
]
