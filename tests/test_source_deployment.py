import hashlib
import inspect
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

import pytest

from lcprop.runners.source_deployment import (
    SourceDeploymentError,
    SourceDeploymentManager,
    _is_git_lfs_pointer,
    build_committed_source_archive,
    resolve_git_source,
)


def _run(*arguments, cwd):
    return subprocess.run(
        arguments, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repository"
    (root / "src/lcprop").mkdir(parents=True)
    (root / "src/lcprop/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src/lcprop/data.txt").write_text("resource\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='lcprop-test'\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs/note.md").write_text("committed but not deployable\n", encoding="utf-8")
    _run("git", "init", "-q", cwd=root)
    _run("git", "config", "user.email", "test@example.invalid", cwd=root)
    _run("git", "config", "user.name", "Test", cwd=root)
    _run("git", "add", ".", cwd=root)
    _run("git", "commit", "-qm", "fixture", cwd=root)
    return root, _run("git", "rev-parse", "HEAD", cwd=root)


class LocalRemoteTransport:
    def __init__(self, root: Path):
        self.root = root
        self.upload_count = 0
        self.upload_bytes = 0
        self.commands = []
        self.fail_extract = False
        self.corrupt_archive_upload = False
        self.simulate_winning_finalizer = False

    def _path(self, remote: str) -> Path:
        assert remote.startswith("/")
        return self.root / remote.lstrip("/")

    @staticmethod
    def _failure(arguments):
        raise subprocess.CalledProcessError(1, arguments)

    def ssh(self, host, *arguments):
        self.commands.append((host, arguments))
        command = arguments[0]
        if command == "test":
            flag, remote = arguments[1:]
            path = self._path(remote)
            valid = path.exists() if flag == "-e" else path.is_dir()
            if not valid:
                self._failure(arguments)
            return ""
        if command == "cat":
            return self._path(arguments[1]).read_text(encoding="ascii").strip()
        if command == "mkdir":
            if arguments[1] == "-p":
                self._path(arguments[2]).mkdir(parents=True, exist_ok=True)
            else:
                self._path(arguments[1]).mkdir()
            return ""
        if command == "sha256sum":
            path = self._path(arguments[1])
            return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {arguments[1]}"
        if command == "tar":
            if self.fail_extract:
                self._failure(arguments)
            archive = self._path(arguments[2])
            destination = self._path(arguments[4])
            with tarfile.open(archive, "r:") as bundle:
                if "filter" in __import__("inspect").signature(bundle.extractall).parameters:
                    bundle.extractall(destination, filter="data")
                else:  # pragma: no cover - Python 3.10/3.11 compatibility
                    bundle.extractall(destination)
            return ""
        if command == "rm":
            path = self._path(arguments[-1])
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            return ""
        if command == "chmod":
            return ""
        if command == "ln":
            target, link = arguments[2:]
            path = self._path(link)
            if self.simulate_winning_finalizer:
                loser = self._path(str(PurePosixPath(link).parent / target))
                winner = path.parent / ".winner"
                shutil.copytree(loser, winner)
                path.symlink_to(winner.name)
                self.simulate_winning_finalizer = False
                self._failure(arguments)
            if path.exists() or path.is_symlink():
                self._failure(arguments)
            path.symlink_to(target)
            return ""
        raise AssertionError(arguments)

    def upload(self, host, local, remote):
        source = Path(local)
        target = self._path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if self.corrupt_archive_upload and target.name == "source.tar":
            target.write_bytes(target.read_bytes() + b"corrupt")
        self.upload_count += 1
        self.upload_bytes += source.stat().st_size


def test_clean_branch_and_detached_checkout_resolve_exact_sha(tmp_path):
    root, sha = _repository(tmp_path)
    branch = resolve_git_source(root)
    assert branch.git_sha == sha
    _run("git", "checkout", "--detach", "-q", sha, cwd=root)
    detached = resolve_git_source(root / "src")
    assert detached.git_sha == sha
    assert detached.repository_root == root


@pytest.mark.parametrize("staged", (False, True))
def test_dirty_or_staged_deployable_source_is_rejected(tmp_path, staged):
    root, _sha = _repository(tmp_path)
    (root / "src/lcprop/__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    if staged:
        _run("git", "add", "src/lcprop/__init__.py", cwd=root)
    with pytest.raises(SourceDeploymentError, match="source_dirty"):
        resolve_git_source(root)


def test_untracked_deployable_source_is_rejected_but_unrelated_artifacts_are_ignored(
    tmp_path,
):
    root, sha = _repository(tmp_path)
    (root / "results").mkdir()
    (root / "results/large.bin").write_bytes(b"private-result")
    assert resolve_git_source(root).git_sha == sha
    (root / "src/lcprop/untracked.py").write_text("SECRET = 1\n", encoding="utf-8")
    with pytest.raises(SourceDeploymentError, match="source_dirty"):
        resolve_git_source(root)


def test_archive_is_stable_committed_content_only(tmp_path):
    root, _sha = _repository(tmp_path)
    (root / "results").mkdir()
    (root / "results/private.dat").write_bytes(b"not archived")
    identity = resolve_git_source(root)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_checksum = build_committed_source_archive(identity, first)
    second_checksum = build_committed_source_archive(identity, second)
    assert first_checksum == second_checksum
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:") as archive:
        names = set(archive.getnames())
    assert "pyproject.toml" in names
    assert "src/lcprop/__init__.py" in names
    assert "src/lcprop/data.txt" in names
    assert not any(name.startswith(".git") for name in names)
    assert not any(name.startswith("results") for name in names)
    assert not any(name.startswith("docs") for name in names)


def test_non_git_install_fails_actionably(tmp_path):
    with pytest.raises(SourceDeploymentError, match="source_not_git_checkout"):
        resolve_git_source(tmp_path)


def test_current_committed_source_deployment_module_is_not_an_lfs_pointer(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    blob = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "show",
            "HEAD:src/lcprop/runners/source_deployment.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert b"version https://git-lfs.github.com/spec/v1" in blob
    assert not _is_git_lfs_pointer(blob)
    root, _sha = _repository(tmp_path)
    deployed_module = root / "src/lcprop/source_deployment.py"
    deployed_module.write_bytes(blob)
    _run("git", "add", str(deployed_module.relative_to(root)), cwd=root)
    _run("git", "commit", "-qm", "add detector source", cwd=root)
    assert resolve_git_source(root).git_sha == _run(
        "git", "rev-parse", "HEAD", cwd=root
    )


def test_lfs_signature_inside_ordinary_source_text_is_not_a_pointer():
    signature = "version https://git-lfs.github.com/spec/v1"
    python_source = (
        f'LFS_SIGNATURE = "{signature}"\n'
        'DOCUMENTATION = "oid sha256:' + "0" * 64 + '\\nsize 1"\n'
    ).encode("ascii")
    documentation = (
        b"Git LFS pointers begin with "
        + signature.encode("ascii")
        + b", but this document is not one.\n"
    )
    assert not _is_git_lfs_pointer(python_source)
    assert not _is_git_lfs_pointer(documentation)


def test_complete_valid_git_lfs_pointer_is_recognized():
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + b"a" * 64
        + b"\nsize 12345\n"
    )
    assert _is_git_lfs_pointer(pointer)


@pytest.mark.parametrize(
    "blob",
    (
        b"version https://git-lfs.github.com/spec/v1\n",
        b"version https://git-lfs.github.com/spec/v1\nsize 1\n",
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:not-a-valid-oid\nsize 1\n",
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + b"0" * 64
        + b"\n",
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + b"0" * 64
        + b"\nsize 1\nunexpected trailing text\n",
    ),
)
def test_malformed_or_partial_lfs_pointer_text_is_not_recognized(blob):
    assert not _is_git_lfs_pointer(blob)


def test_git_lfs_pointer_and_symlink_sources_are_rejected(tmp_path):
    root, _sha = _repository(tmp_path)
    pointer = root / "src/lcprop/large.dat"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 1\n",
        encoding="ascii",
    )
    _run("git", "add", str(pointer.relative_to(root)), cwd=root)
    _run("git", "commit", "-qm", "add lfs pointer", cwd=root)
    with pytest.raises(SourceDeploymentError, match="Git LFS"):
        resolve_git_source(root)

    pointer.unlink()
    link = root / "src/lcprop/link.py"
    link.symlink_to("__init__.py")
    _run("git", "add", "-A", cwd=root)
    _run("git", "commit", "-qm", "replace pointer with link", cwd=root)
    with pytest.raises(SourceDeploymentError, match="symbolic-link"):
        resolve_git_source(root)


def test_absent_snapshot_is_atomic_and_second_resolve_reuses_without_upload(tmp_path):
    root, sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    manager = SourceDeploymentManager(
        host="user@login.example.edu",
        source_root="/sources",
        local_source=root,
        transport=remote,
    )
    first = manager.resolve_or_stage()
    upload_count = remote.upload_count
    second = manager.resolve_or_stage()
    assert first.source_git_sha == sha
    assert first.remote_source_path == f"/sources/git-{sha}"
    assert first.source_checksum and len(first.source_checksum) == 64
    assert first.reused_existing_snapshot is False
    assert second.reused_existing_snapshot is True
    assert second.source_checksum == first.source_checksum
    assert remote.upload_count == upload_count
    final = remote._path(first.remote_source_path)
    assert final.is_symlink()
    assert (final / "src/lcprop").is_dir()
    assert (final / ".lcprop-source-sha").read_text().strip() == sha
    assert (final / ".lcprop-source-checksum").read_text().strip() == first.source_checksum
    assert not (final / "source.tar").exists()


def test_invalid_existing_snapshot_is_rejected_without_overwrite(tmp_path):
    root, sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    final = remote._path(f"/sources/git-{sha}")
    final.mkdir(parents=True)
    (final / ".lcprop-source-sha").write_text("0" * 40 + "\n")
    (final / ".lcprop-source-checksum").write_text("0" * 64 + "\n")
    manager = SourceDeploymentManager(
        host="login.example.edu", source_root="/sources",
        local_source=root, transport=remote,
    )
    with pytest.raises(SourceDeploymentError, match="snapshot_identity_mismatch"):
        manager.resolve_or_stage()
    assert remote.upload_count == 0
    assert (final / ".lcprop-source-sha").read_text().strip() == "0" * 40


def test_failed_extract_leaves_no_valid_final_snapshot(tmp_path):
    root, sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    remote.fail_extract = True
    manager = SourceDeploymentManager(
        host="login.example.edu", source_root="/sources",
        local_source=root, transport=remote,
    )
    with pytest.raises(SourceDeploymentError):
        manager.resolve_or_stage()
    assert not remote._path(f"/sources/git-{sha}").exists()
    assert not list(remote._path("/sources").glob(".staging-*"))


def test_upload_checksum_failure_is_categorized_and_never_finalized(tmp_path):
    root, sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    remote.corrupt_archive_upload = True
    manager = SourceDeploymentManager(
        host="login.example.edu", source_root="/sources",
        local_source=root, transport=remote,
    )
    with pytest.raises(SourceDeploymentError) as caught:
        manager.resolve_or_stage()
    assert caught.value.category == "source_checksum_failed"
    assert not remote._path(f"/sources/git-{sha}").exists()


def test_concurrent_finalizer_winner_is_verified_and_reused(tmp_path):
    root, sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    remote.simulate_winning_finalizer = True
    manager = SourceDeploymentManager(
        host="login.example.edu", source_root="/sources",
        local_source=root, transport=remote,
    )
    result = manager.resolve_or_stage()
    assert result.reused_existing_snapshot is True
    final = remote._path(f"/sources/git-{sha}")
    assert final.is_symlink()
    assert (final / "src/lcprop").is_dir()
    assert not list(remote._path("/sources").glob(".staging-*"))


def test_staging_applies_read_only_command_before_publishing_snapshot(tmp_path):
    root, _sha = _repository(tmp_path)
    remote = LocalRemoteTransport(tmp_path / "remote")
    manager = SourceDeploymentManager(
        host="login.example.edu", source_root="/sources",
        local_source=root, transport=remote,
    )
    manager.resolve_or_stage()
    commands = [arguments for _host, arguments in remote.commands]
    chmod_index = next(
        index for index, value in enumerate(commands)
        if value[:3] == ("chmod", "-R", "a-w")
    )
    link_index = next(
        index for index, value in enumerate(commands) if value[0] == "ln"
    )
    assert chmod_index < link_index


def test_snapshot_name_requires_and_preserves_full_sha():
    sha = "1" * 40
    assert SourceDeploymentManager.snapshot_name(sha) == f"git-{sha}"
    with pytest.raises(ValueError):
        SourceDeploymentManager.snapshot_name(sha[:12])


def test_deployment_manager_is_material_neutral():
    source = inspect.getsource(SourceDeploymentManager)
    assert "material_id" not in source
    assert "workflow_id" not in source
    assert "sbatch" not in source
