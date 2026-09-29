"""Deploy the verified pure-Python distribution owning the active LCProp import.

No build backend, wheel cache, dependency installer or Git identity is involved.
"""
from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from email.parser import Parser
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

from lcprop.runners.source_deployment import (
    ResolvedSourceDeployment, SourceDeploymentError, SourceDeploymentManager,
    _is_git_lfs_pointer, SourceSubmissionBinding,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fail(reason):
    raise SourceDeploymentError("installed_distribution_invalid", reason)


def _safe_path(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or any(ord(c) < 32 for c in name)):
        _fail(f"unsafe distribution path: {name!r}")
    return path


def _regular(path, root):
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"symbolic-link distribution entry: {path}")
    if not path.is_file():
        _fail(f"missing/nonregular distribution entry: {path}")
    return path.read_bytes()


@dataclass(frozen=True)
class InstalledArtifact:
    name: str
    version: str
    files: tuple[tuple[str, bytes], ...]

    @property
    def manifest(self):
        return {
            "artifact_version": 1,
            "source_kind": "installed_distribution",
            "distribution_name": self.name,
            "distribution_version": self.version,
            "files": [dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
                      for n, b in self.files],
        }

    @property
    def digest(self):
        return hashlib.sha256(_json(self.manifest).encode()).hexdigest()

    def archive(self, destination):
        with tarfile.open(destination, "w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in self.files:
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o644
                entry.mtime = 0
                archive.addfile(entry, io.BytesIO(data))
        return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


def owning_distribution(package_file, distributions=None):
    """Match installed ownership to the import, never the working directory."""
    active = Path(package_file).absolute()
    owners = []
    for dist in metadata.distributions() if distributions is None else distributions:
        if re.sub(r"[-_.]+", "-", dist.metadata.get("Name", "")).lower() != "lcprop":
            continue
        if any(str(f) == "lcprop/__init__.py"
               and Path(dist.locate_file(f)).absolute() == active
               for f in dist.files or ()):
            owners.append(dist)
    if len(owners) > 1:
        _fail("ambiguous distribution ownership of active LCProp import")
    return owners[0] if owners else None


def resolve_installed_distribution(package_file, distributions=None):
    """Resolve verified bytes, reporting malformed/unreadable installs locally."""
    try:
        return _resolve_installed_distribution(package_file, distributions)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        _fail(f"cannot verify installed distribution: {exc}")


def _resolve_installed_distribution(package_file, distributions=None):
    active = Path(package_file).absolute()
    dist = owning_distribution(active, distributions)
    if dist is None:
        _fail("no RECORD-owned distribution matches the active LCProp import")
    root = Path(dist.locate_file("")).absolute()
    if active != root / "lcprop/__init__.py":
        _fail("active LCProp import does not match distribution root")
    paths = [str(f) for f in dist.files or ()]
    records = [n for n in paths if n.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        _fail("one installed wheel RECORD is required")
    info = str(_safe_path(records[0]).parent)
    if "/" in info:
        _fail("unsupported nested distribution metadata")
    record = _regular(root / records[0], root).decode("utf-8")
    rows = list(csv.reader(io.StringIO(record)))
    seen = set()
    payload = {}
    for row in rows:
        if len(row) != 3:
            _fail("malformed RECORD row")
        name, checksum, size = row
        if name in seen:
            _fail(f"duplicate RECORD entry: {name}")
        seen.add(name)
        # pip's generated entry-point script is outside site-packages. Never read it.
        if name == "../../../bin/lcprop-pr" or name == "../../../Scripts/lcprop-pr.exe":
            continue
        _safe_path(name)
        if not (name.startswith("lcprop/") or name.startswith(info + "/")):
            _fail(f"unsupported distribution-owned entry: {name}")
        if "__pycache__" in PurePosixPath(name).parts or name.endswith((".pyc", ".pyo")):
            continue
        raw = _regular(root / name, root)
        if name != records[0]:
            if not checksum.startswith("sha256=") or not size.isdecimal():
                _fail(f"missing/unsupported RECORD checksum: {name}")
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            if checksum != "sha256=" + digest or len(raw) != int(size):
                _fail(f"RECORD size/checksum mismatch: {name}")
        if name.startswith("lcprop/"):
            if name.endswith((".so", ".pyd", ".dll", ".dylib", ".exe", ".pth")):
                _fail(f"native or executable payload is unsupported: {name}")
            if not name.endswith((".py", ".json", ".png")):
                _fail(f"unsupported LCProp package resource: {name}")
            if _is_git_lfs_pointer(raw):
                _fail(f"unresolved LFS resource: {name}")
            payload["src/" + name] = raw
        elif (name in {info + "/METADATA", info + "/WHEEL", info + "/top_level.txt"}
              or name.startswith(info + "/licenses/")):
            payload["src/" + name] = raw
    for path in (root / "lcprop").rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            _fail(f"symbolic-link package entry: {name}")
        if not path.is_dir() and not path.is_file():
            _fail(f"nonregular package entry: {name}")
        if path.is_file() and "__pycache__" not in path.parts and not name.endswith((".pyc", ".pyo")):
            if name not in seen:
                _fail(f"unowned package file: {name}")
    try:
        wheel = Parser().parsestr(payload["src/" + info + "/WHEEL"].decode())
        met = Parser().parsestr(payload["src/" + info + "/METADATA"].decode())
        direct = root / info / "direct_url.json"
        if direct.exists() and json.loads(_regular(direct, root)).get("dir_info", {}).get("editable"):
            _fail("editable distributions require the strict Git provider")
        if wheel.get("Root-Is-Purelib", "").lower() != "true" or wheel.get_all("Tag") != ["py3-none-any"]:
            _fail("only pure-Python py3-none-any LCProp distributions are supported")
        if met['Name'] != 'lcprop' or not met['Version']:
            _fail("invalid LCProp distribution metadata")
        if "src/lcprop/__init__.py" not in payload:
            _fail("missing LCProp package")
    except (KeyError, ValueError, UnicodeError) as exc:
        _fail(f"invalid distribution metadata: {exc}")
    return InstalledArtifact(met['Name'], met['Version'], tuple(sorted(payload.items())))


# Standalone stdlib-only verifier: runs before importing deployed scientific code.
# Manifest digest is supplied independently by the request/launcher, not trusted from disk.
VERIFY_PROGRAM = r'''
import hashlib,json,pathlib,sys
class ArtifactIdentityError(RuntimeError):
    pass
def require(condition,message='invalid artifact'):
    if not condition: raise ArtifactIdentityError(message)
root=pathlib.Path(_artifact_root if '_artifact_root' in globals() else sys.argv[1]).resolve()
expected=_artifact_digest if '_artifact_digest' in globals() else sys.argv[2]
m=json.loads((root/'.lcprop-artifact.json').read_text())
encoded=json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
require(hashlib.sha256(encoded).hexdigest()==expected, 'artifact manifest identity mismatch')
require(m['artifact_version']==1 and m['source_kind']=='installed_distribution')
names=set()
for item in m['files']:
    name=item['path']; parts=pathlib.PurePosixPath(name).parts
    require(parts and parts[0]=='src' and '..' not in parts and not name.startswith('/'))
    require(name not in names); names.add(name)
    p=root
    for part in parts:
        p=p/part
        require(not p.is_symlink(), 'symlink payload')
    require(p.is_file() and p.stat().st_size==item['bytes'], 'payload size mismatch')
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    require(digest.hexdigest()==item['sha256'], 'payload checksum mismatch')
actual=set()
for p in (root/'src').rglob('*'):
    require(not p.is_symlink(), 'symlink payload')
    if p.is_file(): actual.add(p.relative_to(root).as_posix())
require(actual==names, 'unexpected payload files')
if not globals().get('_quiet',False): print(expected)
'''

# Only a completed identity assertion emits this protocol signal. SSH, shell,
# Python startup, I/O and unexpected verifier errors remain execution failures.
_IDENTITY_FAILURE_MARKER = "LCPROP_ARTIFACT_IDENTITY_FAILURE: "
_IDENTITY_FAILURE_EXIT = 65
REMOTE_VERIFY_PROGRAM = (
    f"try:\n    exec({VERIFY_PROGRAM!r})\n"
    "except ArtifactIdentityError as exc:\n"
    f"    print({_IDENTITY_FAILURE_MARKER!r} + str(exc), file=sys.stderr)\n"
    f"    sys.exit({_IDENTITY_FAILURE_EXIT})\n"
)


# -I -S prevents PYTHONPATH, user site, .pth and sitecustomize from selecting code
# before verification. Add dependency directories as data paths, without executing
# startup hooks. run_module retains the executor's __main__/CLI semantics.
BOOTSTRAP_PROGRAM = "_quiet=True\n" + VERIFY_PROGRAM + r'''
import sysconfig,runpy
sys.dont_write_bytecode=True
sys.pycache_prefix=None
prefix=pathlib.Path(sys.prefix)
venv=pathlib.Path(sys.executable).absolute().parent.parent
cfg=venv/'pyvenv.cfg'
include_base=False
if cfg.is_file():
    prefix=venv
    settings=dict(line.split('=',1) for line in cfg.read_text().splitlines() if '=' in line)
    include_base=any(k.strip()=='include-system-site-packages' and v.strip().lower()=='true'
                     for k,v in settings.items())
for base in ([prefix,pathlib.Path(sys.base_prefix)] if include_base else [prefix]):
    paths=sysconfig.get_paths(vars={'base':str(base),'platbase':str(base)})
    for key in ('purelib','platlib'):
        if paths[key] not in sys.path: sys.path.append(paths[key])
sys.path.insert(0,str(root/'src'))
import lcprop
require(pathlib.Path(lcprop.__file__).resolve()==root/'src/lcprop/__init__.py',
        'wrong LCProp import origin')
sys._lcprop_source_binding=(str(root),expected)
'''

LAUNCH_PROGRAM = BOOTSTRAP_PROGRAM + r'''
sys.argv=['lcprop.transport.executor','--run-dir',sys.argv[3]]
runpy.run_module('lcprop.transport.executor',run_name='__main__',alter_sys=True)
'''


def verify_execution_binding(source):
    """Fail closed immediately before dispatch for an installed-artifact request."""
    import sys

    bound = getattr(sys, "_lcprop_source_binding", None)
    is_distribution = source.get("source_kind") == "installed_distribution"
    if not is_distribution and bound is None:
        return  # Existing Git/local execution remains unchanged.
    if not is_distribution or bound is None:
        _fail("installed execution requires its submission-bound isolated launcher")
    root, digest = bound
    root = Path(root)
    if (digest != source.get("source_content_sha256")
            or Path(source["remote_source_path"]).resolve() != root):
        _fail("execution artifact differs from submission identity")
    namespace = {"_artifact_root": str(root), "_artifact_digest": digest, "_quiet": True}
    exec(VERIFY_PROGRAM, namespace)
    owned = {str(root / item["path"]) for item in namespace["m"]["files"]}
    for name, module in tuple(sys.modules.items()):
        if name == "lcprop" or name.startswith("lcprop."):
            origin = getattr(module, "__file__", None)
            if origin is None or str(Path(origin).resolve()) not in owned:
                _fail(f"unowned executing LCProp module: {name}")
            paths = getattr(module, "__path__", ())
            if any(Path(path).resolve() != Path(origin).resolve().parent for path in paths):
                _fail(f"contaminated LCProp package search path: {name}")


class InstalledDeploymentManager(SourceDeploymentManager):
    def __init__(self, *, package_file, remote_python, distributions=None, **kwargs):
        super().__init__(local_source=Path(package_file).parent, **kwargs)
        self.package_file = Path(package_file)
        self.remote_python = remote_python
        self.distributions = distributions

    def preflight(self):
        return SourceSubmissionBinding(
            "distribution", resolve_installed_distribution(self.package_file, self.distributions)
        )

    def _snapshot_exists_checked(self, path):
        """Only a quiet POSIX test false result establishes snapshot absence."""
        try:
            self._transport.ssh(self.host, "test", "-e", path)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1 and not exc.stderr and not exc.stdout:
                return False
            raise
        return True

    def _verify_distribution(self, remote_snapshot, digest):
        verifier_started = False
        try:
            if not self._snapshot_exists_checked(remote_snapshot):
                return False
            verifier_started = True
            result = self._transport.ssh(self.host, self.remote_python, "-I", "-S", "-c",
                                         REMOTE_VERIFY_PROGRAM, remote_snapshot, digest)
            if result.strip() != digest:
                raise ValueError("unexpected verification response")
        except Exception as exc:
            stderr = getattr(exc, "stderr", None) or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            identity_failure = (
                verifier_started
                and isinstance(exc, subprocess.CalledProcessError)
                and exc.returncode == _IDENTITY_FAILURE_EXIT
                and any(line.startswith(_IDENTITY_FAILURE_MARKER) for line in stderr.splitlines())
            )
            category = "snapshot_identity_mismatch" if identity_failure else "snapshot_verification_failed"
            detail = str(exc) + (f"\nstderr:\n{stderr}" if stderr else "")
            raise SourceDeploymentError(category, detail) from exc
        return True

    def resolve_or_stage(self, *, binding=None):
        current = self.preflight()
        if binding is not None and current != binding:
            raise SourceDeploymentError("source_changed", "installed product changed after preflight")
        artifact = current.identity
        verify = lambda path: self._verify_distribution(path, artifact.digest)
        remote = f"{self.source_root}/dist-sha256-{artifact.digest}"
        with tempfile.TemporaryDirectory(prefix="lcprop-distribution-") as directory:
            local = Path(directory)
            checksum = artifact.archive(local / "source.tar")
            (local / ".lcprop-artifact.json").write_text(_json(artifact.manifest) + "\n")
            reused = verify(remote)
            if not reused:
                reused = self._stage(archive=local / "source.tar", checksum=checksum,
                                     git_sha="", remote_snapshot=remote, marker_directory=local,
                                     marker_names=(".lcprop-artifact.json",), verify_snapshot=verify)
                if not verify(remote):
                    raise SourceDeploymentError("snapshot_finalize_failed", "snapshot disappeared")
        provenance = dict(source_kind="installed_distribution", source_git_sha=None,
                          source_content_sha256=artifact.digest, source_checksum_sha256=checksum,
                          distribution_name=artifact.name, distribution_version=artifact.version,
                          remote_source_path=remote, snapshot_reused=reused, source_marker_version=2)
        return ResolvedSourceDeployment("installed_distribution", None, remote, checksum, reused, provenance)


def _owns_active_package_tree(package, directory):
    """Require one physical package tree in both effective and declared paths."""
    try:
        directory = Path(directory).resolve(strict=True)
        active = Path(package.__file__)
        spec = package.__spec__
        paths = tuple(package.__path__)
        declared = tuple(spec.submodule_search_locations)
        initializer = (directory / "__init__.py").resolve(strict=True)
        return (
            active.parent.resolve(strict=True) == directory
            and active.resolve(strict=True) == initializer
            and Path(spec.origin).resolve(strict=True) == initializer
            and bool(paths) and bool(declared)
            and all(Path(path).resolve(strict=True) == directory for path in (*paths, *declared))
        )
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError):
        return False


def automatic_deployment_manager(*, host, source_root, remote_python, transport=None):
    import lcprop
    active = Path(lcprop.__file__).absolute()
    # A supported checkout wins even if stray distribution metadata claims it.
    # Selecting a provider must never turn a Git validation error into fallback.
    try:
        probe = subprocess.run(
            ["git", "-C", str(active.parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        probe = None
    if probe is not None and probe.returncode == 0:
        root = Path(probe.stdout.strip()).resolve()
        if _owns_active_package_tree(lcprop, root / "src/lcprop"):
            return SourceDeploymentManager(host=host, source_root=source_root,
                                           local_source=active.parent, transport=transport)
    owner = owning_distribution(active)
    if owner is not None and _owns_active_package_tree(lcprop, active.parent):
        return InstalledDeploymentManager(host=host, source_root=source_root,
                                          package_file=active, remote_python=remote_python,
                                          transport=transport)
    _fail("active LCProp import is neither the supported Git source tree nor a RECORD-owned installation "
          "with an unambiguous package search path")
