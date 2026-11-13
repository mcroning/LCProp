"""Bounded upload/download of existing LCProp file formats."""
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import errno
import os
import shutil
import sys
import tempfile
import zipfile

MAX_UPLOAD_BYTES = 128 * 1024 * 1024
PACKAGE_FILES = {
    "request/READY.json", "request/request.json", "request/request_arrays.npz",
    "output/RESULT_READY.json", "output/result.json", "output/result_arrays.npz",
    "request/manifest.json", "request/checksums.sha256",
    "output/manifest.json", "output/checksums.sha256",
}


def selected_upload(value):
    """Normalize ipywidgets 7 filename mappings and 8 upload records."""
    if not value or len(value) != 1:
        raise ValueError("Select one experiment JSON or completed ZIP")
    if isinstance(value, Mapping):
        name, entry = next(iter(value.items()))
    else:
        entry = value[0]
        name = entry["name"]
    content = entry["content"]
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds 128 MiB")
    return name, bytes(content)


def package_sizes(members):
    """Check serialized and aggregate expanded array bytes before decoding."""
    packed = sum(path.stat().st_size for path in members)
    if packed > MAX_UPLOAD_BYTES:
        raise ValueError("Expanded ZIP members exceed 128 MiB")
    expanded = 0
    for path in members:
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as arrays:
                expanded += sum(item.file_size for item in arrays.infolist())
    if expanded > MAX_UPLOAD_BYTES:
        raise ValueError("Combined expanded array payload exceeds 128 MiB")
    return packed, expanded


def unpack_completed(upload, directory):
    """Allow only the transport package files, into a fresh directory."""
    from io import BytesIO

    if len(upload) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds 128 MiB")
    with zipfile.ZipFile(BytesIO(upload)) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if (len(set(names)) != len(names) or not set(names) <= PACKAGE_FILES
                or sum(entry.file_size for entry in members) > MAX_UPLOAD_BYTES):
            raise ValueError("Unexpected, duplicate or oversized package members")
        directory = Path(tempfile.mkdtemp(prefix="reopen-", dir=directory))
        try:
            paths = []
            for entry in members:
                path = directory.joinpath(*PurePosixPath(entry.filename).parts)
                path.parent.mkdir(exist_ok=True)
                with archive.open(entry) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                paths.append(path)
            package_sizes(paths)
        except BaseException:
            shutil.rmtree(directory)
            raise
    return directory


def archive_completed(directory):
    directory = Path(directory)
    destination = directory.with_suffix(".zip")
    members = [directory / name for name in sorted(PACKAGE_FILES)
               if (directory / name).is_file()]
    package_sizes(members)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(directory))
    if destination.stat().st_size > MAX_UPLOAD_BYTES:
        destination.unlink()
        raise ValueError("Completed ZIP exceeds 128 MiB")
    return destination


def save_verified_copy(source, folder):
    """Copy a staged artifact without overwriting any existing destination."""
    from lcprop.transport.artifacts import sha256_file

    source, folder = Path(source), Path(folder)
    destination = folder / source.name
    if shutil.disk_usage(folder).free < source.stat().st_size:
        raise OSError("Destination has insufficient available storage")
    expected = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix=".lcprop-copy-", dir=folder) as staging:
        staged = Path(staging) / source.name
        with source.open("rb") as input_stream, staged.open("xb") as output:
            shutil.copyfileobj(input_stream, output)
        if sha256_file(staged) != expected:
            raise OSError("Drive copy checksum mismatch")
        _promote_verified_copy(staged, destination)
    return destination


def _promote_verified_copy(staged, destination):
    """Publish complete bytes atomically without replacing an existing name."""
    unsupported = (errno.EOPNOTSUPP, errno.ENOSYS, errno.EPERM, errno.EACCES)
    try:
        os.link(staged, destination)
        return
    except OSError as error:
        if error.errno not in unsupported:
            raise
        if sys.platform != "linux":
            raise OSError("Destination does not support atomic no-overwrite promotion") from error

    # Some Linux mounts support exclusive rename but do not support hard links.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError("Destination does not support atomic no-overwrite promotion")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(staged), at_fdcwd,
              os.fsencode(destination), rename_noreplace) != 0:
        code = ctypes.get_errno()
        if code in (*unsupported, errno.EINVAL):
            raise OSError("Destination does not support atomic no-overwrite promotion")
        raise OSError(code, os.strerror(code), destination)
