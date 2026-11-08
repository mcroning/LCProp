"""Bounded upload/download of existing LCProp file formats."""
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

MAX_UPLOAD_BYTES = 128 * 1024 * 1024
PACKAGE_FILES = {
    "request/READY.json", "request/request.json", "request/request_arrays.npz",
    "output/RESULT_READY.json", "output/result.json", "output/result_arrays.npz",
    "request/manifest.json", "request/checksums.sha256",
    "output/manifest.json", "output/checksums.sha256",
}


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
        for entry in members:
            path = directory.joinpath(*PurePosixPath(entry.filename).parts)
            path.parent.mkdir(exist_ok=True)
            with archive.open(entry) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output)
            if path.suffix == ".npz":
                with zipfile.ZipFile(path) as arrays:
                    if sum(item.file_size for item in arrays.infolist()) > MAX_UPLOAD_BYTES:
                        raise ValueError("Array payload exceeds 128 MiB")
    return directory


def archive_completed(directory):
    directory = Path(directory)
    destination = directory.with_suffix(".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(PACKAGE_FILES):
            path = directory / name
            if path.is_file():
                archive.write(path, name)
    return destination
