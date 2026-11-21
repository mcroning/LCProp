"""Verify/extract a supplied capability kit for local tests; never install/run it."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, zipfile
from stage_pr_capabilities import ENGINE_NAME, ENGINE_SHA


def prepare(kit, expected_sha256, target):
    blob=Path(kit).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected_sha256:
        raise ValueError('Setup kit checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names=archive.namelist();pins=json.loads(archive.read('PINS.json'))
        if len(names)!=len(set(names)) or set(names)!=set(pins['members'])|{'PINS.json'}:
            raise ValueError('Unexpected setup members')
        for name,digest in pins['members'].items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Invalid setup member: '+name)
        wheel=archive.read(ENGINE_NAME)
    if hashlib.sha256(wheel).hexdigest()!=ENGINE_SHA:
        raise ValueError('Engine wheel does not match ce915366')
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for info in archive.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Invalid wheel path')
            if info.is_dir():continue
            destination=target.joinpath(*name.parts)
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(info))
    return {'engine_import_path':str(target.resolve()),'engine_sha256':ENGINE_SHA,'kit_sha256':expected_sha256}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.kit,args.expected_sha256,args.target),indent=2))
