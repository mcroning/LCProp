"""Assemble rc6 frontend candidate using exact reviewed bytes; never install/run."""
from pathlib import Path
import argparse, hashlib, io, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
KIT = 'LCProp-PR-capabilities-setup.zip'
WHEEL = 'lcprop-0.5.0rc3-py3-none-any.whl'
NOTEBOOK = 'LCProp-PR-capabilities-0.5.0rc6-candidate.ipynb'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stage(input_kit, destination):
    pins = dict(line.split('  ', 1)[::-1] for line in
                (ROOT / 'delivery-v5/SHA256SUMS').read_text().splitlines())
    data = Path(input_kit).read_bytes()
    if digest(data) != pins[KIT]:
        raise ValueError('Choose the exact reviewed 65648819 setup ZIP; nothing staged')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read('PINS.json'))
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['members']) | {'PINS.json'}:
            raise ValueError('Unexpected setup members')
        for name, expected in manifest['members'].items():
            if digest(archive.read(name)) != expected:
                raise ValueError('Setup member checksum mismatch: ' + name)
            if name.startswith('frontend/'):
                if (ROOT / name).read_bytes() != archive.read(name):
                    raise ValueError('Reviewed setup does not match source: ' + name)
        wheel = archive.read(WHEEL)
    if digest(wheel) != pins[WHEEL]:
        raise ValueError('Engine identity mismatch')
    notebook = (ROOT / 'delivery-v5/LCProp-PR-capabilities.ipynb').read_bytes()
    if digest(notebook) != pins['LCProp-PR-capabilities.ipynb']:
        raise ValueError('Notebook identity mismatch')
    notes = (ROOT / 'docs/releases/0.5.0rc6.md').read_text()
    # Candidate notes ship flat; source-document relative links become root URLs.
    notes = notes.replace('(../../records/', '(https://github.com/mcroning/LCProp/blob/e96a79319819d905851dcba294aed6af316dd0ed/records/')
    notes = notes.replace('(0.5.0rc6/ACCEPTANCE.md)', '(https://github.com/mcroning/LCProp/pull/31)')
    payloads = {WHEEL: wheel, NOTEBOOK: notebook, KIT: data,
                'CANDIDATE-NOTES.md': notes.encode()}
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (out / name).write_bytes(content)
    hashes = {name: digest(content) for name, content in payloads.items()}
    (out / 'SHA256SUMS').write_text(''.join(h + '  ' + n + '\n' for n, h in hashes.items()))
    return {'assembly_version': '0.5.0rc6', 'engine_distribution_version': '0.5.0rc3',
            'source_base': 'e96a79319819d905851dcba294aed6af316dd0ed',
            'engine_unchanged': True, 'frontend_sha256': manifest['frontend_sha256'],
            'payload_sha256': hashes, 'unverified_interactions': ['browser dragging', 'internal plot scrolling'],
            'new_hosted_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-kit', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
