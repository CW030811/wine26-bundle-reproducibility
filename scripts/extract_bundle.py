#!/usr/bin/env python3
"""Verify an experiment archive, then extract without replacing different files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def extract_bundle(root, entry, check_only=False):
    root = Path(root).resolve()
    archive = root / entry['file']
    if digest(archive.read_bytes()) != entry['sha256']:
        raise ValueError('Archive SHA-256 mismatch: ' + entry['file'])
    expected = entry['files']
    payloads = {}
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle:
            name = member.name
            path = PurePosixPath(name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in name or name in payloads or name not in expected):
                raise ValueError('Unsafe, duplicate, or unlisted member: ' + name)
            destination = root / name
            if root not in destination.resolve().parents:
                raise ValueError('Destination escapes repository: ' + name)
            content = handle.extractfile(member).read()
            if digest(content) != expected[name]:
                raise ValueError('Member SHA-256 mismatch: ' + name)
            payloads[name] = content
    if set(payloads) != set(expected):
        raise ValueError('Archive inventory differs from manifest')
    if check_only:
        return len(payloads)
    # Check every conflict before creating any file.
    for name, content in payloads.items():
        destination = root / name
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != content):
            raise ValueError('Preserving different existing file: ' + name)
    for name, content in payloads.items():
        destination = root / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(content)
    return len(payloads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', help='table2, table5, table6, table7, figure9, figure10, figure11, or all')
    parser.add_argument('--check-only', action='store_true', help='Validate archives without extracting')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'bundles/manifest.json').read_text())
    names = list(manifest['archives']) if args.experiment == 'all' else [args.experiment]
    for name in names:
        if name not in manifest['archives']:
            parser.error('No verified bundle is published for ' + name)
        count = extract_bundle(ROOT, manifest['archives'][name], args.check_only)
        print(f'{name}: {count} files verified' + ('' if args.check_only else ' and extracted'))


if __name__ == '__main__':
    main()
