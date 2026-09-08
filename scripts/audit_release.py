#!/usr/bin/env python3
"""Check release hashes, safe archives and structured metadata for local identifiers."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import subprocess
import tarfile
import zipfile
import msgpack

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r'/(?:Users|home)/[^/\s]+/|/(?:private/)?tmp/[\w.-]+|(?<![\w])[A-Za-z]:[\\/]'
    r'|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    r'|(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'
    r'|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:License' r'ID)\s*[:=]\s*\d+'
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, bytes):
        try:
            yield value.decode('utf-8')
        except UnicodeDecodeError:
            pass
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def metadata_strings(name, content):
    suffix = Path(name).suffix
    if suffix == '.msgpack':
        yield from strings(msgpack.unpackb(content, raw=False, strict_map_key=False, object_pairs_hook=list))
    elif suffix == '.pt':
        with zipfile.ZipFile(io.BytesIO(content)) as checkpoint:
            for member in checkpoint.namelist():
                if member.endswith('.pkl'):
                    for _, argument, _ in pickletools.genops(checkpoint.read(member)):
                        if isinstance(argument, str):
                            yield argument
    elif suffix in {'.py', '.md', '.json', '.csv', '.tex', '.txt', '.toml', '.lock', '.yml', '.yaml'}:
        yield content.decode('utf-8')


def release_files():
    parents = ['src', 'scripts', 'tests', 'docs', 'experiments', 'provenance', 'bundles', '.github']
    paths = [path for parent in parents for path in (ROOT / parent).rglob('*')
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    paths += [ROOT / name for name in ['README.md', 'CONTRIBUTING.md', 'EXPERIMENTS.json',
                                     'pyproject.toml', 'uv.lock', '.gitignore']]
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Maintainer: refresh audit and whole-release checksums')
    args = parser.parse_args()
    from extract_bundle import extract_bundle
    manifest = json.loads((ROOT / 'bundles/manifest.json').read_text())
    findings, members, checkpoints = [], 0, 0
    for name, entry in manifest['archives'].items():
        extract_bundle(ROOT, entry, check_only=True)
        with tarfile.open(ROOT / entry['file']) as archive:
            for member in archive:
                data = archive.extractfile(member).read()
                if any(PATTERN.search(text) for text in metadata_strings(member.name, data)):
                    findings.append(name + ':' + member.name)
                members += 1
                checkpoints += member.name.endswith('.pt')
    for path in release_files():
        if path.suffix == '.gz':
            continue
        if any(PATTERN.search(text) for text in metadata_strings(path.name, path.read_bytes())):
            findings.append(path.relative_to(ROOT).as_posix())
    if findings:
        raise ValueError('Metadata review required: ' + repr(sorted(set(findings))))
    report = {'passed': True, 'archives': len(manifest['archives']), 'archive_members_checked': members,
              'checkpoint_containers_checked': checkpoints, 'findings': [],
              'scope': 'Hashes, archive paths and decoded metadata strings; no pickle execution',
              'identity_policy': 'Personal home paths, emails, license IDs and credentials excluded. Public GitHub owner identity retained.'}
    if args.refresh:
        (ROOT / 'provenance/RELEASE_AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
        lines = [digest(path.read_bytes()) + '  ' + path.relative_to(ROOT).as_posix() for path in release_files()]
        (ROOT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    else:
        hashes = {}
        for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            if relative in hashes:
                raise ValueError('Duplicate checksum entry: ' + relative)
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or digest(path.read_bytes()) != expected:
                raise ValueError('Release checksum mismatch: ' + relative)
            hashes[relative] = expected
        if set(hashes) != {p.relative_to(ROOT).as_posix() for p in release_files()}:
            raise ValueError('Release checksum inventory mismatch')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
