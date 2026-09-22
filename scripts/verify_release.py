"""Offline structural/integrity checks. No GPU, network, or input mutation."""
import argparse
import ast
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from artifact_policy import has_dataset_text

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv', 'build', '__pycache__', '.pytest_cache', '.cache'}


def _excluded(path):
    return bool(EXCLUDE.intersection(path.relative_to(ROOT).parts)) or path.suffix == '.pyc'


def _raise_walk_error(error):
    raise error


def files(*, release_package=False):
    """Ignore checkout metadata only at the root; never traverse its .git entry."""
    found = []
    for current, directories, filenames in os.walk(
        ROOT, topdown=True, followlinks=False, onerror=_raise_walk_error
    ):
        current = Path(current)
        for name in [*directories, *filenames]:
            if name != '.git':
                continue
            path = current / name
            if current == ROOT and not release_package:
                if name in directories:
                    directories.remove(name)
                continue
            raise AssertionError((str(path.relative_to(ROOT)), 'Git metadata is prohibited'))
        # Inspect excluded runtime directories for nested Git metadata too.
        # Other directory symlinks are never followed and cannot enter the release.
        for name in directories:
            path = current / name
            if path.is_symlink() and not _excluded(path):
                raise AssertionError((str(path.relative_to(ROOT)), 'symlink is prohibited'))
        for name in filenames:
            path = current / name
            if current == ROOT and name == '.git' and not release_package:
                continue
            if not _excluded(path) and (path.is_file() or path.is_symlink()):
                found.append(path)
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--release-package', action='store_true',
        help='Validate a publication package: reject all .git entries, including the root. '
             'The default allows only the root .git directory or gitfile of a checkout.'
    )
    args = parser.parse_args(argv)
    inventory = json.loads((ROOT / 'MANIFEST.json').read_text())
    observed = {
        str(path.relative_to(ROOT)): path
        for path in files(release_package=args.release_package)
        if path.name != 'MANIFEST.json'
    }
    expected = {record['path']: record for record in inventory['files']}
    assert observed.keys() == expected.keys(), {
        'missing': sorted(expected.keys() - observed.keys()),
        'extra': sorted(observed.keys() - expected.keys())
    }
    counts = {'files': 0, 'python_sources': 0, 'json_files': 0, 'gzip_files': 0}
    forbidden = re.compile(
        r'/(?:Users|Volumes|workspace|home)/|/(?:opt/RouteVerify|dev/shm/routeverify|private/tmp/routeverify|var/tmp/routeverify)'
        r'|https?://(?:git\.)?overleaf\.com/'
        r'|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,})'
        r'|-----BEGIN [A-Z ]*PRIVATE KEY-----'
    )
    for name, path in observed.items():
        assert not path.is_symlink() and '.git' not in path.relative_to(ROOT).parts, name
        raw = path.read_bytes()
        record = expected[name]
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], name
        assert len(raw) < 100 * 1024 * 1024, (name, 'file exceeds 100 MiB')
        assert path.suffix not in {'.safetensors', '.pt', '.pth', '.bin'}, (name, 'weight-like extension')
        counts['files'] += 1
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
            counts['gzip_files'] += 1
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
        assert not forbidden.search(text), (name, 'private-path, private-URL or credential pattern')
        if path.suffix == '.py':
            ast.parse(text, filename=name)
            counts['python_sources'] += 1
        if name.endswith(('.json', '.json.gz')):
            assert not has_dataset_text(json.loads(text)), (name, 'dataset rendered prompt text')
            counts['json_files'] += 1
        if name.endswith(('.jsonl', '.jsonl.gz')):
            for line in text.splitlines():
                if line.strip():
                    assert not has_dataset_text(json.loads(line)), (name, 'dataset rendered prompt text')
    print(json.dumps({
        'status': 'PASS', **counts,
        'mode': 'release-package' if args.release_package else 'checkout',
        'scope': 'public hashes, file set, Git metadata policy, compressed/text formats, Python syntax, '
                 'dataset-text fields, private-path/URL/credential pattern scan; numerical validation receipts are separate',
        'GPU_forwards': 0
    }, indent=2))


if __name__ == '__main__':
    main()
