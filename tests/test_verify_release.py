"""Regression checks use small synthetic packages; they never initialize Git."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('verify_release', SCRIPTS / 'verify_release.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class VerifyReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='routeverify-verifier-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = mock.patch.object(VERIFIER, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        payload = b'Synthetic public release fixture.\n'
        (self.root / 'sample.txt').write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (self.root / 'MANIFEST.json').write_text(json.dumps({'files': [
            {'path': 'sample.txt', 'bytes': len(payload), 'sha256': digest}
        ]}))

    def verify(self, release_package=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            VERIFIER.main(['--release-package'] if release_package else [])
        return json.loads(output.getvalue())

    def add_git(self, parent, kind):
        parent.mkdir(parents=True, exist_ok=True)
        git = parent / '.git'
        if kind == 'directory':
            git.mkdir()
            (git / 'HEAD').write_text('ref: refs/heads/main\n')
        elif kind == 'empty-directory':
            git.mkdir()
        elif kind == 'gitfile':
            git.write_text('gitdir: fixture-metadata\n')
        elif kind == 'symlink':
            git.symlink_to('absent-fixture-metadata')
        else:
            raise ValueError(kind)
        return git

    def test_plain_package_passes_both_modes(self):
        self.assertEqual(self.verify()['mode'], 'checkout')
        self.assertEqual(self.verify(True)['mode'], 'release-package')

    def test_top_level_git_variants_only_pass_checkout_mode(self):
        for kind in ('directory', 'empty-directory', 'gitfile', 'symlink'):
            with self.subTest(kind=kind):
                git = self.add_git(self.root, kind)
                self.assertEqual(self.verify()['status'], 'PASS')
                with self.assertRaisesRegex(AssertionError, 'Git metadata is prohibited'):
                    self.verify(True)
                if git.is_dir():
                    for child in git.iterdir():
                        child.unlink()
                    git.rmdir()
                else:
                    git.unlink()

    def test_ignored_root_git_directory_is_not_traversed(self):
        git = self.add_git(self.root, 'directory')
        real_scandir = os.scandir

        def guarded_scandir(path):
            if Path(path) == git:
                raise AssertionError('Root .git was traversed')
            return real_scandir(path)

        with mock.patch.object(VERIFIER.os, 'scandir', side_effect=guarded_scandir):
            self.assertEqual(self.verify()['status'], 'PASS')

    def test_nested_git_variants_fail_both_modes(self):
        for parent_name in ('nested', '.cache/deeper'):
            for kind in ('directory', 'empty-directory', 'gitfile', 'symlink'):
                with self.subTest(parent=parent_name, kind=kind):
                    git = self.add_git(self.root / parent_name, kind)
                    for strict in (False, True):
                        with self.assertRaisesRegex(AssertionError, 'Git metadata is prohibited'):
                            self.verify(strict)
                    if git.is_dir():
                        for child in git.iterdir():
                            child.unlink()
                        git.rmdir()
                    else:
                        git.unlink()

    def test_unexpected_regular_file_still_fails_both_modes(self):
        (self.root / 'unexpected.txt').write_text('Extra file.\n')
        for strict in (False, True):
            with self.assertRaisesRegex(AssertionError, 'unexpected.txt'):
                self.verify(strict)

    def test_manifested_symlink_is_rejected(self):
        (self.root / 'sample.txt').unlink()
        (self.root / 'sample.txt').symlink_to('absent-fixture')
        with self.assertRaises(AssertionError):
            self.verify()

    def test_correctly_hashed_artifact_with_dataset_text_is_rejected(self):
        (self.root / 'sample.txt').unlink()
        text = json.dumps({'sample': {'rendered_prompt_text': 'Synthetic input text.'}}) + '\n'
        for suffix in ('.json', '.json.gz', '.jsonl', '.jsonl.gz'):
            with self.subTest(suffix=suffix):
                name = 'sample' + suffix
                payload = text.encode()
                if suffix.endswith('.gz'):
                    payload = gzip.compress(payload, mtime=0)
                (self.root / name).write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                (self.root / 'MANIFEST.json').write_text(json.dumps({'files': [
                    {'path': name, 'bytes': len(payload), 'sha256': digest}
                ]}))
                for strict in (False, True):
                    with self.assertRaisesRegex(AssertionError, 'dataset rendered prompt text'):
                        self.verify(strict)
                (self.root / name).unlink()


if __name__ == '__main__':
    unittest.main()
