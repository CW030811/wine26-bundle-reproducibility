import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


class BundleTests(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/extract_bundle.py'
        self.assertTrue(script.exists(), 'Verified bundle extraction is not implemented')
        spec = importlib.util.spec_from_file_location('extract_bundle', script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, names=None, symlink=False):
        names = names or {'data/example.txt': b'verified'}
        archive = self.root / 'example.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            for name, content in names.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = '../escape'
                handle.addfile(info, io.BytesIO(content))
        return {'file': archive.name,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in names.items()}}

    def test_round_trip_and_idempotence(self):
        entry = self.bundle()
        self.module.extract_bundle(self.root, entry)
        self.assertEqual((self.root / 'data/example.txt').read_bytes(), b'verified')
        self.module.extract_bundle(self.root, entry)

    def test_rejects_changed_archive(self):
        entry = self.bundle()
        entry['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.module.extract_bundle(self.root, entry)
        self.assertFalse((self.root / 'data').exists())

    def test_rejects_traversal_before_writing(self):
        entry = self.bundle({'data/good.txt': b'a', '../escape': b'b'})
        with self.assertRaises(ValueError):
            self.module.extract_bundle(self.root, entry)
        self.assertFalse((self.root / 'data').exists())

    def test_rejects_symlink(self):
        with self.assertRaises(ValueError):
            self.module.extract_bundle(self.root, self.bundle(symlink=True))

    def test_preserves_conflicting_existing_file(self):
        entry = self.bundle()
        target = self.root / 'data/example.txt'
        target.parent.mkdir()
        target.write_bytes(b'user result')
        with self.assertRaises(ValueError):
            self.module.extract_bundle(self.root, entry)
        self.assertEqual(target.read_bytes(), b'user result')

    def test_rejects_unlisted_archive_file(self):
        entry = self.bundle()
        entry['files'] = {}
        with self.assertRaises(ValueError):
            self.module.extract_bundle(self.root, entry)


if __name__ == '__main__':
    unittest.main()
