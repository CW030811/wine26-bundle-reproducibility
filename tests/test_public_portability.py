from pathlib import Path
import re
import unittest


class PortabilityTests(unittest.TestCase):
    def test_sources_do_not_embed_machine_specific_paths(self):
        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/|/(?:private/)?tmp/[A-Za-z0-9_.-]+')
        bad = []
        for parent in ['src', 'scripts', 'docs', 'provenance']:
            for path in (root / parent).rglob('*'):
                if path.suffix in {'.py', '.md', '.json'} and pattern.search(path.read_text()):
                    bad.append(str(path.relative_to(root)))
        self.assertEqual(bad, [], 'Machine-specific paths must not be published')


if __name__ == '__main__':
    unittest.main()
