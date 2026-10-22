import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_utils import CLASSES, sha256, verify_dataset
from check_data import verify_sources

ROOT = Path(__file__).resolve().parents[1]


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        samples = []
        for split in ('train', 'val', 'test'):
            for label in CLASSES:
                path = self.root / split / label / 'sample.jpg'
                path.parent.mkdir(parents=True)
                path.write_bytes((split + label).encode())
                samples.append({'path': path.relative_to(self.root).as_posix(), 'split': split,
                                'label': label, 'group': 'group-' + split, 'sha256': sha256(path)})
        self.manifest = {'classes': CLASSES, 'split_method': 'group_disjoint', 'samples': samples,
                         'counts': {s: {c: 1 for c in CLASSES} for s in ('train', 'val', 'test')},
                         'group_assignments': {'group-' + s: s for s in ('train', 'val', 'test')}}
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_original_fingerprint_format_is_preserved(self):
        _, actual = verify_dataset(self.root)
        expected = hashlib.sha256(json.dumps(self.manifest['samples'], sort_keys=True,
                                             ensure_ascii=False).encode('utf-8')).hexdigest()
        self.assertEqual(actual, expected)

    def test_rejects_modified_image(self):
        (self.root / 'train/0/sample.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            verify_dataset(self.root)

    def test_rejects_same_group_across_splits(self):
        self.manifest['samples'][6]['group'] = 'group-train'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            verify_dataset(self.root)

    def test_rejects_identical_image_across_splits(self):
        data = (self.root / 'train/0/sample.jpg').read_bytes()
        target = self.root / 'val/0/sample.jpg'
        target.write_bytes(data)
        self.manifest['samples'][6]['sha256'] = sha256(target)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            verify_dataset(self.root)

    def test_rejects_unlisted_file(self):
        (self.root / 'train/0/extra.jpg').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'differ from manifest'):
            verify_dataset(self.root)

    def test_rejects_duplicate_manifest_path(self):
        self.manifest['samples'].append(self.manifest['samples'][0].copy())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate sample path'):
            verify_dataset(self.root)

    def test_rejects_incorrect_count(self):
        self.manifest['counts']['train']['0'] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'counts do not match'):
            verify_dataset(self.root)

    def test_rejects_incorrect_group_assignment(self):
        self.manifest['group_assignments']['group-train'] = 'test'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Group assignments'):
            verify_dataset(self.root)

    def test_rejects_path_outside_dataset(self):
        self.manifest['samples'][0]['path'] = '../outside.jpg'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Invalid dataset path'):
            verify_dataset(self.root)


class PreparationTests(unittest.TestCase):
    def test_preparation_keeps_groups_disjoint_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / 'annotations.csv'
            rows = []
            for group in range(4):
                for label in range(6):
                    path = root / f'g{group}_{label}.jpg'
                    Image.new('RGB', (40, 40), (group * 50, label * 40, 150)).save(path)
                    rows.append({'path': path.name, 'label': str(label), 'group': f'g{group}',
                                 'source': f'g{group}/raw.jpg', 'box': '0,0,40,40'})
            with annotations.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            out = root / 'prepared'
            command = [sys.executable, str(ROOT / 'prepare_data.py'), '--annotations', str(annotations), '--out', str(out)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            manifest, fingerprint = verify_dataset(out)
            self.assertEqual(len(manifest['samples']), 24)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(verify_dataset(out)[1], fingerprint)
            # A class present in just two groups must fail before creating output.
            remaining = [r for r in rows if r['label'] != '0' or r['group'] in ('g0', 'g1')]
            with annotations.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(remaining)
            failed_out = root / 'too-few-groups'
            failed = subprocess.run(command[:-1] + [str(failed_out)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('at least 3', failed.stderr + failed.stdout)
            self.assertFalse(failed_out.exists())

    def test_source_index_detects_missing_or_changed_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'sample.jpg'
            Image.new('RGB', (40, 40), (30, 40, 50)).save(path)
            with (root / 'index.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path', 'group', 'sha256'])
                writer.writeheader(); writer.writerow({'path': path.name, 'group': 'g1', 'sha256': sha256(path)})
            self.assertEqual(verify_sources(root)['source_images'], 1)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                verify_sources(root)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                verify_sources(root)


if __name__ == '__main__':
    unittest.main()
