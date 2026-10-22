"""Dataset integrity checks that work without PyTorch or a GPU."""
import hashlib
import json
from collections import Counter
from pathlib import Path

CLASSES = [str(i) for i in range(6)]
SPLITS = ('train', 'val', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(root):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing {manifest_path}. Run annotate.py and prepare_data.py first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('classes') != CLASSES or manifest.get('split_method') != 'group_disjoint':
        raise ValueError('Expected a six-class, group-disjoint dataset from prepare_data.py.')
    records = manifest.get('samples', [])
    if not records:
        raise ValueError('Dataset is empty.')
    group_split, hash_split, expected_files = {}, {}, set()
    counts = {split: Counter() for split in SPLITS}
    for record in records:
        if record['split'] not in SPLITS or record['label'] not in CLASSES:
            raise ValueError('Invalid split or class in manifest.')
        if not isinstance(record['group'], str) or not record['group'].strip():
            raise ValueError('Every sample must have a nonempty collection group.')
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid dataset path: {record["path"]}')
        parts = Path(record['path']).parts
        if len(parts) != 3 or parts[:2] != (record['split'], record['label']):
            raise ValueError('Manifest label/split does not agree with image directory.')
        if path in expected_files:
            raise ValueError('Duplicate sample path in manifest.')
        digest = sha256(path)
        if digest != record['sha256']:
            raise ValueError(f'Image changed after data preparation: {record["path"]}')
        for lookup, key in [(group_split, record['group']), (hash_split, digest)]:
            if key in lookup and lookup[key] != record['split']:
                raise ValueError('Group or identical-image leakage between dataset splits.')
            lookup[key] = record['split']
        expected_files.add(path)
        counts[record['split']][record['label']] += 1
    actual_files = {p.resolve() for split in SPLITS
                    for p in (root / split).rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise ValueError('Dataset image folders differ from manifest. Rebuild the dataset.')
    for split in SPLITS:
        if set(counts[split]) != set(CLASSES):
            raise ValueError(f'{split} must contain all six classes 0..5.')
        if dict(counts[split]) != manifest.get('counts', {}).get(split):
            raise ValueError(f'Manifest counts do not match samples in {split}.')
    if group_split != manifest.get('group_assignments'):
        raise ValueError('Group assignments do not match manifest samples.')
    # Keep the original fingerprint format so existing checkpoints still match.
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True,
                                           ensure_ascii=False).encode('utf-8')).hexdigest()
    return manifest, fingerprint
