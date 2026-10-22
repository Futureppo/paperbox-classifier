"""Check bundled source frames or a prepared split without installing PyTorch."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dataset_utils import sha256, verify_dataset


def verify_sources(root):
    root = Path(root).resolve()
    index = root / 'index.csv'
    if not index.is_file():
        raise ValueError(f'Missing source index: {index}')
    with index.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError('Source index is empty.')
    seen, hashes, counts = set(), set(), Counter()
    for row in records:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path in seen:
            raise ValueError(f'Missing, duplicate, or invalid source path: {row["path"]}')
        digest = sha256(path)
        if digest != row['sha256'] or digest in hashes:
            raise ValueError(f'Changed or duplicate source image: {row["path"]}')
        if not row['group'].strip():
            raise ValueError('Every source image needs a collection group.')
        seen.add(path)
        hashes.add(digest)
        counts[row['group']] += 1
    actual = {p.resolve() for p in root.rglob('*')
              if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    if seen != actual:
        raise ValueError('Source image files differ from index.csv.')
    return {'source_images': len(records), 'collection_groups': len(counts),
            'groups': dict(sorted(counts.items())), 'sha256_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--sources', type=Path, help='Check source_frames/index.csv and image hashes')
    target.add_argument('--data', type=Path, help='Check prepared/manifest.json, hashes, counts and splits')
    args = parser.parse_args()
    try:
        if args.sources:
            result = verify_sources(args.sources)
        else:
            manifest, fingerprint = verify_dataset(args.data)
            result = {'counts': manifest['counts'], 'sample_count': len(manifest['samples']),
                      'dataset_fingerprint': fingerprint, 'group_disjoint': True}
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Data check failed: {error}\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
