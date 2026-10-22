"""Create ImageFolder train/val/test directories with group-disjoint splits."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = [str(i) for i in range(6)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, default=Path('data/annotations.csv'))
    parser.add_argument('--out', type=Path, default=Path('data/prepared'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f'{args.out} already exists. Use a new output directory to preserve prior splits.')
    with args.annotations.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit('No annotated crops. Run annotate.py first.')
    groups, seen_paths, seen_hashes = defaultdict(list), set(), {}
    for row in rows:
        if row.get('label') not in CLASSES or not row.get('group', '').strip():
            raise SystemExit('Every row needs label 0..5 and a nonempty group.')
        path = (args.annotations.parent / row['path']).resolve()
        if not path.is_relative_to(args.annotations.parent.resolve()) or not path.is_file():
            raise SystemExit(f'Invalid crop path: {path}')
        if path.suffix.lower() not in ('.jpg', '.jpeg') or path.read_bytes()[:3] != b'\xff\xd8\xff':
            raise SystemExit(f'Expected a JPEG crop from annotate.py: {path}')
        if path in seen_paths:
            raise SystemExit(f'Duplicate annotation path: {path}')
        seen_paths.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise SystemExit(f'Identical crop files: {path} and {seen_hashes[digest]}. Remove a duplicate annotation.')
        seen_hashes[digest] = str(path)
        row = dict(row, absolute_path=str(path), sha256=digest)
        groups[row['group']].append(row)
    for label in CLASSES:
        count = sum(any(r['label'] == label for r in group) for group in groups.values())
        if count < 3:
            raise SystemExit(f'Class {label} occurs in only {count} groups. Annotate it in at least 3 independent groups.')
    names = sorted(groups)
    n_val = max(1, round(len(names) * .15))
    n_test = max(1, round(len(names) * .15))
    rng, best = random.Random(args.seed), None
    total = Counter(row['label'] for group in groups.values() for row in group)
    ratios = {'train': .70, 'val': .15, 'test': .15}
    for _ in range(3000):
        order = names.copy()
        rng.shuffle(order)
        mapping = {name: ('val' if i < n_val else 'test' if i < n_val + n_test else 'train')
                   for i, name in enumerate(order)}
        counts = {split: Counter() for split in ratios}
        for group, samples in groups.items():
            counts[mapping[group]].update(r['label'] for r in samples)
        if any(counts[split][c] == 0 for split in ratios for c in CLASSES):
            continue
        score = sum(abs(counts[s][c] / total[c] - ratios[s]) for s in ratios for c in CLASSES)
        if best is None or score < best[0]:
            best = (score, mapping, counts)
    if best is None:
        raise SystemExit('Cannot obtain six-class coverage in all group-disjoint splits. Add diverse groups; do not split neighboring frames randomly.')
    args.out.mkdir(parents=True)
    samples = []
    for group in sorted(groups):
        split = best[1][group]
        for row in sorted(groups[group], key=lambda r: r['path']):
            filename = f'{len(samples):05d}_{row["sha256"][:12]}.jpg'
            relative = Path(split) / row['label'] / filename
            destination = args.out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['absolute_path'], destination)
            samples.append({'path': relative.as_posix(), 'label': row['label'], 'group': group,
                            'split': split, 'sha256': row['sha256'],
                            'source': row.get('source', ''), 'box': row.get('box', '')})
    manifest = {'seed': args.seed, 'classes': CLASSES, 'split_method': 'group_disjoint',
                'target_ratios': ratios, 'group_assignments': best[1],
                'counts': {s: dict(best[2][s]) for s in ratios}, 'samples': samples}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=2))
    print(f'Prepared {len(samples)} crops. Inspect manifest.json and class coverage before training.')


if __name__ == '__main__':
    main()
