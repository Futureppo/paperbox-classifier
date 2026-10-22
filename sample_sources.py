"""Copy a small, traceable selection of raw frames. No ML or labeling occurs."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Folder containing YYYY-MM-DD subdirectories of raw JPEG images')
    parser.add_argument('--out', type=Path, default=Path('source_frames'))
    parser.add_argument('--per-day', type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists; choose a new folder to preserve source provenance.')
    if args.per_day < 1:
        parser.error('per-day must be positive.')
    root = args.source_dir.resolve()
    groups = defaultdict(list)
    for path in root.rglob('*.jpg'):
        relative = path.relative_to(root)
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', relative.parts[0]):
            groups[relative.parts[0]].append(path)
    if not groups:
        parser.error(f'No dated raw JPGs found: {root}')
    rows, seen_hashes = [], set()
    args.out.mkdir(parents=True)
    for group, paths in sorted(groups.items()):
        paths.sort()
        count = min(args.per_day, len(paths))
        indices = [round(i * (len(paths) - 1) / max(1, count - 1)) for i in range(count)]
        for index in indices:
            path = paths[index]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            relative = Path(group) / f'{index:04d}_{path.name}'
            destination = args.out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            rows.append({'path': relative.as_posix(), 'group': group,
                         'sha256': digest})
    with (args.out / 'index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'group', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    report = {'source_jpeg_count': sum(map(len, groups.values())),
              'source_counts_by_date': {g: len(p) for g, p in sorted(groups.items())},
              'copied_count': len(rows), 'requested_per_day': args.per_day,
              'method': 'Evenly spaced sorted file positions within each date; exact duplicate bytes removed.',
              'labels': 'Unannotated. This selection does not guarantee class balance or independent objects.'}
    (args.out / 'sampling_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
