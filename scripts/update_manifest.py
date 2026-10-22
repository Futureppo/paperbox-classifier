"""Regenerate checksums after intentionally changing bundled data or result assets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOTS = ('source_frames', 'reference_labels', 'results', 'docs/images')
TEXT_SUFFIXES = {'.csv', '.json', '.md', '.txt'}


def main():
    files = []
    for folder in ASSET_ROOTS:
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file():
                continue
            mode = 'text-lf' if path.suffix.lower() in TEXT_SUFFIXES else 'binary'
            data = path.read_bytes()
            if mode == 'text-lf':
                data = data.replace(b'\r\n', b'\n')
            files.append({'path': path.relative_to(ROOT).as_posix(), 'hash_mode': mode,
                          'sha256': hashlib.sha256(data).hexdigest()})
    payload = {'schema_version': 1, 'scope': list(ASSET_ROOTS),
               'text_hashing': 'UTF-8 file bytes with CRLF normalized to LF; BOM preserved.',
               'files': files}
    (ROOT / 'release_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Updated manifest for {len(files)} bundled assets. Review changes before committing.')


if __name__ == '__main__':
    main()
