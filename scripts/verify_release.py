"""Verify published assets and cross-check the reference results, without torch."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_data import verify_sources


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    inventory = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for item in inventory['files']:
        path = (ROOT / item['path']).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f'Missing or invalid file: {item["path"]}')
        data = path.read_bytes()
        if item['hash_mode'] == 'text-lf':
            data = data.replace(b'\r\n', b'\n')
        require(hashlib.sha256(data).hexdigest() == item['sha256'], f'File changed: {item["path"]}')
    source = verify_sources(ROOT / 'source_frames')
    require(source['source_images'] == 84 and source['collection_groups'] == 7, 'Unexpected source counts')
    require(all((ROOT / 'reference_labels' / f'{i}.jpg').is_file() for i in range(6)), 'Missing reference image')
    run = ROOT / 'results/experiment01'
    config = json.loads((run / 'config.json').read_text())
    metrics = json.loads((run / 'test_report/metrics.json').read_text())
    weights = json.loads((run / 'weights.json').read_text())
    history = list(csv.DictReader((run / 'history.csv').open(encoding='utf-8-sig')))
    predictions = list(csv.DictReader((run / 'test_report/predictions.csv').open(encoding='utf-8-sig')))
    require([sum(config['counts'][s].values()) for s in ('train', 'val', 'test')] == [24, 22, 6], 'Unexpected split counts')
    require(len(history) == 20 and len(predictions) == 6, 'Unexpected history/test length')
    best = max(history, key=lambda row: float(row['val_accuracy']))
    require(int(best['epoch']) == metrics['selected_epoch'] == weights['selected_epoch'] == 15, 'Best epoch mismatch')
    require(abs(float(best['val_accuracy']) - 18 / 22) < 1e-12, 'Validation accuracy mismatch')
    require(config['dataset_fingerprint'] == metrics['dataset_fingerprint'] == weights['dataset_fingerprint'], 'Fingerprint mismatch')
    matrix = [[0] * 6 for _ in range(6)]
    correct = 0
    for row in predictions:
        truth, predicted = int(row['truth']), int(row['predicted'])
        matrix[truth][predicted] += 1
        require((row['correct'] == 'True') == (truth == predicted), 'Correct flag mismatch')
        scores = [float(row[f'score_{i}']) for i in range(6)]
        require(abs(sum(scores) - 1) < 1e-5 and max(range(6), key=scores.__getitem__) == predicted, 'Invalid prediction scores')
        correct += truth == predicted
    require(matrix == metrics['confusion_matrix'], 'Confusion matrix mismatch')
    require(metrics['sample_count'] == 6 and metrics['accuracy'] == correct / 6 == 1, 'Test result mismatch')
    require(metrics['macro_f1'] == 1 and all(p['support'] == 1 and p['f1'] == 1 for p in metrics['per_class']), 'F1/support mismatch')
    print(f'PASS: {len(inventory["files"])} published assets verified; 84 source images; reference records agree.')
    print('This checks file integrity and recorded results. It does not rerun training or the unpublished test set.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f'Release check failed: {error}') from error
