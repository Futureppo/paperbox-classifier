"""Evaluate the selected checkpoint once on the held-out test partition."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import CLASSES, choose_device, classification_metrics, dataset_for, load_checkpoint, verify_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/experiment01/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, default=Path('runs/experiment01/test_report'))
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Report directory exists. Preserve it and choose another --out if an audit is needed.')
    if args.batch_size < 1:
        parser.error('batch-size must be positive.')
    device = choose_device(args.device)
    _, fingerprint = verify_dataset(args.data)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint['dataset_fingerprint'] != fingerprint:
        raise ValueError('Dataset changed since training. Use the original fixed data split.')
    dataset = dataset_for(args.data, 'test')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    matrix, rows, offset = np.zeros((6, 6), dtype=np.int64), [], 0
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images.to(device)).softmax(1).cpu()
            predictions = scores.argmax(1)
            for truth, predicted, values in zip(labels.tolist(), predictions.tolist(), scores.tolist()):
                matrix[truth, predicted] += 1
                relative = Path(dataset.samples[offset][0]).resolve().relative_to(args.data.resolve()).as_posix()
                rows.append({'path': relative, 'truth': str(truth), 'predicted': str(predicted),
                             'correct': truth == predicted, **{f'score_{c}': values[i] for i, c in enumerate(CLASSES)}})
                offset += 1
    metrics = classification_metrics(matrix)
    metrics.update(dataset_fingerprint=fingerprint, checkpoint=str(args.checkpoint), selected_epoch=checkpoint['epoch'],
                   scope='Six-class single-carton image classification; not robot grasp success rate.')
    args.out.mkdir(parents=True)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    with (args.out / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(display, ax=ax)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='True class', title=f'Test confusion matrix (n={matrix.sum()})')
    fig.tight_layout()
    fig.savefig(args.out / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    print(f'Test n={metrics["sample_count"]}; accuracy={metrics["accuracy"]:.4f}; macro-F1={metrics["macro_f1"]:.4f}')
    print(f'Report saved in {args.out}. Report the sample count and split method alongside these results.')


if __name__ == '__main__':
    main()
