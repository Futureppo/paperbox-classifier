"""Two-stage ResNet18 transfer learning. Run only on the target notebook."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import CLASSES, build_model, choose_device, dataset_for, seed_everything, verify_dataset


def plot_history(history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'val'):
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r[f'{split}_loss'] for r in history], label=split)
        axes[1].plot(epochs, [r[f'{split}_accuracy'] for r in history], label=split)
    for ax, title in zip(axes, ('Cross-entropy loss', 'Accuracy')):
        ax.set(xlabel='Epoch', title=title)
        ax.legend()
        ax.grid(alpha=.2)
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, default=Path('runs/experiment01'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--freeze-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--finetune-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--pretrained-file', type=Path, help='Optional offline official ResNet18 ImageNet state_dict')
    args = parser.parse_args()
    if args.epochs < 1 or args.freeze_epochs < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error('Epochs/batch-size must be positive; freeze-epochs/workers must be nonnegative.')
    if args.out.exists():
        parser.error('Output exists. Choose a new --out directory; previous experiments are preserved.')
    device = choose_device(args.device)
    seed_everything(args.seed)
    manifest, fingerprint = verify_dataset(args.data)
    datasets = {s: dataset_for(args.data, s, training=(s == 'train')) for s in ('train', 'val')}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=(s == 'train'),
                            num_workers=args.workers, pin_memory=(device.type == 'cuda'),
                            generator=generator) for s in datasets}
    model = build_model(pretrained=True, pretrained_file=args.pretrained_file).to(device)
    criterion = nn.CrossEntropyLoss()
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    args.out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch_version=str(torch.__version__), device=str(device), dataset_fingerprint=fingerprint,
                  counts=manifest['counts'])
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    history, best_accuracy, previous_stage, optimizer = [], -1.0, None, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        stage = 'head' if frozen else 'finetune'
        if stage != previous_stage:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = not frozen or name.startswith('fc.')
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=args.head_lr if frozen else args.finetune_lr, weight_decay=1e-4)
            previous_stage = stage
        row = {'epoch': epoch, 'stage': stage}
        for split in ('train', 'val'):
            training = split == 'train'
            model.train(training)
            if training and frozen:
                # Keep pretrained BatchNorm statistics fixed during head-only training.
                model.eval()
                model.fc.train()
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in loaders[split]:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = model(images)
                        loss = criterion(logits, labels)
                    if training:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
            row[f'{split}_loss'] = total_loss / total
            row[f'{split}_accuracy'] = correct / total
        row['elapsed_seconds'] = round(time.perf_counter() - start, 2)
        history.append(row)
        checkpoint = {'architecture': 'resnet18', 'classes': CLASSES,
                      'preprocess': 'square_pad_224_imagenet_v1', 'model': model.state_dict(),
                      'epoch': epoch, 'val_accuracy': row['val_accuracy'],
                      'dataset_fingerprint': fingerprint, 'config': config}
        torch.save(checkpoint, args.out / 'last.pt')
        if row['val_accuracy'] > best_accuracy:
            best_accuracy = row['val_accuracy']
            torch.save(checkpoint, args.out / 'best.pt')
        with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        plot_history(history, args.out / 'curves.png')
        print(f'Epoch {epoch:02d}/{args.epochs} {stage:8s} | train {row["train_accuracy"]:.3f}'
              f' | val {row["val_accuracy"]:.3f} | best val {best_accuracy:.3f}', flush=True)
    print(f'Saved {args.out / "best.pt"}. Test images were not used for model selection.')


if __name__ == '__main__':
    main()
