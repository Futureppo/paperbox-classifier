"""Predict one manually cropped carton with local trained weights."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from common import CLASSES, choose_device, image_transform, load_checkpoint


def predict_image(model, image, device):
    tensor = image_transform(False)(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        values = model(tensor).softmax(1)[0].cpu().tolist()
    return {label: values[i] for i, label in enumerate(CLASSES)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/experiment01/best.pt'))
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists():
        parser.error('JSON output already exists; choose a new file.')
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        scores = predict_image(model, image, device)
    result = {'image': str(args.image), 'class': max(scores, key=scores.get), 'scores': scores,
              'checkpoint_epoch': checkpoint['epoch'],
              'note': 'Closed-set scores, not calibrated probabilities or robot grasp decisions.'}
    content = json.dumps(result, indent=2, ensure_ascii=False)
    print(content)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
