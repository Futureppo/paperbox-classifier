"""Shared model/data functions. Transfer learning follows the PyTorch tutorial.

See NOTICE.md and LICENSE for provenance. All paths are local and portable.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import datasets, models, transforms

from dataset_utils import CLASSES, sha256, verify_dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SquarePad:
    """Keep the whole rectangular carton face instead of center-cropping it."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image.convert('RGB'), (self.size, self.size),
                            method=Image.Resampling.BILINEAR, color=(127, 127, 127))


def image_transform(training=False):
    operations = []
    if training:
        operations += [transforms.RandomRotation(180, expand=True, fill=(127, 127, 127)),
                       transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.10)]
    operations += [SquarePad(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(operations)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run check_environment.py --require-cuda.')
    return torch.device(name)


def build_model(pretrained=False, pretrained_file=None):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained and not pretrained_file else None
    model = models.resnet18(weights=weights)
    if pretrained_file:
        state = torch.load(pretrained_file, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_checkpoint(path, device):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {path}. Train your own model with train.py, '
            'or download the reference weight from the GitHub Release. See docs/WEIGHTS.md.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'resnet18' or checkpoint.get('classes') != CLASSES:
        raise ValueError('Expected a six-class ResNet18 checkpoint produced by this project.')
    if checkpoint.get('preprocess') != 'square_pad_224_imagenet_v1':
        raise ValueError('Checkpoint preprocessing does not match this project.')
    model = build_model()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def dataset_for(root, split, training=False):
    dataset = datasets.ImageFolder(Path(root) / split, transform=image_transform(training))
    if dataset.classes != CLASSES:
        raise ValueError(f'{split} must contain all six classes 0..5; found {dataset.classes}')
    counts = np.bincount(dataset.targets, minlength=6)
    if not np.all(counts > 0):
        raise ValueError(f'Empty classes in {split}.')
    return dataset


def classification_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(6), where=matrix.sum(0) != 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(6), where=matrix.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(6), where=(precision + recall) != 0)
    return {'accuracy': float(tp.sum() / max(1, matrix.sum())),
            'macro_f1': float(f1.mean()), 'sample_count': int(matrix.sum()),
            'per_class': [{'label': str(i), 'support': int(matrix[i].sum()),
                           'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i])} for i in range(6)],
            'confusion_matrix': matrix.tolist()}
