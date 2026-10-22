"""Run on the target notebook before training; does not download models."""
import argparse
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    import torch
    import torchvision
    print('Python:', sys.version.split()[0], '| OS:', platform.platform())
    print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
    print('CUDA runtime:', torch.version.cuda, '| available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print('GPU:', device.name, '| VRAM GiB:', round(device.total_memory / 1024**3, 2))
        x = torch.ones((32, 32), device='cuda')
        print('GPU operation:', float((x @ x).mean()))
    elif args.require_cuda:
        raise SystemExit('CUDA unavailable. Check NVIDIA driver and CUDA PyTorch installation.')
    print('Environment check finished. No model has been trained.')


if __name__ == '__main__':
    main()
