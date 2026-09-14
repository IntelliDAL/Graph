"""Resolve runtime devices and repository-relative storage locations."""

from pathlib import Path
import warnings

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path, device):
    """Load weights while filtering a known PyTorch legacy-storage warning.

    Args:
        path: Path to a state-dictionary checkpoint.
        device: Device onto which checkpoint tensors are mapped.

    Returns:
        The loaded state dictionary. Loading errors and other warnings propagate.
    """
    # PyTorch 2.1 may inspect TypedStorage while reading existing checkpoints.
    # Keep the filter local to loading and specific to this internal warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'TypedStorage is deprecated\..*',
            category=UserWarning,
            module=r'torch\._utils',
        )
        return torch.load(path, map_location=device, weights_only=True)


def resolve_device(value):
    """Resolve a CPU device or CUDA index, falling back to CPU without CUDA.

    Args:
        value: CPU, a CUDA device string, or a nonnegative GPU index.

    Returns:
        The device used for execution and checkpoint loading.
    """
    value = str(value)
    if value == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    index = int(value.split(':')[-1])
    if index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {index} is unavailable.')
    return torch.device(f'cuda:{index}')
