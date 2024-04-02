"""
Utils for PyTorch
"""

import pickle

from io import BytesIO
from typing import BinaryIO

import torch


def determine_device() -> str:
    """
    Returns device on which generation should run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def patched_torch_unpickle(file_handle: BinaryIO, device: str) -> object:
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-777059657

    unpickler = _PyTorchPatchedUnpickler(file_handle, map_location=device)
    return unpickler.load()


class _PyTorchPatchedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location: str, **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return _load_with_map_location(self._map_location)
        else:
            return super().find_class(module, name)


def _load_with_map_location(map_location: str) -> callable:
    return lambda b: torch.load(BytesIO(b), map_location=map_location)
