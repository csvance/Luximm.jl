"""Shared helpers for Luximm.jl parity dump sidecars.

The `timm-to-lux` skill (skills/timm-to-lux/SKILL.md) documents how a port
uses these helpers.

Each dump script writes a single HDF5 file containing:

    /input              dataset, PyTorch tensor (any shape)
    /output             dataset OR group with one dataset per named output
    /state_dict/<key>   dataset per PyTorch state_dict entry

The Julia side reads these via Luximm.Interop.read_parity.
"""

from __future__ import annotations

import os
from typing import Mapping

import h5py
import numpy as np
import torch


# Batch size of every fixture input. Two distinct samples, so a Julia forward that
# mixes the batch axis into tokens, heads or a reduction fails parity instead of
# matching exactly as it would at batch 1.
PARITY_BATCH = 2


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().contiguous().float().numpy()


def dump(
    path: str,
    *,
    inp: torch.Tensor,
    state_dict: Mapping[str, torch.Tensor],
    out,
) -> None:
    """Write input, state_dict, and one or more named outputs to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("input", data=to_numpy(inp))
        sd = f.create_group("state_dict")
        for k, v in state_dict.items():
            sd.create_dataset(k, data=to_numpy(v))
        if isinstance(out, torch.Tensor):
            f.create_dataset("output", data=to_numpy(out))
        elif isinstance(out, Mapping):
            og = f.create_group("output")
            for k, v in out.items():
                og.create_dataset(k, data=to_numpy(v))
        else:
            raise TypeError(f"unexpected output type: {type(out)}")
