from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def clamp_unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).clamp(0.0, 1.0)


def ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image must have shape [batch, height, width, 3].")
    if min(image.shape[0], image.shape[1], image.shape[2]) < 1:
        raise ValueError("image must contain at least one pixel.")
    return clamp_unit_interval(image)


def ensure_mask_batch(mask: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a torch.Tensor.")

    normalized = mask.to(dtype=torch.float32)
    if normalized.ndim == 2:
        normalized = normalized.unsqueeze(0)
    elif normalized.ndim == 4 and normalized.shape[1] == 1:
        normalized = normalized.squeeze(1)
    elif normalized.ndim == 4 and normalized.shape[-1] == 1:
        normalized = normalized.squeeze(-1)
    elif normalized.ndim != 3:
        raise ValueError("mask must have shape [height, width], [batch, height, width], or a single-channel 4D shape.")

    if normalized.shape[-2:] != (height, width):
        raise ValueError("mask height and width must match the image tensor.")

    # Mini title: enforce per-frame alpha hints
    if normalized.shape[0] != batch_size:
        raise ValueError("mask batch dimension must match the image batch exactly.")

    return clamp_unit_interval(normalized)


def batch_to_numpy(batch: torch.Tensor) -> list[np.ndarray]:
    return [frame.detach().cpu().numpy().astype(np.float32) for frame in batch]


def stack_rgb_frames(frames: Iterable[np.ndarray]) -> torch.Tensor:
    materialized = [np.clip(frame.astype(np.float32), 0.0, 1.0) for frame in frames]
    if not materialized:
        raise ValueError("No RGB frames were produced.")
    return torch.from_numpy(np.stack(materialized, axis=0))


def stack_mask_frames(frames: Iterable[np.ndarray]) -> torch.Tensor:
    materialized: list[np.ndarray] = []
    for frame in frames:
        working = frame.astype(np.float32)
        if working.ndim == 3 and working.shape[-1] == 1:
            working = working[:, :, 0]
        materialized.append(np.clip(working, 0.0, 1.0))
    if not materialized:
        raise ValueError("No mask frames were produced.")
    return torch.from_numpy(np.stack(materialized, axis=0))
