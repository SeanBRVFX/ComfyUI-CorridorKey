from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def clamp_unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).clamp(0.0, 1.0)


def convert_gamma_space(image: torch.Tensor, gamma_space: str) -> torch.Tensor:
    normalized = clamp_unit_interval(image)
    if gamma_space == "Linear":
        return normalized.clamp_min(0.0).pow(1.0 / 2.2)
    return normalized


def ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image must have shape [batch, height, width, 3].")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("image must contain at least one pixel.")
    return clamp_unit_interval(image)


def ensure_mask_batch(
    mask: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a torch.Tensor.")

    normalized = mask.to(dtype=torch.float32)

    if normalized.ndim == 2:
        normalized = normalized.unsqueeze(0)
    elif normalized.ndim == 3:
        pass
    elif normalized.ndim == 4 and normalized.shape[1] == 1:
        normalized = normalized.squeeze(1)
    elif normalized.ndim == 4 and normalized.shape[-1] == 1:
        normalized = normalized.squeeze(-1)
    else:
        raise ValueError("mask must have shape [height, width], [batch, height, width], or a single-channel 4D shape.")

    if normalized.ndim != 3:
        raise ValueError("mask normalization failed to produce a [batch, height, width] tensor.")
    if normalized.shape[-2:] != (height, width):
        raise ValueError("mask height and width must match the image tensor.")

    if normalized.shape[0] == 1 and batch_size > 1:
        normalized = normalized.repeat(batch_size, 1, 1)
    elif normalized.shape[0] != batch_size:
        raise ValueError("mask batch dimension must match the image batch or be a single mask.")

    return clamp_unit_interval(normalized)


def gaussian_blur_map(mask_batch: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return clamp_unit_interval(mask_batch)

    sigma = max(radius / 2.0, 0.5)
    positions = torch.arange(-radius, radius + 1, device=mask_batch.device, dtype=mask_batch.dtype)
    kernel_1d = torch.exp(-(positions**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)

    working = mask_batch.unsqueeze(1)
    working = F.pad(working, (radius, radius, 0, 0), mode="replicate")
    working = F.conv2d(working, kernel_x)
    working = F.pad(working, (0, 0, radius, radius), mode="replicate")
    working = F.conv2d(working, kernel_y)
    return clamp_unit_interval(working.squeeze(1))


def despeckle_binary_mask(mask_batch: torch.Tensor, minimum_size: int) -> torch.Tensor:
    if minimum_size <= 1:
        return clamp_unit_interval(mask_batch)

    binary = (clamp_unit_interval(mask_batch) >= 0.5).to(dtype=torch.uint8)
    output = binary.clone()
    cpu_masks = binary.detach().cpu().numpy()

    for batch_index, cpu_mask in enumerate(cpu_masks):
        keep_mask = np.zeros_like(cpu_mask, dtype=np.uint8)
        visited = np.zeros_like(cpu_mask, dtype=bool)
        height, width = cpu_mask.shape

        for y_coord in range(height):
            for x_coord in range(width):
                if cpu_mask[y_coord, x_coord] == 0 or visited[y_coord, x_coord]:
                    continue

                queue: deque[tuple[int, int]] = deque([(y_coord, x_coord)])
                component: list[tuple[int, int]] = []
                visited[y_coord, x_coord] = True

                while queue:
                    current_y, current_x = queue.popleft()
                    component.append((current_y, current_x))

                    for next_y, next_x in (
                        (current_y - 1, current_x),
                        (current_y + 1, current_x),
                        (current_y, current_x - 1),
                        (current_y, current_x + 1),
                    ):
                        if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                            continue
                        if visited[next_y, next_x] or cpu_mask[next_y, next_x] == 0:
                            continue
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))

                if len(component) >= minimum_size:
                    for keep_y, keep_x in component:
                        keep_mask[keep_y, keep_x] = 1

        output[batch_index] = torch.from_numpy(keep_mask).to(device=mask_batch.device)

    return output.to(dtype=mask_batch.dtype)


def compute_edge_map(image: torch.Tensor) -> torch.Tensor:
    luminance = (
        (image[..., 0] * 0.299)
        + (image[..., 1] * 0.587)
        + (image[..., 2] * 0.114)
    ).unsqueeze(1)

    sobel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = image.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)

    padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, sobel_x)
    grad_y = F.conv2d(padded, sobel_y)
    magnitude = torch.sqrt((grad_x.square() + grad_y.square()).clamp_min(1e-12)).squeeze(1)

    batch_max = magnitude.flatten(1).max(dim=1).values.clamp_min(1e-6).view(-1, 1, 1)
    return clamp_unit_interval(magnitude / batch_max)


def mask_to_image(mask_batch: torch.Tensor) -> torch.Tensor:
    grayscale = clamp_unit_interval(mask_batch).unsqueeze(-1)
    return grayscale.repeat(1, 1, 1, 3)


def build_checkerboard(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    tile_size: int = 8,
) -> torch.Tensor:
    y_index = torch.arange(height, device=device)
    x_index = torch.arange(width, device=device)
    grid_y = (y_index[:, None] // tile_size) % 2
    grid_x = (x_index[None, :] // tile_size) % 2
    board = ((grid_x + grid_y) % 2).to(dtype=dtype)
    board = (board * 0.30) + 0.35
    board = board.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 3)
    return clamp_unit_interval(board)
