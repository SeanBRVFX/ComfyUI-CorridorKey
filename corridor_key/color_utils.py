from __future__ import annotations

import numpy as np
import torch


def _is_tensor(value: np.ndarray | torch.Tensor) -> bool:
    return isinstance(value, torch.Tensor)


def linear_to_srgb(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if _is_tensor(value):
        value = value.clamp(min=0.0)
        mask = value <= 0.0031308
        return torch.where(mask, value * 12.92, 1.055 * torch.pow(value, 1.0 / 2.4) - 0.055)

    value = np.clip(value, 0.0, None)
    mask = value <= 0.0031308
    return np.where(mask, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def srgb_to_linear(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if _is_tensor(value):
        value = value.clamp(min=0.0)
        mask = value <= 0.04045
        return torch.where(mask, value / 12.92, torch.pow((value + 0.055) / 1.055, 2.4))

    value = np.clip(value, 0.0, None)
    mask = value <= 0.04045
    return np.where(mask, value / 12.92, np.power((value + 0.055) / 1.055, 2.4))


def premultiply(
    foreground: np.ndarray | torch.Tensor,
    alpha: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    return foreground * alpha


def composite_straight(
    foreground: np.ndarray | torch.Tensor,
    background: np.ndarray | torch.Tensor,
    alpha: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    return foreground * alpha + background * (1.0 - alpha)


def composite_premul(
    foreground: np.ndarray | torch.Tensor,
    background: np.ndarray | torch.Tensor,
    alpha: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    return foreground + background * (1.0 - alpha)


def despill(
    image: np.ndarray | torch.Tensor,
    green_limit_mode: str = "average",
    strength: float = 1.0,
) -> np.ndarray | torch.Tensor:
    if strength <= 0.0:
        return image

    if _is_tensor(image):
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]

        limit = torch.max(red, blue) if green_limit_mode == "max" else (red + blue) / 2.0
        spill_amount = torch.clamp(green - limit, min=0.0)

        despilled = torch.stack(
            [
                red + (spill_amount * 0.5),
                green - spill_amount,
                blue + (spill_amount * 0.5),
            ],
            dim=-1,
        )
        if strength < 1.0:
            return image * (1.0 - strength) + despilled * strength
        return despilled

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]

    limit = np.maximum(red, blue) if green_limit_mode == "max" else (red + blue) / 2.0
    spill_amount = np.maximum(green - limit, 0.0)

    despilled_np = np.stack(
        [
            red + (spill_amount * 0.5),
            green - spill_amount,
            blue + (spill_amount * 0.5),
        ],
        axis=-1,
    )
    if strength < 1.0:
        return image * (1.0 - strength) + despilled_np * strength
    return despilled_np


def clean_matte(
    alpha_np: np.ndarray,
    area_threshold: int = 300,
    dilation: int = 15,
    blur_size: int = 5,
) -> np.ndarray:
    import cv2

    is_3d = alpha_np.ndim == 3
    working = alpha_np[:, :, 0] if is_3d else alpha_np

    mask_8u = (working > 0.5).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    cleaned_mask = np.zeros_like(mask_8u)
    for label_index in range(1, num_labels):
        if stats[label_index, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned_mask[labels == label_index] = 255

    if dilation > 0:
        kernel_size = int(dilation * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned_mask = cv2.dilate(cleaned_mask, kernel)

    if blur_size > 0:
        blur_kernel = int(blur_size * 2 + 1)
        cleaned_mask = cv2.GaussianBlur(cleaned_mask, (blur_kernel, blur_kernel), 0)

    safe_zone = cleaned_mask.astype(np.float32) / 255.0
    result = working * safe_zone
    if is_3d:
        return result[:, :, np.newaxis]
    return result


def create_checkerboard(
    width: int,
    height: int,
    checker_size: int = 64,
    color1: float = 0.2,
    color2: float = 0.4,
) -> np.ndarray:
    x_tiles = np.arange(width) // checker_size
    y_tiles = np.arange(height) // checker_size
    x_grid, y_grid = np.meshgrid(x_tiles, y_tiles)
    checker = (x_grid + y_grid) % 2
    background = np.where(checker == 0, color1, color2).astype(np.float32)
    return np.stack([background, background, background], axis=-1)
