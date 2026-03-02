from __future__ import annotations

import logging

import torch

from .config import CorridorKeySettings
from .tensor_ops import (
    build_checkerboard,
    clamp_unit_interval,
    compute_edge_map,
    convert_gamma_space,
    despeckle_binary_mask,
    ensure_image_tensor,
    ensure_mask_batch,
    gaussian_blur_map,
    mask_to_image,
)

LOGGER = logging.getLogger(__name__)


class CorridorKeyProcessor:
    # Mini title: lightweight green-spill suppression for edge pixels.
    def _apply_despill(self, image: torch.Tensor, alpha: torch.Tensor, strength: float) -> torch.Tensor:
        if strength <= 0.0:
            return image

        edge_weight = 1.0 - (alpha.mul(2.0).sub(1.0).abs())
        rb_max = torch.maximum(image[..., 0], image[..., 2])
        spill = torch.relu(image[..., 1] - rb_max)
        reduction = spill * strength * (0.25 + (0.75 * edge_weight))

        red = clamp_unit_interval(image[..., 0] + (reduction * 0.10))
        green = clamp_unit_interval(image[..., 1] - reduction)
        blue = clamp_unit_interval(image[..., 2] + (reduction * 0.10))
        return torch.stack((red, green, blue), dim=-1)

    # Mini title: choose the user-requested primary preview.
    def _select_output(
        self,
        output_mode: str,
        alpha: torch.Tensor,
        foreground: torch.Tensor,
        processed: torch.Tensor,
        comp: torch.Tensor,
    ) -> torch.Tensor:
        if output_mode == "FG":
            return foreground
        if output_mode == "Matte":
            return mask_to_image(alpha)
        if output_mode == "Comp":
            return comp
        return processed

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: CorridorKeySettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(settings, CorridorKeySettings):
            raise ValueError("settings must be a CorridorKeySettings instance.")

        image_batch = ensure_image_tensor(image)
        working_image = convert_gamma_space(image_batch, settings.gamma_space)
        mask_batch = ensure_mask_batch(
            mask=mask,
            batch_size=working_image.shape[0],
            height=working_image.shape[1],
            width=working_image.shape[2],
        )

        # Mini title: compute the soft starting matte.
        softened_mask = gaussian_blur_map(mask_batch, settings.feather_radius)

        # Mini title: derive edge guidance from image luminance.
        edge_map = compute_edge_map(working_image)
        edge_gain = 1.0 + (edge_map * settings.edge_focus)

        centered_mask = (
            (softened_mask - settings.threshold)
            * settings.transition_gain
            * settings.refiner_strength
        )
        alpha = torch.sigmoid(centered_mask * edge_gain)

        low_lock = max(0.0, settings.threshold - settings.preserve_core)
        high_lock = min(1.0, settings.threshold + settings.preserve_core)
        alpha = torch.where(mask_batch <= low_lock, torch.zeros_like(alpha), alpha)
        alpha = torch.where(mask_batch >= high_lock, torch.ones_like(alpha), alpha)

        # Mini title: suppress tiny disconnected islands when requested.
        if settings.despeckle_enabled:
            support_mask = despeckle_binary_mask(
                (alpha >= settings.threshold).to(dtype=alpha.dtype),
                settings.despeckle_size,
            )
            alpha = torch.where(support_mask > 0.0, alpha, torch.zeros_like(alpha))

        alpha = clamp_unit_interval(alpha)

        # Mini title: build the foreground-style outputs.
        despilled_image = self._apply_despill(working_image, alpha, settings.despill_factor)
        visible_support = (alpha > 0.01).to(dtype=despilled_image.dtype).unsqueeze(-1)
        foreground = clamp_unit_interval(despilled_image * visible_support)

        alpha_rgb = alpha.unsqueeze(-1)
        processed = clamp_unit_interval(foreground * alpha_rgb)
        checkerboard = build_checkerboard(
            batch_size=working_image.shape[0],
            height=working_image.shape[1],
            width=working_image.shape[2],
            device=working_image.device,
            dtype=working_image.dtype,
        )
        comp = clamp_unit_interval(processed + (checkerboard * (1.0 - alpha_rgb)))
        selected_output = self._select_output(
            output_mode=settings.output_mode,
            alpha=alpha,
            foreground=foreground,
            processed=processed,
            comp=comp,
        )

        LOGGER.debug(
            "CorridorKey refinement complete for batch=%s height=%s width=%s",
            working_image.shape[0],
            working_image.shape[1],
            working_image.shape[2],
        )

        return alpha, selected_output, foreground, processed, comp
