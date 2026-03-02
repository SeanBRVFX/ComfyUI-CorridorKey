from __future__ import annotations

import logging
from typing import Callable

import torch

from .config import CorridorKeySettings
from .engine import get_cached_engine
from .tensor_ops import (
    batch_to_numpy,
    ensure_image_tensor,
    ensure_mask_batch,
    stack_mask_frames,
    stack_rgb_frames,
)

LOGGER = logging.getLogger(__name__)


class CorridorKeyProcessor:
    def __init__(self, device: str | None = None) -> None:
        self._device = device

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: CorridorKeySettings,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(settings, CorridorKeySettings):
            raise ValueError("settings must be a CorridorKeySettings instance.")

        image_batch = ensure_image_tensor(image)
        mask_batch = ensure_mask_batch(
            mask=mask,
            batch_size=image_batch.shape[0],
            height=image_batch.shape[1],
            width=image_batch.shape[2],
        )

        total_frames = int(image_batch.shape[0])
        if progress_callback is not None:
            progress_callback("Loading CorridorKey model...", 0, total_frames)

        engine = get_cached_engine(device=self._device)

        fg_frames: list = []
        matte_frames: list = []
        processed_frames: list = []
        comp_frames: list = []

        image_frames = batch_to_numpy(image_batch)
        mask_frames = batch_to_numpy(mask_batch)

        for frame_index, (image_frame, mask_frame) in enumerate(zip(image_frames, mask_frames, strict=True), start=1):
            if progress_callback is not None:
                progress_callback(f"Processing frame {frame_index}/{total_frames}...", frame_index - 1, total_frames)

            result = engine.process_frame(
                image=image_frame,
                mask_linear=mask_frame,
                refiner_scale=settings.refiner_strength,
                input_is_linear=settings.input_is_linear,
                despill_strength=settings.despill_strength,
                auto_despeckle=settings.despeckle_enabled,
                despeckle_size=settings.despeckle_size,
            )
            fg_frames.append(result["fg"])
            matte_frames.append(result["matte"])
            processed_frames.append(result["processed"])
            comp_frames.append(result["comp"])
            LOGGER.debug("CorridorKey processed frame %s", frame_index)

        if progress_callback is not None:
            progress_callback("CorridorKey complete.", total_frames, total_frames)

        return (
            stack_rgb_frames(fg_frames),
            stack_mask_frames(matte_frames),
            stack_rgb_frames(processed_frames),
            stack_rgb_frames(comp_frames),
        )
