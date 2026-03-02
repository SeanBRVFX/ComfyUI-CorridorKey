from __future__ import annotations

import pytest
import torch

from corridor_key import (
    CorridorKeyProcessor,
    CorridorKeySettings,
    is_verified_check_conclusions,
    select_latest_verified_commit,
)


def make_image(batch: int = 1, height: int = 6, width: int = 6) -> torch.Tensor:
    image = torch.zeros((batch, height, width, 3), dtype=torch.float32)
    image[..., : width // 2, :] = 0.15
    image[..., width // 2 :, :] = 0.85
    return image


def test_settings_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        CorridorKeySettings(threshold=1.2)


def test_settings_match_upstream_like_defaults() -> None:
    settings = CorridorKeySettings()

    assert settings.gamma_space == "sRGB"
    assert settings.despill_strength == 1.0
    assert settings.refiner_strength == 1.0
    assert settings.auto_despeckle == "On"
    assert settings.despeckle_size == 400
    assert settings.output_mode == "Processed"


def test_refine_broadcasts_single_mask_across_batch() -> None:
    processor = CorridorKeyProcessor()
    image = make_image(batch=2)
    mask = torch.zeros((6, 6), dtype=torch.float32)
    mask[:, 3:] = 1.0

    refined_mask, selected_output, foreground, processed, comp = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(auto_despeckle="Off"),
    )

    assert refined_mask.shape == (2, 6, 6)
    assert selected_output.shape == image.shape
    assert foreground.shape == image.shape
    assert processed.shape == image.shape
    assert comp.shape == image.shape
    assert torch.allclose(refined_mask[0], refined_mask[1])


def test_refine_preserves_full_foreground() -> None:
    processor = CorridorKeyProcessor()
    image = make_image()
    mask = torch.ones((1, 6, 6), dtype=torch.float32)

    refined_mask, selected_output, foreground, processed, comp = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(auto_despeckle="Off"),
    )

    assert torch.allclose(refined_mask, torch.ones_like(refined_mask))
    assert torch.allclose(foreground, image)
    assert torch.allclose(processed, image)
    assert torch.allclose(selected_output, processed)
    assert torch.allclose(comp, image)


def test_refine_preserves_full_background() -> None:
    processor = CorridorKeyProcessor()
    image = make_image()
    mask = torch.zeros((1, 6, 6), dtype=torch.float32)

    refined_mask, selected_output, foreground, processed, comp = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(auto_despeckle="Off"),
    )

    assert torch.allclose(refined_mask, torch.zeros_like(refined_mask))
    assert torch.allclose(foreground, torch.zeros_like(foreground))
    assert torch.allclose(processed, torch.zeros_like(processed))
    assert not torch.allclose(comp, torch.zeros_like(comp))
    assert torch.allclose(selected_output, processed)


def test_refine_rejects_wrong_mask_size() -> None:
    processor = CorridorKeyProcessor()
    image = make_image(height=6, width=6)
    mask = torch.zeros((1, 5, 6), dtype=torch.float32)

    with pytest.raises(ValueError, match="height and width"):
        processor.refine(
            image=image,
            mask=mask,
            settings=CorridorKeySettings(auto_despeckle="Off"),
        )


def test_output_mode_matte_returns_grayscale_preview() -> None:
    processor = CorridorKeyProcessor()
    image = make_image()
    mask = torch.zeros((1, 6, 6), dtype=torch.float32)
    mask[:, :, 3:] = 1.0

    refined_mask, selected_output, _, _, _ = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(output_mode="Matte", auto_despeckle="Off"),
    )

    assert torch.allclose(selected_output[..., 0], refined_mask)
    assert torch.allclose(selected_output[..., 1], refined_mask)
    assert torch.allclose(selected_output[..., 2], refined_mask)


def test_despill_reduces_green_bias() -> None:
    processor = CorridorKeyProcessor()
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    image[..., 0] = 0.1
    image[..., 1] = 0.9
    image[..., 2] = 0.1
    mask = torch.ones((1, 4, 4), dtype=torch.float32)

    _, _, foreground_without, _, _ = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(despill_strength=0.0, auto_despeckle="Off"),
    )
    _, _, foreground_with, _, _ = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(despill_strength=1.0, auto_despeckle="Off"),
    )

    assert torch.mean(foreground_with[..., 1]) < torch.mean(foreground_without[..., 1])


def test_auto_despeckle_can_remove_single_pixel_island() -> None:
    processor = CorridorKeyProcessor()
    image = make_image()
    mask = torch.zeros((1, 6, 6), dtype=torch.float32)
    mask[:, 2, 2] = 1.0

    refined_mask, _, _, _, _ = processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(
            auto_despeckle="On",
            despeckle_size=9,
            feather_radius=0,
            preserve_core=0.0,
        ),
    )

    assert float(refined_mask[0, 2, 2]) == 0.0


def test_verified_check_policy_requires_success_and_no_failure() -> None:
    assert is_verified_check_conclusions(("success", "cancelled"))
    assert not is_verified_check_conclusions(("success", "failure"))
    assert not is_verified_check_conclusions(("cancelled",))


def test_select_latest_verified_commit_picks_first_clean_candidate() -> None:
    commit_payloads = [
        {
            "sha": "newest-bad",
            "commit": {
                "author": {"date": "2026-03-02T00:00:00Z"},
                "message": "Newest bad commit",
            },
        },
        {
            "sha": "next-good",
            "commit": {
                "author": {"date": "2026-03-01T00:00:00Z"},
                "message": "Next good commit",
            },
        },
    ]
    check_payloads_by_sha = {
        "newest-bad": {
            "check_runs": [
                {"conclusion": "success"},
                {"conclusion": "failure"},
            ]
        },
        "next-good": {
            "check_runs": [
                {"conclusion": "success"},
                {"conclusion": "cancelled"},
            ]
        },
    }

    selected = select_latest_verified_commit(commit_payloads, check_payloads_by_sha)

    assert selected is not None
    assert selected.sha == "next-good"
