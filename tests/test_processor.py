from __future__ import annotations

import numpy as np
import pytest
import torch

from corridor_key import (
    CorridorKeyProcessor,
    CorridorKeySettings,
    is_verified_check_conclusions,
    select_latest_verified_commit,
)
from corridor_key.color_utils import composite_straight, linear_to_srgb, srgb_to_linear
from corridor_key.tensor_ops import ensure_image_tensor, ensure_mask_batch
from nodes import CorridorKey


def test_settings_match_upstream_defaults() -> None:
    settings = CorridorKeySettings()

    assert settings.gamma_space == "sRGB"
    assert settings.despill_strength == 1.0
    assert settings.refiner_strength == 1.0
    assert settings.auto_despeckle == "On"
    assert settings.despeckle_size == 400


def test_settings_reject_invalid_despill() -> None:
    with pytest.raises(ValueError, match="despill_strength"):
        CorridorKeySettings(despill_strength=1.5)


def test_srgb_linear_round_trip_is_stable() -> None:
    source = np.array([0.0, 0.018, 0.5, 1.0], dtype=np.float32)
    round_trip = linear_to_srgb(srgb_to_linear(source))
    assert np.allclose(round_trip, source, atol=1e-5)


def test_composite_straight_matches_expected_formula() -> None:
    fg = torch.tensor([[[0.8, 0.2, 0.0]]], dtype=torch.float32)
    bg = torch.tensor([[[0.2, 0.2, 0.2]]], dtype=torch.float32)
    alpha = torch.tensor([[[0.5]]], dtype=torch.float32)
    comp = composite_straight(fg, bg, alpha)
    expected = torch.tensor([[[0.5, 0.2, 0.1]]], dtype=torch.float32)
    assert torch.allclose(comp, expected)


def test_ensure_image_tensor_requires_bhwc() -> None:
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    assert ensure_image_tensor(image).shape == image.shape

    with pytest.raises(ValueError, match="shape"):
        ensure_image_tensor(torch.zeros((1, 3, 4, 4), dtype=torch.float32))


def test_ensure_mask_batch_rejects_mismatched_batch() -> None:
    mask = torch.ones((4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly"):
        ensure_mask_batch(mask, batch_size=2, height=4, width=4)


def test_node_metadata_populates_ui_info() -> None:
    input_types = CorridorKey.INPUT_TYPES()
    required_inputs = input_types["required"]

    assert "coarse" in required_inputs["mask"][1]["tooltip"].lower()
    assert "per frame" in required_inputs["mask"][1]["tooltip"].lower()
    assert input_types["hidden"]["unique_id"] == "UNIQUE_ID"
    assert CorridorKey.RETURN_NAMES == ("fg", "matte", "processed", "QC")
    assert len(CorridorKey.OUTPUT_TOOLTIPS) == 4
    assert CorridorKey.DESCRIPTION


def test_processor_reports_progress_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        def process_frame(self, **_kwargs):
            return {
                "fg": np.zeros((4, 4, 3), dtype=np.float32),
                "matte": np.zeros((4, 4, 1), dtype=np.float32),
                "processed": np.zeros((4, 4, 3), dtype=np.float32),
                "comp": np.zeros((4, 4, 3), dtype=np.float32),
            }

    monkeypatch.setattr("corridor_key.processor.get_cached_engine", lambda device=None: FakeEngine())

    processor = CorridorKeyProcessor()
    messages: list[tuple[str, int, int]] = []
    image = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    mask = torch.zeros((2, 4, 4), dtype=torch.float32)

    processor.refine(
        image=image,
        mask=mask,
        settings=CorridorKeySettings(),
        progress_callback=lambda message, completed, total: messages.append((message, completed, total)),
    )

    assert messages[0] == ("Loading CorridorKey model...", 0, 2)
    assert messages[1] == ("Processing frame 1/2...", 0, 2)
    assert messages[2] == ("Processing frame 2/2...", 1, 2)
    assert messages[-1] == ("CorridorKey complete.", 2, 2)


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
