from __future__ import annotations

from typing import Callable

try:
    from .corridor_key import CorridorKeyProcessor, CorridorKeySettings
except ImportError:
    from corridor_key import CorridorKeyProcessor, CorridorKeySettings


def _build_progress_reporter(unique_id: str | None) -> Callable[[str, int, int], None]:
    progress_bar = None
    prompt_server = None

    # Mini title: best-effort ComfyUI progress hooks
    try:
        from comfy.utils import ProgressBar

        progress_bar = ProgressBar(1, node_id=unique_id)
    except Exception:
        progress_bar = None

    if unique_id:
        try:
            from server import PromptServer

            prompt_server = PromptServer.instance
        except Exception:
            prompt_server = None

    def report(message: str, completed: int, total: int) -> None:
        bounded_total = max(int(total), 1)
        bounded_completed = max(0, min(int(completed), bounded_total))

        if progress_bar is not None:
            progress_bar.update_absolute(bounded_completed, total=bounded_total)
        if prompt_server is not None and unique_id:
            prompt_server.send_progress_text(message, unique_id)

        print(f"[CorridorKey] {message}")

    return report


class CorridorKey:
    DESCRIPTION = (
        "Refines a coarse per-frame alpha hint into CorridorKey FG, Matte, Processed, and QC passes. "
        "The alpha hint should be rough, slightly eroded, and soft rather than tightly expanded."
    )
    OUTPUT_TOOLTIPS = (
        "Raw straight foreground color. The model predicts this in sRGB space; convert to linear before manual compositing with the matte.",
        "Raw linear alpha matte.",
        "Linear foreground premultiplied by the linear matte for quick preview and simple downstream export.",
        "QC preview composite over a checkerboard in sRGB.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input RGB image or image batch to refine.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Coarse Alpha Hint for the current frame. Keep it rough, soft, and slightly eroded. "
                            "For batched images, provide one matching mask per frame."
                        ),
                    },
                ),
                "gamma_space": (
                    ["sRGB", "Linear"],
                    {
                        "tooltip": "Interpret the incoming image as sRGB or already-linear before the fixed 2048x2048 inference pass.",
                    },
                ),
                "despill_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Green despill amount after inference. 0 disables despill, 1 is the standard default. "
                            "Higher values are stronger and can push heavy spill toward magenta or purple."
                        ),
                    },
                ),
                "refiner_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Scales the learned refiner delta. 1.0 is the standard default behavior.",
                    },
                ),
                "auto_despeckle": (
                    ["On", "Off"],
                    {
                        "tooltip": "Enable connected-component cleanup on the predicted alpha matte.",
                    },
                ),
                "despeckle_size": (
                    "INT",
                    {
                        "default": 400,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Minimum island area in pixels to preserve when auto-despeckle is enabled. 400 is the standard default.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("fg", "matte", "processed", "QC")
    FUNCTION = "run"
    CATEGORY = "CorridorKey"

    def __init__(self) -> None:
        self._processor = CorridorKeyProcessor()

    def run(
        self,
        image,
        mask,
        gamma_space: str,
        despill_strength: float,
        refiner_strength: float,
        auto_despeckle: str,
        despeckle_size: int,
        unique_id: str | None = None,
    ):
        settings = CorridorKeySettings(
            gamma_space=str(gamma_space),
            despill_strength=float(despill_strength),
            refiner_strength=float(refiner_strength),
            auto_despeckle=str(auto_despeckle),
            despeckle_size=int(despeckle_size),
        )
        progress_callback = _build_progress_reporter(unique_id)
        return self._processor.refine(
            image=image,
            mask=mask,
            settings=settings,
            progress_callback=progress_callback,
        )
