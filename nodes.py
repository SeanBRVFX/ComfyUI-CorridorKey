from __future__ import annotations

try:
    from .corridor_key import CorridorKeyProcessor, CorridorKeySettings
except ImportError:
    from corridor_key import CorridorKeyProcessor, CorridorKeySettings


class CorridorKey:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "gamma_space": (["sRGB", "Linear"],),
                "feather_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                    },
                ),
                "edge_focus": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "preserve_core": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 0.45,
                        "step": 0.01,
                    },
                ),
                "despill_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "refiner_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.1,
                    },
                ),
                "auto_despeckle": (["On", "Off"],),
                "despeckle_size": (
                    "INT",
                    {
                        "default": 400,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "output_mode": (["Processed", "FG", "Matte", "Comp"],),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("refined_mask", "selected_output", "foreground", "processed", "comp")
    FUNCTION = "run"
    CATEGORY = "CorridorKey"

    def __init__(self) -> None:
        self._processor = CorridorKeyProcessor()

    def run(
        self,
        image,
        mask,
        gamma_space: str,
        feather_radius: int,
        edge_focus: float,
        threshold: float,
        preserve_core: float,
        despill_strength: float,
        refiner_strength: float,
        auto_despeckle: str,
        despeckle_size: int,
        output_mode: str,
    ):
        settings = CorridorKeySettings(
            gamma_space=str(gamma_space),
            feather_radius=int(feather_radius),
            edge_focus=float(edge_focus),
            threshold=float(threshold),
            preserve_core=float(preserve_core),
            despill_strength=float(despill_strength),
            refiner_strength=float(refiner_strength),
            auto_despeckle=str(auto_despeckle),
            despeckle_size=int(despeckle_size),
            output_mode=str(output_mode),
        )
        return self._processor.refine(image=image, mask=mask, settings=settings)
