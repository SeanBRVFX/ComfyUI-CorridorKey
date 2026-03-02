from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorridorKeySettings:
    gamma_space: str = "sRGB"
    feather_radius: int = 4
    edge_focus: float = 1.5
    threshold: float = 0.5
    preserve_core: float = 0.2
    despill_strength: float = 1.0
    refiner_strength: float = 1.0
    auto_despeckle: str = "On"
    despeckle_size: int = 400
    output_mode: str = "Processed"

    def __post_init__(self) -> None:
        if self.gamma_space not in {"sRGB", "Linear"}:
            raise ValueError("gamma_space must be 'sRGB' or 'Linear'.")
        if not 0 <= self.feather_radius <= 64:
            raise ValueError("feather_radius must be between 0 and 64.")
        if not 0.0 <= self.edge_focus <= 4.0:
            raise ValueError("edge_focus must be between 0.0 and 4.0.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if not 0.0 <= self.preserve_core <= 0.45:
            raise ValueError("preserve_core must be between 0.0 and 0.45.")
        if not 0.0 <= self.despill_strength <= 1.0:
            raise ValueError("despill_strength must be between 0.0 and 1.0.")
        if not 0.1 <= self.refiner_strength <= 4.0:
            raise ValueError("refiner_strength must be between 0.1 and 4.0.")
        if self.auto_despeckle not in {"Off", "On"}:
            raise ValueError("auto_despeckle must be 'Off' or 'On'.")
        if not 1 <= self.despeckle_size <= 4096:
            raise ValueError("despeckle_size must be between 1 and 4096.")
        if self.output_mode not in {"FG", "Matte", "Processed", "Comp"}:
            raise ValueError("output_mode must be one of: FG, Matte, Processed, Comp.")

    @property
    def transition_gain(self) -> float:
        return 8.0 + (self.edge_focus * 16.0)

    @property
    def despill_factor(self) -> float:
        return self.despill_strength

    @property
    def despeckle_enabled(self) -> bool:
        return self.auto_despeckle == "On"
