from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorridorKeySettings:
    gamma_space: str = "sRGB"
    despill_strength: float = 1.0
    refiner_strength: float = 1.0
    auto_despeckle: str = "On"
    despeckle_size: int = 400

    def __post_init__(self) -> None:
        if self.gamma_space not in {"sRGB", "Linear"}:
            raise ValueError("gamma_space must be 'sRGB' or 'Linear'.")
        if not 0.0 <= self.despill_strength <= 1.0:
            raise ValueError("despill_strength must be between 0.0 and 1.0.")
        if not 0.0 <= self.refiner_strength <= 4.0:
            raise ValueError("refiner_strength must be between 0.0 and 4.0.")
        if self.auto_despeckle not in {"Off", "On"}:
            raise ValueError("auto_despeckle must be 'Off' or 'On'.")
        if not 0 <= self.despeckle_size <= 4096:
            raise ValueError("despeckle_size must be between 0 and 4096.")

    @property
    def input_is_linear(self) -> bool:
        return self.gamma_space == "Linear"

    @property
    def despeckle_enabled(self) -> bool:
        return self.auto_despeckle == "On"
