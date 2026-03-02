from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from . import color_utils as cu
from .model_transformer import GreenFormer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_NAME = "CorridorKey.pth"
CHECKPOINT_DOWNLOAD_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
_ENGINE_CACHE: dict[tuple[str, str, int, bool], "CorridorKeyEngine"] = {}


def _import_cv2():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for CorridorKey. Install requirements.txt in the ComfyUI Python environment."
        ) from exc
    return cv2


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _prefer_channels_last() -> bool:
    return _parse_bool_env("CORRIDORKEY_PREFER_CHANNELS_LAST", True)


def _enable_tf32() -> bool:
    return _parse_bool_env("CORRIDORKEY_ENABLE_TF32", True)


def _configure_torch_for_inference(device_type: str) -> None:
    if device_type != "cuda":
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = _enable_tf32()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = _enable_tf32()


def resolve_checkpoint_path() -> Path:
    explicit = MODELS_DIR / DEFAULT_CHECKPOINT_NAME
    if explicit.is_file():
        return explicit

    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"CorridorKey model directory is missing: {MODELS_DIR}. "
            "Create it and place CorridorKey.pth inside. "
            f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
        )

    candidates = sorted(MODELS_DIR.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No CorridorKey model file found in {MODELS_DIR}. "
            "Download the model checkpoint and place it there. "
            f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
        )
    return candidates[0]


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        img_size: int = 2048,
        use_refiner: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = img_size
        self.checkpoint_path = Path(checkpoint_path)
        self.use_refiner = use_refiner
        self.channels_last = self.device.type == "cuda" and _prefer_channels_last()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        _configure_torch_for_inference(self.device.type)
        self.model = self._load_model()

    def _load_model(self) -> GreenFormer:
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
        )
        model = model.to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.eval()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CorridorKey model file not found: {self.checkpoint_path}. "
                "Place CorridorKey.pth in the local models directory. "
                f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
            )

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_checkpoint(state_dict)
        return model

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        cv2 = _import_cv2()

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        elif mask_linear.dtype == np.uint16:
            mask_linear = mask_linear.astype(np.float32) / 65535.0
        else:
            mask_linear = mask_linear.astype(np.float32)

        height, width = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        if input_is_linear:
            img_resized_linear = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_linear)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        img_norm = (img_resized - self.mean) / self.std
        input_np = np.concatenate([img_norm, mask_resized], axis=-1)
        input_tensor = torch.from_numpy(input_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        if self.channels_last:
            input_tensor = input_tensor.to(memory_format=torch.channels_last)

        hook_handle = None
        if refiner_scale != 1.0 and self.model.refiner is not None:
            def scale_hook(_module, _inputs, output):
                return output * refiner_scale

            hook_handle = self.model.refiner.register_forward_hook(scale_hook)

        with self._autocast_context():
            output = self.model(input_tensor)

        if hook_handle is not None:
            hook_handle.remove()

        pred_alpha = output["alpha"]
        pred_fg = output["fg"]

        resized_alpha = pred_alpha[0].permute(1, 2, 0).cpu().numpy()
        resized_fg = pred_fg[0].permute(1, 2, 0).cpu().numpy()
        resized_alpha = cv2.resize(resized_alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized_fg = cv2.resize(resized_fg, (width, height), interpolation=cv2.INTER_LANCZOS4)

        if resized_alpha.ndim == 2:
            resized_alpha = resized_alpha[:, :, np.newaxis]

        if auto_despeckle:
            processed_alpha = cu.clean_matte(resized_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = resized_alpha

        fg_despilled = cu.despill(resized_fg, green_limit_mode="average", strength=despill_strength)
        fg_despilled_linear = cu.srgb_to_linear(fg_despilled)
        fg_premul_linear = cu.premultiply(fg_despilled_linear, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_linear, processed_alpha], axis=-1)

        checkerboard_srgb = cu.create_checkerboard(width, height, checker_size=128, color1=0.15, color2=0.55)
        checkerboard_linear = cu.srgb_to_linear(checkerboard_srgb)
        if fg_is_straight:
            comp_linear = cu.composite_straight(fg_despilled_linear, checkerboard_linear, processed_alpha)
        else:
            comp_linear = cu.composite_premul(fg_despilled_linear, checkerboard_linear, processed_alpha)
        comp_srgb = cu.linear_to_srgb(comp_linear)

        return {
            "fg": np.clip(resized_fg.astype(np.float32), 0.0, 1.0),
            "raw_alpha": np.clip(resized_alpha.astype(np.float32), 0.0, 1.0),
            "matte": np.clip(processed_alpha.astype(np.float32), 0.0, 1.0),
            "processed": np.clip(fg_premul_linear.astype(np.float32), 0.0, 1.0),
            "processed_rgba": np.clip(processed_rgba.astype(np.float32), 0.0, 1.0),
            "comp": np.clip(comp_srgb.astype(np.float32), 0.0, 1.0),
        }


def get_cached_engine(device: str | None = None, img_size: int = 2048, use_refiner: bool = True) -> CorridorKeyEngine:
    checkpoint_path = resolve_checkpoint_path()
    cache_key = (str(checkpoint_path), device or "", img_size, use_refiner)
    cached = _ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    engine = CorridorKeyEngine(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
        use_refiner=use_refiner,
    )
    _ENGINE_CACHE[cache_key] = engine
    return engine
