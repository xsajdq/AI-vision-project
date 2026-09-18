import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import IMAGE_SIZE
from .model import ConvAutoencoder

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models_store" / "vision_autoencoder.pt"


@dataclass
class VisionResult:
    anomaly_score: float
    threshold: float
    is_defective: bool
    heatmap_png_base64: str


class VisionInspector:
    """Loads the trained autoencoder once and serves inspection requests."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained vision model found at {checkpoint_path}. "
                "Run `python scripts/train_vision.py` first."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = ConvAutoencoder()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(checkpoint["threshold"])

    @staticmethod
    def _preprocess(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor, image

    @staticmethod
    def _make_heatmap(original: Image.Image, error_map: np.ndarray) -> str:
        normalized = error_map - error_map.min()
        denom = normalized.max() + 1e-8
        normalized = np.clip(normalized / denom, 0, 1)

        base = original.convert("RGB")
        base_arr = np.asarray(base, dtype=np.float32)

        red_overlay = np.zeros_like(base_arr)
        red_overlay[..., 0] = 255
        alpha = (normalized ** 0.7)[..., None]  # emphasize mid/high errors

        blended = base_arr * (1 - 0.65 * alpha) + red_overlay * (0.65 * alpha)
        blended_img = Image.fromarray(blended.astype(np.uint8), mode="RGB")
        blended_img = blended_img.resize((256, 256), Image.NEAREST)

        buffer = io.BytesIO()
        blended_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def inspect(self, image_bytes: bytes) -> VisionResult:
        tensor, original = self._preprocess(image_bytes)
        with torch.no_grad():
            reconstruction = self.model(tensor)
            error_map = torch.square(tensor - reconstruction).squeeze().numpy()

        # Robust score: mean of the worst 5% of pixels, so a single hot pixel
        # doesn't dominate but a localized defect region still stands out.
        flat_sorted = np.sort(error_map.flatten())[::-1]
        top_k = max(1, int(0.05 * flat_sorted.size))
        anomaly_score = float(flat_sorted[:top_k].mean())

        heatmap = self._make_heatmap(original, error_map)

        return VisionResult(
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            is_defective=anomaly_score > self.threshold,
            heatmap_png_base64=heatmap,
        )
