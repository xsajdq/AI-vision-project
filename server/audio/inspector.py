import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from .dataset import SAMPLE_RATE, waveform_to_spectrogram
from .model import SpectrogramAutoencoder

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models_store" / "audio_autoencoder.pt"

# A handful of control points for a dark-blue -> teal -> amber -> red
# colormap, so the spectrogram doesn't have to be shown as flat grayscale.
_COLOR_STOPS = np.array(
    [
        [12, 14, 40],
        [37, 84, 124],
        [56, 163, 152],
        [233, 196, 87],
        [214, 64, 64],
    ],
    dtype=np.float32,
)


def _colorize(normalized: np.ndarray) -> np.ndarray:
    positions = np.linspace(0, 1, len(_COLOR_STOPS))
    flat = normalized.flatten()
    r = np.interp(flat, positions, _COLOR_STOPS[:, 0])
    g = np.interp(flat, positions, _COLOR_STOPS[:, 1])
    b = np.interp(flat, positions, _COLOR_STOPS[:, 2])
    rgb = np.stack([r, g, b], axis=-1).reshape(*normalized.shape, 3)
    return rgb


@dataclass
class AudioResult:
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    spectrogram_png_base64: str


class AudioInspector:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained audio model found at {checkpoint_path}. "
                "Run `python scripts/train_audio.py` first."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = SpectrogramAutoencoder()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(checkpoint["threshold"])

    @staticmethod
    def _load_waveform(audio_bytes: bytes) -> np.ndarray:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)  # downmix to mono

        if sample_rate != SAMPLE_RATE:
            # Simple linear resampling; good enough for this offline demo.
            duration = len(data) / sample_rate
            target_len = int(duration * SAMPLE_RATE)
            data = np.interp(
                np.linspace(0, len(data), target_len, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        peak = np.max(np.abs(data)) + 1e-8
        return (data / peak).astype(np.float32)

    def _render(self, spec_norm: np.ndarray, error_map: np.ndarray) -> str:
        base_rgb = _colorize(spec_norm)

        error_norm = error_map - error_map.min()
        error_norm = error_norm / (error_norm.max() + 1e-8)
        alpha = (error_norm ** 0.7)[..., None]

        white_overlay = np.full_like(base_rgb, 255.0)
        blended = base_rgb * (1 - 0.7 * alpha) + white_overlay * (0.7 * alpha)

        # Spectrograms are conventionally drawn with low frequencies at the
        # bottom, so flip vertically before rendering.
        blended = np.flipud(blended)

        img = Image.fromarray(blended.astype(np.uint8), mode="RGB")
        img = img.resize((256, 256), Image.BICUBIC)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def inspect(self, audio_bytes: bytes) -> AudioResult:
        waveform = self._load_waveform(audio_bytes)
        spec = waveform_to_spectrogram(waveform)  # (1, SPEC_SIZE, SPEC_SIZE)

        with torch.no_grad():
            reconstruction = self.model(spec.unsqueeze(0))
            error_map = torch.square(spec - reconstruction.squeeze(0)).squeeze(0).numpy()

        # Unlike surface defects (which cover a patch of pixels), a short
        # transient fault - a knock, a single bearing click - can land on
        # just one or two spectrogram time frames. Averaging over the worst
        # 5% of cells dilutes that signal away, so the single worst cell is
        # used instead: it is exactly what a brief, sharp, out-of-distribution
        # event produces.
        anomaly_score = float(error_map.max())

        spectrogram_png = self._render(spec.squeeze(0).numpy(), error_map)

        return AudioResult(
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            is_anomalous=anomaly_score > self.threshold,
            spectrogram_png_base64=spectrogram_png,
        )
