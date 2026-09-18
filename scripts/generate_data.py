"""Create a handful of ready-to-try sample files (normal + defective/anomalous)
so the app can be tried immediately after installation, without needing a
camera or microphone on hand.

Usage:
    python scripts/generate_data.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.audio.dataset import SAMPLE_RATE, add_anomaly, generate_normal_sound
from server.vision.dataset import add_defects, generate_texture, to_pil

ROOT = Path(__file__).resolve().parent.parent / "data" / "samples"
SEED = 123


def main() -> None:
    rng = np.random.default_rng(SEED)

    vision_normal_dir = ROOT / "vision" / "normal"
    vision_defect_dir = ROOT / "vision" / "defective"
    audio_normal_dir = ROOT / "audio" / "normal"
    audio_anomaly_dir = ROOT / "audio" / "anomalous"
    for directory in (vision_normal_dir, vision_defect_dir, audio_normal_dir, audio_anomaly_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("Generating sample surface images...")
    for i in range(4):
        texture = generate_texture(rng=rng)
        to_pil(texture).save(vision_normal_dir / f"good_surface_{i + 1}.png")

    for i in range(4):
        texture = add_defects(generate_texture(rng=rng), rng=rng)
        to_pil(texture).save(vision_defect_dir / f"defective_surface_{i + 1}.png")

    print("Generating sample machine-sound clips...")
    for i in range(4):
        wave = generate_normal_sound(rng=rng)
        sf.write(audio_normal_dir / f"normal_operation_{i + 1}.wav", wave, SAMPLE_RATE)

    for i in range(4):
        wave = add_anomaly(generate_normal_sound(rng=rng), rng=rng)
        sf.write(audio_anomaly_dir / f"anomalous_operation_{i + 1}.wav", wave, SAMPLE_RATE)

    print(f"\nSample files written under {ROOT}")


if __name__ == "__main__":
    main()
