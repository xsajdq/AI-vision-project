"""Train the acoustic-anomaly autoencoder on synthetic 'normal' machine hum.

Usage:
    python scripts/train_audio.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.audio.dataset import add_anomaly, generate_normal_sound, waveform_to_spectrogram
from server.audio.model import SpectrogramAutoencoder

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "server" / "models_store" / "audio_autoencoder.pt"
SEED = 7
NUM_TRAIN = 500
NUM_VAL = 120
NUM_ANOMALY_CHECK = 100
EPOCHS = 40
BATCH_SIZE = 32


def build_spectrograms(waveforms: list[np.ndarray]) -> torch.Tensor:
    specs = [waveform_to_spectrogram(w) for w in waveforms]
    return torch.stack(specs)  # (N, 1, SPEC_SIZE, SPEC_SIZE)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"Synthesizing {NUM_TRAIN} training and {NUM_VAL} validation sound clips...")
    train_waves = [generate_normal_sound(rng=rng) for _ in range(NUM_TRAIN)]
    val_waves = [generate_normal_sound(rng=rng) for _ in range(NUM_VAL)]
    anomaly_waves = [add_anomaly(generate_normal_sound(rng=rng), rng=rng) for _ in range(NUM_ANOMALY_CHECK)]

    train_tensor = build_spectrograms(train_waves)
    val_tensor = build_spectrograms(val_waves)
    anomaly_tensor = build_spectrograms(anomaly_waves)

    loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = SpectrogramAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        avg_loss = running_loss / len(train_tensor)
        print(f"  epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.5f}")

    model.eval()

    def max_error_scores(tensor: torch.Tensor) -> np.ndarray:
        # Mirrors AudioInspector.inspect: a brief transient fault can be
        # confined to a single spectrogram cell, so the worst single cell -
        # not an average over the worst region - is the anomaly score.
        with torch.no_grad():
            reconstruction = model(tensor)
            error = torch.square(tensor - reconstruction).numpy()
        return np.array([sample_error.max() for sample_error in error])

    val_scores = max_error_scores(val_tensor)
    anomaly_scores = max_error_scores(anomaly_tensor)

    # A percentile of the (right-skewed) validation-score distribution is a
    # more robust cutoff here than mean + k*std.
    threshold = float(np.percentile(val_scores, 99) * 1.15)

    print(f"\nValidation (normal) scores: mean={val_scores.mean():.5f}  std={val_scores.std():.5f}")
    print(f"Anomaly-check scores:       mean={anomaly_scores.mean():.5f}  std={anomaly_scores.std():.5f}")
    print(f"Chosen threshold:           {threshold:.5f}")
    caught = (anomaly_scores > threshold).mean() * 100
    print(f"Synthetic anomalies flagged above threshold: {caught:.1f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, OUTPUT_PATH)
    print(f"\nSaved checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
