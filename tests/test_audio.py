import base64
import io

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from server.audio.dataset import (
    NUM_SAMPLES,
    SAMPLE_RATE,
    SPEC_SIZE,
    add_anomaly,
    generate_normal_sound,
    waveform_to_spectrogram,
)
from server.audio.inspector import AudioInspector
from server.audio.model import SpectrogramAutoencoder


def test_generate_normal_sound_shape_and_range():
    rng = np.random.default_rng(0)
    wave = generate_normal_sound(rng=rng)
    assert wave.shape == (NUM_SAMPLES,)
    assert np.max(np.abs(wave)) <= 1.0 + 1e-6


def test_add_anomaly_changes_waveform():
    rng = np.random.default_rng(1)
    wave = generate_normal_sound(rng=rng)
    anomalous = add_anomaly(wave, rng=rng)
    assert anomalous.shape == wave.shape
    assert not np.array_equal(wave, anomalous)


def test_spectrogram_shape():
    rng = np.random.default_rng(2)
    wave = generate_normal_sound(rng=rng)
    spec = waveform_to_spectrogram(wave)
    assert spec.shape == (1, SPEC_SIZE, SPEC_SIZE)
    assert torch.all((spec >= 0) & (spec <= 1))


def _wave_to_wav_bytes(wave: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wave, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def _train_tiny_audio_checkpoint(tmp_path, rng):
    torch.manual_seed(0)
    train_waves = [generate_normal_sound(rng=rng) for _ in range(300)]
    val_waves = [generate_normal_sound(rng=rng) for _ in range(30)]
    anomaly_waves = [add_anomaly(generate_normal_sound(rng=rng), rng=rng) for _ in range(30)]

    def to_tensor(waves):
        return torch.stack([waveform_to_spectrogram(w) for w in waves])

    train_tensor = to_tensor(train_waves)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

    model = SpectrogramAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    # The max-error anomaly score (see AudioInspector.inspect) needs the
    # model to reconstruct normal spectrograms tightly almost everywhere,
    # which takes more epochs than a loosely-fit average-error metric would.
    for _ in range(35):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()

    def max_error(tensor):
        with torch.no_grad():
            error = torch.square(tensor - model(tensor)).numpy()
        return np.array([e.max() for e in error])

    val_scores = max_error(to_tensor(val_waves))
    threshold = float(np.percentile(val_scores, 99) * 1.15)

    checkpoint_path = tmp_path / "audio_autoencoder.pt"
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, checkpoint_path)
    return checkpoint_path, val_waves, anomaly_waves


def test_audio_inspector_flags_anomalies(tmp_path):
    rng = np.random.default_rng(99)
    checkpoint_path, val_waves, anomaly_waves = _train_tiny_audio_checkpoint(tmp_path, rng)

    inspector = AudioInspector(checkpoint_path=checkpoint_path)

    normal_result = inspector.inspect(_wave_to_wav_bytes(val_waves[0]))
    assert not normal_result.is_anomalous

    # Anomaly kinds are drawn randomly (knock / bearing_click / grinding), so
    # check several instead of asserting on a single sample to avoid
    # flakiness from any one hard-to-catch case.
    anomaly_flags = [inspector.inspect(_wave_to_wav_bytes(w)).is_anomalous for w in anomaly_waves[:8]]
    assert sum(anomaly_flags) >= 6

    decoded = Image.open(io.BytesIO(base64.b64decode(normal_result.spectrogram_png_base64)))
    assert decoded.size == (256, 256)
