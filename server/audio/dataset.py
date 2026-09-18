"""Synthetic machine-sound generator.

Real predictive-maintenance datasets (bearing rigs, compressors, etc.) are
not something this project can ship, so "normal" operation is modelled as a
steady motor hum (a fundamental frequency plus decaying harmonics and a
little broadband noise) and "anomalous" operation adds the kind of things
that actually show up in acoustic condition monitoring: knocks, periodic
clicking from a worn bearing, or a rise in broadband (grinding) noise.
"""

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16_000
DURATION_SECONDS = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
SPEC_SIZE = 64

_MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=SPEC_SIZE,
)
_TO_DB = torchaudio.transforms.AmplitudeToDB()


def waveform_to_spectrogram(waveform: np.ndarray) -> torch.Tensor:
    """Convert a mono waveform into a fixed-size (1, SPEC_SIZE, SPEC_SIZE)
    log-mel spectrogram tensor normalized to [0, 1]."""
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, samples)
    mel = _MEL_TRANSFORM(tensor)
    mel_db = _TO_DB(mel)  # (1, n_mels, time)

    mel_db = mel_db.unsqueeze(0)  # (1, 1, n_mels, time)
    mel_db = torch.nn.functional.interpolate(
        mel_db, size=(SPEC_SIZE, SPEC_SIZE), mode="bilinear", align_corners=False
    )
    mel_db = mel_db.squeeze(0)  # (1, SPEC_SIZE, SPEC_SIZE)

    min_val, max_val = mel_db.min(), mel_db.max()
    normalized = (mel_db - min_val) / (max_val - min_val + 1e-8)
    return normalized


def generate_normal_sound(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    t = np.linspace(0, DURATION_SECONDS, NUM_SAMPLES, endpoint=False)

    # A real predictive-maintenance model is trained on recordings from one
    # specific machine, whose running speed only drifts a little from clip
    # to clip - not anywhere near this project's full illustrative range.
    # Narrowing it here (instead of drawing wildly different RPMs) keeps the
    # "normal" class tight enough that an actual acoustic fault is what
    # stands out, rather than getting lost in unrelated pitch variation.
    fundamental = rng.uniform(97, 103)  # Hz, stand-in for one motor's RPM
    signal = np.zeros(NUM_SAMPLES)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t + phase)

    # Very slow amplitude drift, as if load fluctuates a little.
    drift = 1.0 + 0.05 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))
    signal *= drift

    signal += rng.normal(scale=0.05, size=NUM_SAMPLES)  # steady background noise
    signal /= np.max(np.abs(signal)) + 1e-8
    return signal.astype(np.float32)


def add_anomaly(signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    t = np.linspace(0, DURATION_SECONDS, NUM_SAMPLES, endpoint=False)
    out = signal.copy()

    kind = rng.choice(["knock", "bearing_click", "grinding"])

    if kind == "knock":
        for _ in range(rng.integers(1, 3)):
            center = rng.uniform(0.2, DURATION_SECONDS - 0.2)
            width = rng.uniform(0.005, 0.02)
            envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
            out += rng.uniform(1.5, 3.0) * envelope

    elif kind == "bearing_click":
        click_rate = rng.uniform(15, 35)  # Hz
        click_positions = np.arange(0, DURATION_SECONDS, 1.0 / click_rate)
        for pos in click_positions:
            width = 0.003
            envelope = np.exp(-0.5 * ((t - pos) / width) ** 2)
            out += rng.uniform(0.6, 1.2) * envelope

    else:  # grinding: burst of broadband high-frequency noise
        start = rng.uniform(0.1, DURATION_SECONDS - 0.6)
        stop = start + rng.uniform(0.3, 0.6)
        mask = (t >= start) & (t <= stop)
        out[mask] += rng.normal(scale=0.9, size=mask.sum())

    out /= np.max(np.abs(out)) + 1e-8
    return out.astype(np.float32)
