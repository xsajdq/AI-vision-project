"""Train the visual-inspection autoencoder on synthetic 'good' surfaces only.

Usage:
    python scripts/train_vision.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.vision.dataset import add_defects, generate_texture
from server.vision.model import ConvAutoencoder

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "server" / "models_store" / "vision_autoencoder.pt"
SEED = 42
NUM_TRAIN = 700
NUM_VAL = 150
NUM_DEFECT_CHECK = 100
EPOCHS = 35
BATCH_SIZE = 32


def build_tensor(images: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(images).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(1)  # (N, 1, H, W)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"Generating {NUM_TRAIN} training and {NUM_VAL} validation textures...")
    train_images = [generate_texture(rng=rng) for _ in range(NUM_TRAIN)]
    val_images = [generate_texture(rng=rng) for _ in range(NUM_VAL)]
    defect_images = [add_defects(generate_texture(rng=rng), rng=rng) for _ in range(NUM_DEFECT_CHECK)]

    train_tensor = build_tensor(train_images)
    val_tensor = build_tensor(val_images)
    defect_tensor = build_tensor(defect_images)

    loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = ConvAutoencoder()
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

    def top5_scores(tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            reconstruction = model(tensor)
            error = torch.square(tensor - reconstruction).numpy()
        scores = []
        for sample_error in error:
            flat = np.sort(sample_error.flatten())[::-1]
            top_k = max(1, int(0.05 * flat.size))
            scores.append(flat[:top_k].mean())
        return np.array(scores)

    val_scores = top5_scores(val_tensor)
    defect_scores = top5_scores(defect_tensor)

    # A percentile of the (right-skewed) validation-score distribution is a
    # more robust cutoff here than mean + k*std.
    threshold = float(np.percentile(val_scores, 99) * 1.15)

    print(f"\nValidation (normal) scores:  mean={val_scores.mean():.5f}  std={val_scores.std():.5f}")
    print(f"Defect-check scores:         mean={defect_scores.mean():.5f}  std={defect_scores.std():.5f}")
    print(f"Chosen threshold:            {threshold:.5f}")
    caught = (defect_scores > threshold).mean() * 100
    print(f"Synthetic defects flagged above threshold: {caught:.1f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, OUTPUT_PATH)
    print(f"\nSaved checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
