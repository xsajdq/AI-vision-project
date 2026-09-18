import base64
import io

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from server.vision.dataset import IMAGE_SIZE, add_defects, generate_texture, to_pil
from server.vision.inspector import VisionInspector
from server.vision.model import ConvAutoencoder


def test_generate_texture_shape_and_range():
    rng = np.random.default_rng(0)
    texture = generate_texture(rng=rng)
    assert texture.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert texture.dtype == np.float32
    assert 0.0 <= texture.min() and texture.max() <= 1.0


def test_add_defects_modifies_image():
    rng = np.random.default_rng(1)
    texture = generate_texture(rng=rng)
    defective = add_defects(texture, rng=rng)
    assert defective.shape == texture.shape
    assert not np.array_equal(texture, defective)


def test_autoencoder_preserves_spatial_size():
    model = ConvAutoencoder()
    x = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)
    assert out.shape == x.shape
    assert torch.all((out >= 0) & (out <= 1))


def _train_tiny_vision_checkpoint(tmp_path, rng):
    torch.manual_seed(0)
    train_images = [generate_texture(rng=rng) for _ in range(150)]
    val_images = [generate_texture(rng=rng) for _ in range(40)]
    defect_images = [add_defects(generate_texture(rng=rng), rng=rng) for _ in range(40)]

    def to_tensor(images):
        return torch.from_numpy(np.stack(images).astype(np.float32)).unsqueeze(1)

    train_tensor = to_tensor(train_images)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

    model = ConvAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(14):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()

    def top5(tensor):
        with torch.no_grad():
            error = torch.square(tensor - model(tensor)).numpy()
        return np.array([np.sort(e.flatten())[::-1][: max(1, int(0.05 * e.size))].mean() for e in error])

    val_scores = top5(to_tensor(val_images))
    threshold = float(np.percentile(val_scores, 99) * 1.15)

    checkpoint_path = tmp_path / "vision_autoencoder.pt"
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, checkpoint_path)
    return checkpoint_path, val_images, defect_images


def _image_to_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    to_pil(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_vision_inspector_flags_defects(tmp_path):
    rng = np.random.default_rng(42)
    checkpoint_path, val_images, defect_images = _train_tiny_vision_checkpoint(tmp_path, rng)

    inspector = VisionInspector(checkpoint_path=checkpoint_path)

    normal_result = inspector.inspect(_image_to_bytes(val_images[0]))
    defect_result = inspector.inspect(_image_to_bytes(defect_images[0]))

    assert not normal_result.is_defective
    assert defect_result.anomaly_score > normal_result.anomaly_score
    assert isinstance(normal_result.heatmap_png_base64, str) and len(normal_result.heatmap_png_base64) > 100

    # The base64 payload should decode back into a valid PNG.
    decoded = Image.open(io.BytesIO(base64.b64decode(normal_result.heatmap_png_base64)))
    assert decoded.size == (256, 256)
