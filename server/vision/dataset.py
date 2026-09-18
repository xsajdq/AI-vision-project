"""Synthetic surface-texture generator used for both training data and
ready-to-try sample images.

There is no proprietary factory-floor dataset behind this project, so the
"normal" class is a procedurally generated material texture (think brushed
metal / woven fabric) and the "defective" class is the same texture with
scratches, pits and stains painted on top. This keeps the whole pipeline
self-contained and reproducible with a fixed seed.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_SIZE = 128

# A real inspection line looks at one product type with a fixed, stable
# surface pattern - the only thing that legitimately varies between good
# parts is minor phase/alignment jitter and sensor noise. Fixing the grating
# angles/frequencies (instead of drawing new ones per sample) reflects that,
# and lets the autoencoder specialize tightly enough that an actual defect
# clearly stands out from the reconstruction-error noise floor.
_GRATING_ANGLES = (18.0, 82.0, 141.0)
_GRATING_FREQUENCIES = (9.0, 13.0, 16.0)


def _grating(size: int, angle_deg: float, frequency: float, phase: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    xx, yy = np.meshgrid(x, y)
    projected = xx * np.cos(angle) + yy * np.sin(angle)
    return np.sin(2 * np.pi * frequency * projected / size + phase)


def generate_texture(size: int = IMAGE_SIZE, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a grayscale float32 array in [0, 1] resembling a machined surface."""
    rng = rng or np.random.default_rng()

    base = np.zeros((size, size), dtype=np.float64)
    for angle_deg, frequency in zip(_GRATING_ANGLES, _GRATING_FREQUENCIES):
        base += _grating(
            size,
            angle_deg=angle_deg + rng.uniform(-2, 2),
            frequency=frequency,
            phase=rng.uniform(0, 2 * np.pi),
        )
    base /= len(_GRATING_ANGLES)

    # Keep the stochastic component small relative to the periodic pattern:
    # the autoencoder needs a mostly-regular signal to learn confidently, so
    # that a localized defect (much larger than this noise floor) stands out
    # in the reconstruction error instead of getting lost in it.
    noise = rng.normal(loc=0.0, scale=0.03, size=(size, size))
    combined = base * 0.94 + noise

    # A fixed (rather than per-image min/max) normalization keeps overall
    # brightness/contrast consistent across "normal" samples, so the only
    # thing that legitimately varies between them is the small phase/noise
    # jitter above - exactly what the autoencoder should learn to tolerate.
    combined = np.clip((combined + 1.1) / 2.2, 0.0, 1.0)

    img = Image.fromarray((combined * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.asarray(img, dtype=np.float32) / 255.0


def add_defects(texture: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a copy of ``texture`` with scratches / pits / stains added."""
    rng = rng or np.random.default_rng()
    size = texture.shape[0]

    img = Image.fromarray((texture * 255).astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)

    kind = rng.choice(["scratch", "pit", "stain"])

    if kind == "scratch":
        for _ in range(rng.integers(1, 3)):
            x0, y0 = rng.integers(0, size, size=2)
            length = rng.integers(size // 4, size // 2)
            angle = rng.uniform(0, 2 * np.pi)
            x1 = int(np.clip(x0 + length * np.cos(angle), 0, size - 1))
            y1 = int(np.clip(y0 + length * np.sin(angle), 0, size - 1))
            shade = int(rng.uniform(0, 60))
            draw.line([(x0, y0), (x1, y1)], fill=shade, width=int(rng.integers(1, 3)))

    elif kind == "pit":
        for _ in range(rng.integers(2, 6)):
            cx, cy = rng.integers(size // 6, size - size // 6, size=2)
            r = rng.integers(2, 6)
            shade = int(rng.uniform(0, 50))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade)

    else:  # stain
        cx, cy = rng.integers(size // 4, size - size // 4, size=2)
        r = rng.integers(size // 10, size // 6)
        shade = int(rng.uniform(150, 220))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return np.asarray(img, dtype=np.float32) / 255.0


def to_pil(texture: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8), mode="L")
