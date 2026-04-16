from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path: str | Path, image_mode: str, resize: tuple[int, int]) -> np.ndarray:
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image = image.convert(image_mode)
        image = image.resize((resize[1], resize[0]))
        array = np.asarray(image, dtype=np.float64) / 255.0

    flat = array.reshape(-1)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat = flat / norm
    return flat
