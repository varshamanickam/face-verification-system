from pathlib import Path

import numpy as np
from PIL import Image

EPSILON = 1e-12


def preprocess_image(image_path: Path, image_mode: str, resize: list[int]) -> np.ndarray:
    resize_h, resize_w = resize
    with Image.open(image_path) as image:
        image = image.convert(image_mode)
        image = image.resize((resize_w, resize_h), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64) / 255.0
    return array.reshape(-1)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + EPSILON)


def project_vectors(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    normalize_output: str | None = None,
) -> np.ndarray:
    embeddings = (inputs @ weights) + bias
    if normalize_output == "l2":
        embeddings = l2_normalize(embeddings)
    return embeddings


def load_projection_model(model_path: Path) -> dict:
    with np.load(model_path, allow_pickle=False) as data:
        return {
            "weights": data["weights"].astype(np.float64),
            "bias": data["bias"].astype(np.float64),
            "input_dim": int(data["input_dim"]),
            "embedding_dim": int(data["embedding_dim"]),
            "image_mode": str(data["image_mode"]),
            "resize": [int(data["resize_h"]), int(data["resize_w"])],
            "normalize_output": str(data["normalize_output"]),
        }


def save_projection_model(
    model_path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    image_mode: str,
    resize: list[int],
    normalize_output: str | None,
) -> None:
    resize_h, resize_w = resize
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_path,
        weights=weights,
        bias=bias,
        input_dim=np.asarray(weights.shape[0], dtype=np.int64),
        embedding_dim=np.asarray(weights.shape[1], dtype=np.int64),
        image_mode=np.asarray(image_mode),
        resize_h=np.asarray(resize_h, dtype=np.int64),
        resize_w=np.asarray(resize_w, dtype=np.int64),
        normalize_output=np.asarray(normalize_output if normalize_output is not None else "none"),
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    losses = -(labels * np.log(probabilities) + ((1.0 - labels) * np.log(1.0 - probabilities)))
    return float(np.mean(losses))
