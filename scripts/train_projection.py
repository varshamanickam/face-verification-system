import argparse
import json
from pathlib import Path

import numpy as np

from src.learned_projection import (
    binary_cross_entropy,
    preprocess_image,
    project_vectors,
    save_projection_model,
    sigmoid,
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as file:
        return json.load(file)


def read_pairs(split_path: Path, expected_split: str) -> list[dict]:
    pairs = []
    with split_path.open("r") as file:
        for line in file:
            record = json.loads(line)
            if record["split"] != expected_split:
                raise ValueError(
                    f"Pair split mismatch in {split_path}: expected '{expected_split}', got '{record['split']}'"
                )
            pairs.append(record)
    if not pairs:
        raise ValueError(f"No pairs found in {split_path}")
    return pairs


def build_image_cache(pairs: list[dict], image_mode: str, resize: list[int]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    unique_paths = sorted({pair["left_path"] for pair in pairs} | {pair["right_path"] for pair in pairs})
    for image_path in unique_paths:
        cache[image_path] = preprocess_image(Path(image_path), image_mode, resize)
    return cache


def build_training_matrices(
    pairs: list[dict],
    image_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.vstack([image_cache[pair["left_path"]] for pair in pairs])
    right = np.vstack([image_cache[pair["right_path"]] for pair in pairs])
    labels = np.asarray([int(pair["label"]) for pair in pairs], dtype=np.float64)
    return left, right, labels


def train_projection(
    left_inputs: np.ndarray,
    right_inputs: np.ndarray,
    labels: np.ndarray,
    embedding_dim: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    input_dim = int(left_inputs.shape[1])
    weights = rng.normal(loc=0.0, scale=0.01, size=(input_dim, embedding_dim))
    bias = np.zeros(embedding_dim, dtype=np.float64)
    history = []
    logit_scale = float(np.sqrt(max(1, embedding_dim)))

    num_samples = int(labels.shape[0])
    for epoch in range(epochs):
        order = rng.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            batch_index = order[start : start + batch_size]
            left_batch = left_inputs[batch_index]
            right_batch = right_inputs[batch_index]
            label_batch = labels[batch_index]

            left_proj = (left_batch @ weights) + bias
            right_proj = (right_batch @ weights) + bias
            logits = np.sum(left_proj * right_proj, axis=1) / logit_scale
            probs = sigmoid(logits)

            grad_logits = (probs - label_batch) / float(label_batch.shape[0])
            grad_left = (grad_logits[:, np.newaxis] * right_proj) / logit_scale
            grad_right = (grad_logits[:, np.newaxis] * left_proj) / logit_scale

            grad_weights = (left_batch.T @ grad_left) + (right_batch.T @ grad_right) + (weight_decay * weights)
            grad_bias = np.sum(grad_left + grad_right, axis=0)
            grad_weights = np.clip(grad_weights, -1.0, 1.0)
            grad_bias = np.clip(grad_bias, -1.0, 1.0)

            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias

        epoch_left = (left_inputs @ weights) + bias
        epoch_right = (right_inputs @ weights) + bias
        epoch_logits = np.sum(epoch_left * epoch_right, axis=1) / logit_scale
        epoch_probs = sigmoid(epoch_logits)
        epoch_loss = binary_cross_entropy(epoch_probs, labels)
        epoch_accuracy = float(np.mean((epoch_probs >= 0.5) == labels))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "pair_accuracy": epoch_accuracy,
            }
        )

    return weights, bias, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned resize-flatten-projection embedding")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/run2.json"),
        help="Path to learned projection config JSON",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs_dir = Path(cfg["pairs_dir"])
    train_split = cfg.get("train_split", "train")
    learned_cfg = cfg["learned_projection"]
    train_cfg = learned_cfg.get("train", {})
    image_mode = learned_cfg.get("image_mode", "L")
    resize = learned_cfg.get("resize", [32, 32])
    normalize_output = learned_cfg.get("normalize_output", "l2")
    embedding_dim = int(learned_cfg.get("embedding_dim", 128))
    model_path = Path(learned_cfg["model_path"])

    pairs = read_pairs(pairs_dir / f"{train_split}.jsonl", expected_split=train_split)
    image_cache = build_image_cache(pairs, image_mode=image_mode, resize=resize)
    left_inputs, right_inputs, labels = build_training_matrices(pairs, image_cache)

    weights, bias, history = train_projection(
        left_inputs,
        right_inputs,
        labels,
        embedding_dim=embedding_dim,
        learning_rate=float(train_cfg.get("learning_rate", 0.05)),
        epochs=int(train_cfg.get("epochs", 12)),
        batch_size=int(train_cfg.get("batch_size", 128)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        seed=int(cfg.get("seed", 42)),
    )
    save_projection_model(
        model_path,
        weights,
        bias,
        image_mode=image_mode,
        resize=resize,
        normalize_output=normalize_output,
    )

    train_embeddings_left = project_vectors(left_inputs, weights, bias, normalize_output=normalize_output)
    train_embeddings_right = project_vectors(right_inputs, weights, bias, normalize_output=normalize_output)
    if cfg["score_metric"] == "cosine_similarity":
        final_scores = np.sum(train_embeddings_left * train_embeddings_right, axis=1)
    else:
        final_scores = np.sqrt(np.sum((train_embeddings_left - train_embeddings_right) ** 2, axis=1))

    history_path = model_path.with_suffix(".history.json")
    with history_path.open("w") as file:
        json.dump(
            {
                "config_path": str(args.config),
                "model_path": str(model_path),
                "score_metric": cfg["score_metric"],
                "epochs": history,
                "train_score_mean": float(np.mean(final_scores)),
            },
            file,
            indent=2,
        )

    print(f"Trained learned projection model: {model_path}")
    print(f"Wrote training history: {history_path}")
    print(f"Final training loss: {history[-1]['loss']:.6f}")
    print(f"Final training pair accuracy: {history[-1]['pair_accuracy']:.4f}")


if __name__ == "__main__":
    main()
