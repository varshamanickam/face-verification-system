from pathlib import Path

import numpy as np

from src.similarity_metrics import cosine_similarity_vector
from src.validation import read_pairs

from scripts.embedding_generation import build_image_cache, pairs_to_arrays


def score_split(
    split: str,
    pairs_dir: Path,
    image_mode: str,
    resize: tuple[int, int],
    embedding_backend: str = "raw_pixels",
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    cfg = cfg or {}
    pairs = read_pairs(pairs_dir / f"{split}.jsonl", expected_split=split)
    image_cache = build_image_cache(
        pairs=pairs,
        image_mode=image_mode,
        resize=resize,
        embedding_backend=embedding_backend,
        cfg=cfg,
    )
    left_inputs, right_inputs, labels = pairs_to_arrays(pairs, image_cache)

    scores = cosine_similarity_vector(left_inputs, right_inputs)
    if scores.shape[0] != len(pairs):
        raise ValueError(
            f"Score count {scores.shape[0]} does not match pair count {len(pairs)} for split={split}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite scores found for split={split}")

    score_rows = []
    for pair, score in zip(pairs, scores.tolist()):
        score_rows.append(
            {
                "left_path": pair["left_path"],
                "right_path": pair["right_path"],
                "label": int(pair["label"]),
                "score": float(score),
                "split": split,
            }
        )
    return labels, scores, score_rows
