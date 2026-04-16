import numpy as np


def confidence_from_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
	# Scores are cosine similarities in [-1, 1], so dividing by 2 maps max margin to 1.
	margins = np.abs(scores - threshold)
	return np.clip(margins / 2.0, 0.0, 1.0)


def add_confidence_to_rows(score_rows: list[dict], threshold: float) -> list[dict]:
	if not score_rows:
		return score_rows
	scores = np.asarray([row["score"] for row in score_rows], dtype=np.float64)
	confidences = confidence_from_margin(scores, threshold)
	out = []
	for row, confidence in zip(score_rows, confidences.tolist()):
		enriched = dict(row)
		enriched["confidence"] = float(confidence)
		out.append(enriched)
	return out


def summarize_confidence(score_rows: list[dict]) -> dict:
	if not score_rows:
		return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

	values = np.asarray([row["confidence"] for row in score_rows], dtype=np.float64)
	return {
		"count": int(values.shape[0]),
		"mean": float(np.mean(values)),
		"min": float(np.min(values)),
		"max": float(np.max(values)),
	}
