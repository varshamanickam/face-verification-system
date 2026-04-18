import numpy as np


def confidence_from_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
	# Cnofidence is based on the absolute margin between the similarity score and the operating threshold.
	#since cosine similarity is in [-1,2], the margin is normalized by dividing by 2 to map values into [0, 1]
	#Largeer margins indicate predictions farther from the decision boundary meaning higher confidence
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
